import argparse
import copy
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.KeenKT import KEENKT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, csv_path: str, use_uncertainty_aug: bool, aug_flip_prob: float):
        self.df = pd.read_csv(csv_path)
        self.use_uncertainty_aug = use_uncertainty_aug
        self.aug_flip_prob = aug_flip_prob

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _parse_int_list(text: str):
        return [int(x) for x in str(text).split(",")]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q = self._parse_int_list(row["questions"])
        c = self._parse_int_list(row["concepts"])
        r = self._parse_int_list(row["responses"])
        sm = self._parse_int_list(row["selectmasks"])

        valid = np.array([x == 1 for x in sm], dtype=np.bool_)
        q = np.array(q, dtype=np.int64)
        c = np.array(c, dtype=np.int64)
        r = np.array(r, dtype=np.int64)

        # pad value -1 -> 0; masked positions will not contribute to loss.
        q[q < 0] = 0
        c[c < 0] = 0
        r[r < 0] = 0

        qseqs = q[:-1]
        cseqs = c[:-1]
        rseqs = r[:-1]
        shft_qseqs = q[1:]
        shft_cseqs = c[1:]
        shft_rseqs = r[1:]

        masks = valid[:-1]
        smasks = valid[:-1] & valid[1:]

        item = {
            "qseqs": torch.from_numpy(qseqs),
            "cseqs": torch.from_numpy(cseqs),
            "rseqs": torch.from_numpy(rseqs),
            "shft_qseqs": torch.from_numpy(shft_qseqs),
            "shft_cseqs": torch.from_numpy(shft_cseqs),
            "shft_rseqs": torch.from_numpy(shft_rseqs),
            "masks": torch.from_numpy(masks),
            "smasks": torch.from_numpy(smasks),
        }

        if self.use_uncertainty_aug:
            r_aug = r.copy()
            flip_mask = (np.random.rand(len(r_aug)) < self.aug_flip_prob) & valid
            r_aug[flip_mask] = 1 - r_aug[flip_mask]
            item["r_aug"] = torch.from_numpy(r_aug[:-1])
            item["shft_r_aug"] = torch.from_numpy(r_aug[1:])

        return item


def collate_fn(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def move_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_step(model, batch, optimizer, device):
    batch = move_to_device(batch, device)
    optimizer.zero_grad()
    outputs = model(batch, train=True)
    preds = outputs[0][:, 1:]
    aux_loss = outputs[1] if len(outputs) > 1 else 0.0
    sm = batch["smasks"]
    target = batch["shft_rseqs"]
    bce = binary_cross_entropy(preds[sm].double(), target[sm].double())
    loss = bce + aux_loss
    loss.backward()
    optimizer.step()
    return float(loss.item()), float(bce.item())


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ts = [], []
    for batch in loader:
        batch = move_to_device(batch, device)
        preds = model(batch)[:, 1:]
        sm = batch["smasks"]
        target = batch["shft_rseqs"]
        ys.append(preds[sm].detach().cpu().numpy())
        ts.append(target[sm].detach().cpu().numpy())
    y = np.concatenate(ys)
    t = np.concatenate(ts)
    auc = roc_auc_score(t, y)
    acc = accuracy_score(t, (y >= 0.5).astype(np.int64))
    return float(auc), float(acc)


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.abspath(args.data_dir)
    train_csv = os.path.join(data_dir, "train_valid_sequences.csv")
    test_csv = os.path.join(data_dir, "test_sequences.csv")
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found: {meta_path}. Run preprocess/prepare_assist2009.py first.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    train_ds = SequenceDataset(train_csv, use_uncertainty_aug=bool(args.use_uncertainty_aug), aug_flip_prob=args.aug_flip_prob)
    test_ds = SequenceDataset(test_csv, use_uncertainty_aug=False, aug_flip_prob=0.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = KEENKT(
        n_question=meta["num_c"],
        n_pid=meta["num_q"],
        emb_type=args.emb_type,
        use_uncertainty_aug=bool(args.use_uncertainty_aug),
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        d_model=args.d_model,
        d_ff=args.d_ff,
        final_fc_dim=args.final_fc_dim,
        final_fc_dim2=args.final_fc_dim2,
        seq_len=meta["maxlen"],
        use_diffusion=bool(args.use_diffusion),
        diffusion_weight=args.diffusion_weight,
        noise_level=args.noise_level,
        use_CL=bool(args.use_cl),
        cl_weight=args.cl_weight,
        num_attn_heads=args.num_attn_heads,
        atten_type=args.atten_type,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    os.makedirs(args.save_dir, exist_ok=True)
    best_auc = -1.0
    best_path = os.path.join(args.save_dir, "best_model.pth")
    best_epoch = -1
    best_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, bce_sum = 0.0, 0.0
        for batch in train_loader:
            loss, bce = train_step(model, batch, optimizer, device)
            loss_sum += loss
            bce_sum += bce

        test_auc, test_acc = evaluate(model, test_loader, device)
        avg_loss = loss_sum / max(1, len(train_loader))
        avg_bce = bce_sum / max(1, len(train_loader))
        print(
            f"Epoch {epoch:03d} | train_loss={avg_loss:.6f} | train_bce={avg_bce:.6f} | "
            f"test_auc={test_auc:.4f} | test_acc={test_acc:.4f}",
            flush=True,
        )

        if test_auc > best_auc:
            best_auc = test_auc
            best_epoch = epoch
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())
            tagged_path = os.path.join(args.save_dir, f"best_epoch{epoch:03d}_auc{test_auc:.6f}.pth")
            torch.save(best_state, tagged_path)
            torch.save(best_state, best_path)
            print(f"  Saved best: {tagged_path}", flush=True)
            print(f"  Updated latest best: {best_path}", flush=True)

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint. Please check data and training settings.")

    summary = {
        "best_epoch": best_epoch,
        "best_test_auc": best_auc,
        "best_test_acc": best_acc,
        "best_model_path": best_path,
    }
    summary_path = os.path.join(args.save_dir, "best_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Training done. Best epoch={best_epoch}, test_auc={best_auc:.6f}, test_acc={best_acc:.6f}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple KeenKT trainer for assist2009")
    parser.add_argument("--data_dir", type=str, default=os.path.join(ROOT, "data", "assist2009"))
    parser.add_argument("--save_dir", type=str, default="./saved_model_simple")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--emb_type", type=str, default="stoc_qid")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--final_fc_dim", type=int, default=256)
    parser.add_argument("--final_fc_dim2", type=int, default=256)
    parser.add_argument("--atten_type", type=str, default="w2")

    parser.add_argument("--use_cl", type=int, default=1)
    parser.add_argument("--cl_weight", type=float, default=0.02)
    parser.add_argument("--use_uncertainty_aug", type=int, default=1)
    parser.add_argument("--aug_flip_prob", type=float, default=0.1)
    parser.add_argument("--use_diffusion", type=int, default=1)
    parser.add_argument("--diffusion_weight", type=float, default=0.08)
    parser.add_argument("--noise_level", type=float, default=0.3)

    main(parser.parse_args())
