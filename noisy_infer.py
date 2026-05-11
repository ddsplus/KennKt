import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

'''
python noisy_infer.py --model_path <模型路径> --dataset_name <数据集> --noise_strength <噪声强度>
'''

# ROOT 指向 KeenKT_Code 目录（当前脚本所在目录）
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.KeenKT import KEENKT


class SequenceDataset(Dataset):
    """数据集加载器，与训练时保持一致"""
    
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

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

        return item


def collate_fn(batch):
    """批处理函数"""
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def move_to_device(batch, device):
    """将数据移动到指定设备"""
    return {k: v.to(device) for k, v in batch.items()}


def add_noise_to_batch(batch, noise_strength, device):
    """
    对批次数据进行噪声处理：随机置换noise_strength比例的答案
    
    Args:
        batch: 数据批次（包含各种序列）
        noise_strength: 噪声强度（0-1之间，表示置换的答案比例）
        device: 计算设备
        
    Returns:
        noisy_batch: 添加噪声后的批次
    """
    noisy_batch = {}
    
    for k, v in batch.items():
        if k in ["rseqs", "shft_rseqs"]:
            # 对响应序列进行随机置换
            # 响应是0或1的二分类，置换就是翻转 (1 - rseqs)
            noisy_v = v.clone()
            
            # 生成随机掩码，标记要置换的位置
            # noise_strength 表示置换的比例（0-1）
            flip_mask = torch.rand(v.shape, device=device) < noise_strength
            
            # 对选中位置进行置换（0->1, 1->0）
            noisy_v = noisy_v.float()
            noisy_v[flip_mask] = 1.0 - noisy_v[flip_mask]
            noisy_batch[k] = noisy_v.long()
        else:
            # 其他字段（问题、概念、掩码等）保持不变
            noisy_batch[k] = v
    
    return noisy_batch


@torch.no_grad()
def evaluate_with_noise(model, loader, device, noise_strength):
    """
    带噪声的评估函数（随机置换答案）
    
    Args:
        model: KeenKT 模型
        loader: 数据加载器
        device: 计算设备
        noise_strength: 噪声强度（0-1，表示置换的答案比例）
        
    Returns:
        auc: ROC-AUC 分数
        acc: 准确率
    """
    model.eval()
    ys, ts = [], []
    
    for batch in loader:
        batch = move_to_device(batch, device)
        
        # 添加噪声到输入数据
        batch = add_noise_to_batch(batch, noise_strength, device)
        
        preds = model(batch)[:, 1:]  # 预测下一个时间步
        sm = batch["smasks"]
        target = batch["shft_rseqs"]
        
        ys.append(preds[sm].detach().cpu().numpy())
        ts.append(target[sm].detach().cpu().numpy())
    
    y = np.concatenate(ys)
    t = np.concatenate(ts)
    
    auc = roc_auc_score(t, y)
    acc = accuracy_score(t, (y >= 0.5).astype(np.int64))
    
    return float(auc), float(acc)


def noisy_infer_model(model_path, dataset_name, noise_strength=0.1, data_dir=None, batch_size=64):
    """
    带噪声的主推理函数
    
    Args:
        model_path: 模型文件路径 (.pth)
        dataset_name: 数据集名称 (assist2009, assist2017, statics2011, xes3g5m)
        noise_strength: 噪声强度（0-1之间，表示随机置换的答案比例，默认0.1）
        data_dir: 数据目录路径（可选，默认自动定位）
        batch_size: 批次大小
        
    Returns:
        dict: 包含 auc 和 acc 的字典
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Answer flip ratio: {noise_strength}")
    
    # 确定数据目录
    if data_dir is None:
        # 默认数据目录在 KeenKT_Code/data/<dataset_name>
        data_dir = os.path.join(ROOT, "data", dataset_name)
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        # 尝试其他可能的路径
        alt_data_dir = os.path.join(os.path.dirname(ROOT), "data", dataset_name)
        if os.path.exists(alt_data_dir):
            data_dir = alt_data_dir
            print(f"Warning: Using alternative data directory: {data_dir}")
        else:
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # 加载元信息
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found: {meta_path}. Please run preprocess first.")
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    print(f"Dataset: {dataset_name}")
    print(f"Meta info: num_q={meta['num_q']}, num_c={meta['num_c']}, maxlen={meta['maxlen']}")
    
    # 准备测试数据
    test_csv = os.path.join(data_dir, "test_sequences.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test sequences file not found: {test_csv}")
    
    print(f"Loading test data from: {test_csv}")
    test_ds = SequenceDataset(test_csv)
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=collate_fn
    )
    print(f"Test samples: {len(test_ds)}")
    
    # 默认超参数（与训练脚本保持一致）
    hyperparams = {
        "emb_type": "stoc_qid",
        "use_uncertainty_aug": False,
        "n_blocks": 4,
        "dropout": 0.2,
        "d_model": 256,
        "d_ff": 512,
        "final_fc_dim": 256,
        "final_fc_dim2": 256,
        "use_diffusion": True,
        "diffusion_weight": 0.08,
        "noise_level": 0.3,
        "use_CL": True,
        "cl_weight": 0.02,
        "num_attn_heads": 8,
        "atten_type": "w2",
    }
    
    print(f"\nInitializing model with hyperparameters:")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")
    
    # 初始化模型
    model = KEENKT(
        n_question=meta["num_c"],
        n_pid=meta["num_q"],
        emb_type=hyperparams["emb_type"],
        use_uncertainty_aug=hyperparams["use_uncertainty_aug"],
        n_blocks=hyperparams["n_blocks"],
        dropout=hyperparams["dropout"],
        d_model=hyperparams["d_model"],
        d_ff=hyperparams["d_ff"],
        final_fc_dim=hyperparams["final_fc_dim"],
        final_fc_dim2=hyperparams["final_fc_dim2"],
        seq_len=meta["maxlen"],
        use_diffusion=hyperparams["use_diffusion"],
        diffusion_weight=hyperparams["diffusion_weight"],
        noise_level=hyperparams["noise_level"],
        use_CL=hyperparams["use_CL"],
        cl_weight=hyperparams["cl_weight"],
        num_attn_heads=hyperparams["num_attn_heads"],
        atten_type=hyperparams["atten_type"],
    ).to(device)
    
    # 加载模型权重
    print(f"\nLoading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")
    
    # 评估模型（带噪声）
    print("\nEvaluating on test set with answer flipping...")
    auc, acc = evaluate_with_noise(model, test_loader, device, noise_strength)
    
    print("\n" + "="*60)
    print(f"Results for {dataset_name} (answer flip ratio={noise_strength}):")
    print(f"  AUC: {auc:.6f}")
    print(f"  ACC: {acc:.6f}")
    print("="*60)
    
    return {
        "dataset": dataset_name,
        "noise_strength": noise_strength,
        "auc": auc,
        "acc": acc
    }


def main():
    parser = argparse.ArgumentParser(
        description="KeenKT Noisy Inference Script - Evaluate model on test set with answer flipping"
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the trained model file (.pth)"
    )
    
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        required=True,
        choices=["assist2009", "assist2017", "statics2011", "xes3g5m"],
        help="Dataset name"
    )
    
    parser.add_argument(
        "--noise_strength", 
        type=float, 
        default=0.1,
        help="Noise strength (0-1, ratio of answers to flip) to add to input sequences (default: 0.1)"
    )
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=None,
        help="Path to data directory (default: auto-detect based on dataset_name)"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64,
        help="Batch size for evaluation (default: 64)"
    )
    
    args = parser.parse_args()
    
    # 执行推理
    results = noisy_infer_model(
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        noise_strength=args.noise_strength,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    return results


if __name__ == "__main__":
    main()
