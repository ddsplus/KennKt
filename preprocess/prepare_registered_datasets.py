import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from preprocess.assist2009_preprocess import read_data_from_csv as assist2009_to_txt


def _split_8_2(items: List[Tuple[str, list]], seed: int) -> Tuple[List[Tuple[str, list]], List[Tuple[str, list]]]:
    rng = random.Random(seed)
    copied = items[:]
    rng.shuffle(copied)
    cut = int(len(copied) * 0.8)
    return copied[:cut], copied[cut:]


def _id_map_and_build_sequences(
    train_user_seqs: List[Tuple[str, list]],
    test_user_seqs: List[Tuple[str, list]],
    maxlen: int,
    min_seq_len: int,
):
    q_vocab, c_vocab = {}, {}

    def _qid(x):
        if x not in q_vocab:
            q_vocab[x] = len(q_vocab)
        return q_vocab[x]

    def _cid(x):
        if x not in c_vocab:
            c_vocab[x] = len(c_vocab)
        return c_vocab[x]

    def _map_one(seq):
        mapped = []
        for q_raw, c_raw, r in seq:
            mapped.append((_qid(q_raw), _cid(c_raw), int(r)))
        return mapped

    train_mapped = [(u, _map_one(seq)) for u, seq in train_user_seqs]
    test_mapped = [(u, _map_one(seq)) for u, seq in test_user_seqs]

    def _to_rows(user_mapped):
        rows = []
        for uid, seq in user_mapped:
            if len(seq) < min_seq_len:
                continue
            q = [x[0] for x in seq]
            c = [x[1] for x in seq]
            r = [x[2] for x in seq]
            start = 0
            while start < len(seq):
                end = min(start + maxlen, len(seq))
                q_part = q[start:end]
                c_part = c[start:end]
                r_part = r[start:end]
                if len(q_part) < min_seq_len:
                    break
                pad = maxlen - len(q_part)
                rows.append(
                    {
                        "uid": uid,
                        "questions": ",".join(map(str, q_part + ([-1] * pad))),
                        "concepts": ",".join(map(str, c_part + ([-1] * pad))),
                        "responses": ",".join(map(str, r_part + ([-1] * pad))),
                        "selectmasks": ",".join(["1"] * len(q_part) + ["-1"] * pad),
                    }
                )
                start += maxlen
        return pd.DataFrame(rows)

    return _to_rows(train_mapped), _to_rows(test_mapped), q_vocab, c_vocab


def _save_outputs(out_dir: str, train_df: pd.DataFrame, test_df: pd.DataFrame, q_vocab: dict, c_vocab: dict, maxlen: int, min_seq_len: int, dataset_name: str):
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train_valid_sequences.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test_sequences.csv"), index=False)
    with open(os.path.join(out_dir, "keyid2idx.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "questions": {k: v for k, v in q_vocab.items()},
                "concepts": {k: v for k, v in c_vocab.items()},
                "max_concepts": 1,
            },
            f,
            ensure_ascii=False,
        )
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "data_dir": out_dir,
                "train_file": "train_valid_sequences.csv",
                "test_file": "test_sequences.csv",
                "num_q": len(q_vocab),
                "num_c": len(c_vocab),
                "max_concepts": 1,
                "min_seq_len": min_seq_len,
                "maxlen": maxlen,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def _prepare_assist2009(data_dir: str, args):
    raw_csv = os.path.join(data_dir, args.raw_file)
    data_txt = os.path.join(data_dir, "data.txt")
    assist2009_to_txt(raw_csv, data_txt)

    users = defaultdict(list)
    with open(data_txt, "r", encoding="utf8") as f:
        lines = [x.strip() for x in f.readlines()]
    i = 0
    while i < len(lines):
        uid = lines[i].split(",")[0].replace("(", "")
        qs = lines[i + 1].split(",")
        cs = lines[i + 2].split(",")
        rs = [int(x) for x in lines[i + 3].split(",")]
        seq = [(q, c, r) for q, c, r in zip(qs, cs, rs)]
        users[uid].extend(seq)
        i += 6

    all_items = list(users.items())
    train_items, test_items = _split_8_2(all_items, args.seed)
    train_df, test_df, q_vocab, c_vocab = _id_map_and_build_sequences(train_items, test_items, args.maxlen, args.min_seq_len)
    _save_outputs(data_dir, train_df, test_df, q_vocab, c_vocab, args.maxlen, args.min_seq_len, "assist2009")


def _prepare_statics2011(data_dir: str, args):
    raw_csv = os.path.join(data_dir, args.raw_file)
    users = defaultdict(list)
    with open(raw_csv, "r", encoding=args.encoding, newline="") as f:
        reader = csv.DictReader(f)
        skill_col = args.skill_col
        for row in reader:
            uid = (row.get("Anon Student Id") or "").strip()
            problem = (row.get("Problem Name") or "").strip()
            step = (row.get("Step Name") or "").strip()
            concept = (row.get(skill_col) or "").strip()
            if not uid or not problem or not step or not concept or concept == ".":
                continue
            first_attempt = (row.get("First Attempt") or "").strip().lower()
            correct = 1 if first_attempt == "correct" else 0
            question = f"{problem}::{step}"
            first_concept = concept.split("~~")[0].strip()
            if not first_concept or first_concept == ".":
                continue
            users[uid].append((question, first_concept, correct))

    all_items = list(users.items())
    train_items, test_items = _split_8_2(all_items, args.seed)
    train_df, test_df, q_vocab, c_vocab = _id_map_and_build_sequences(train_items, test_items, args.maxlen, args.min_seq_len)
    _save_outputs(data_dir, train_df, test_df, q_vocab, c_vocab, args.maxlen, args.min_seq_len, "statics2011")


def _read_xes_user_csv(path: str):
    users = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = str(row["uid"])
            qs = [x.strip() for x in str(row["questions"]).split(",") if x.strip()]
            cs = [x.strip() for x in str(row["concepts"]).split(",") if x.strip()]
            rs = [int(x.strip()) for x in str(row["responses"]).split(",") if x.strip()]
            n = min(len(qs), len(cs), len(rs))
            seq = []
            for i in range(n):
                if int(qs[i]) <= 0 or int(cs[i]) <= 0 or rs[i] < 0:
                    continue
                seq.append((qs[i], cs[i], rs[i]))
            users[uid] = seq
    return list(users.items())


def _prepare_xes3g5m(data_dir: str, args):
    train_csv = os.path.join(data_dir, args.train_file)
    test_csv = os.path.join(data_dir, args.test_file)
    if os.path.exists(train_csv) and os.path.exists(test_csv):
        train_items = _read_xes_user_csv(train_csv)
        test_items = _read_xes_user_csv(test_csv)
    else:
        merged_csv = os.path.join(data_dir, args.raw_file)
        all_items = _read_xes_user_csv(merged_csv)
        train_items, test_items = _split_8_2(all_items, args.seed)

    train_df, test_df, q_vocab, c_vocab = _id_map_and_build_sequences(train_items, test_items, args.maxlen, args.min_seq_len)
    _save_outputs(data_dir, train_df, test_df, q_vocab, c_vocab, args.maxlen, args.min_seq_len, "xes3g5m")


REGISTERED_PREPARE = {
    "assist2009": _prepare_assist2009,
    "assist2017": None,
    "statics2011": _prepare_statics2011,
    "xes3g5m": _prepare_xes3g5m,
}


def _prepare_assist2017(data_dir: str, args):
    raw_csv = os.path.join(data_dir, args.raw_file)
    users = defaultdict(list)
    with open(raw_csv, "r", encoding=args.encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        required = ["studentId", "skill", "problemId", "correct", "action_num"]
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"assist2017 missing required columns: {missing}")

        for row in reader:
            uid = str((row.get("studentId") or "").strip())
            skill = str((row.get("skill") or "").strip())
            problem = str((row.get("problemId") or "").strip())
            correct_text = str((row.get("correct") or "").strip())
            action_num_text = str((row.get("action_num") or "").strip())
            if not uid or not skill or not problem or not correct_text:
                continue
            try:
                correct = int(float(correct_text))
            except ValueError:
                continue
            if correct not in (0, 1):
                continue
            try:
                order_id = int(float(action_num_text))
            except ValueError:
                order_id = 0

            users[uid].append((order_id, problem, skill, correct))

    # Keep the same logic as assist2009: sort within each student by action order.
    user_items = []
    for uid, seq in users.items():
        seq_sorted = sorted(seq, key=lambda x: x[0])
        triplets = [(q, c, r) for _, q, c, r in seq_sorted]
        user_items.append((uid, triplets))

    train_items, test_items = _split_8_2(user_items, args.seed)
    train_df, test_df, q_vocab, c_vocab = _id_map_and_build_sequences(
        train_items, test_items, args.maxlen, args.min_seq_len
    )
    _save_outputs(data_dir, train_df, test_df, q_vocab, c_vocab, args.maxlen, args.min_seq_len, "assist2017")


REGISTERED_PREPARE["assist2017"] = _prepare_assist2017


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare registered datasets for KeenKT simple trainer")
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(REGISTERED_PREPARE.keys()))
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--raw_file", type=str, default="")
    parser.add_argument("--train_file", type=str, default="train.csv")
    parser.add_argument("--test_file", type=str, default="test.csv")
    parser.add_argument("--encoding", type=str, default="utf-8-sig")
    parser.add_argument("--skill_col", type=str, default="KC (F2011)")
    parser.add_argument("--min_seq_len", type=int, default=3)
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    if args.dataset == "assist2009" and not args.raw_file:
        args.raw_file = "skill_builder_data.csv"
    if args.dataset == "assist2017" and not args.raw_file:
        args.raw_file = "anonymized_full_release_competition_dataset.csv"
    if args.dataset == "statics2011" and not args.raw_file:
        args.raw_file = "AllData_student_step_2011F.csv"
    if args.dataset == "xes3g5m" and not args.raw_file:
        args.raw_file = "all.csv"

    REGISTERED_PREPARE[args.dataset](data_dir, args)
    print("done")
    print(f"dataset={args.dataset}")
    print(f"data_dir={data_dir}")
    print("outputs=train_valid_sequences.csv,test_sequences.csv,keyid2idx.json,meta.json")


if __name__ == "__main__":
    main()
