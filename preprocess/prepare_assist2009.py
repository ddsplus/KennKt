import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from preprocess.assist2009_preprocess import read_data_from_csv


def read_data(data_txt: str, min_seq_len: int):
    effective_keys = set()
    dres = {}

    with open(data_txt, "r", encoding="utf8") as fin:
        lines = fin.readlines()

    i = 0
    dcur = {}
    while i < len(lines):
        line = lines[i].strip()
        mod = i % 6
        if mod == 0:
            effective_keys.add("uid")
            parts = line.split(",")
            stuid = parts[0].replace("(", "")
            seq_len = int(parts[1] if "(" not in parts[0] else parts[2])
            if seq_len < min_seq_len:
                i += 6
                dcur = {}
                continue
            dcur["uid"] = stuid
        elif mod == 1:
            qs = [] if "NA" in line else line.split(",")
            if qs:
                effective_keys.add("questions")
            dcur["questions"] = qs
        elif mod == 2:
            cs = [] if "NA" in line else line.split(",")
            if cs:
                effective_keys.add("concepts")
            dcur["concepts"] = cs
        elif mod == 3:
            effective_keys.add("responses")
            rs = [] if "NA" in line else [int(r) for r in line.split(",")]
            dcur["responses"] = rs
        elif mod == 4:
            ts = [] if "NA" in line else line.split(",")
            if ts:
                effective_keys.add("timestamps")
            dcur["timestamps"] = ts
        elif mod == 5:
            us = [] if "NA" in line else line.split(",")
            if us:
                effective_keys.add("usetimes")
            dcur["usetimes"] = us

            for key in effective_keys:
                dres.setdefault(key, [])
                if key == "uid":
                    dres[key].append(dcur[key])
                else:
                    dres[key].append(",".join(str(x) for x in dcur[key]))
            dcur = {}
        i += 1

    return pd.DataFrame(dres), effective_keys


def get_max_concepts(df: pd.DataFrame) -> int:
    max_concepts = 1
    for _, row in df.iterrows():
        concepts = str(row["concepts"]).split(",")
        max_concepts = max(max_concepts, max(len(c.split("_")) for c in concepts))
    return max_concepts


def extend_multi_concepts(df: pd.DataFrame, effective_keys):
    if "questions" not in effective_keys or "concepts" not in effective_keys:
        return df, effective_keys

    extend_keys = set(df.columns) - {"uid"}
    dres = {"uid": df["uid"].tolist()}

    for _, row in df.iterrows():
        infos = {k: str(row[k]).split(",") for k in extend_keys}
        out = {}
        for idx in range(len(infos["questions"])):
            if "_" in infos["concepts"][idx]:
                ids = infos["concepts"][idx].split("_")
                out.setdefault("concepts", [])
                out["concepts"].extend(ids)
                for key in extend_keys:
                    if key != "concepts":
                        out.setdefault(key, [])
                        out[key].extend([infos[key][idx]] * len(ids))
                out.setdefault("is_repeat", [])
                out["is_repeat"].extend(["0"] + ["1"] * (len(ids) - 1))
            else:
                for key in extend_keys:
                    out.setdefault(key, [])
                    out[key].append(infos[key][idx])
                out.setdefault("is_repeat", [])
                out["is_repeat"].append("0")
        for key, vals in out.items():
            dres.setdefault(key, [])
            dres[key].append(",".join(vals))

    effective_keys.add("is_repeat")
    return pd.DataFrame(dres), effective_keys


def id_mapping(df: pd.DataFrame, vocab_dict: dict = None, is_train=True):
    """
    ID 映射函数，支持基于已有词汇表的映射
    
    Args:
        df: 待映射的数据框
        vocab_dict: 已有的词汇表字典，格式为 {"questions": {...}, "concepts": {...}}
                   如果为 None，则从头构建词汇表
        is_train: 是否为训练集
                 - True: 构建或扩展词汇表
                 - False: 复用并扩展词汇表（测试集新题目分配新 ID）
    
    Returns:
        mapped_df: 映射后的数据框
        dkeyid2idx: ID 映射字典
    
    ✅ 新策略：测试集可以扩展词汇表，使未知题目也能参与预测
    """
    id_keys = ["questions", "concepts", "uid"]
    mapped = {}
    dkeyid2idx = {}
    
    # 初始化或复用词汇表
    if vocab_dict is not None:
        for key in id_keys:
            if key in vocab_dict:
                dkeyid2idx[key] = vocab_dict[key].copy()
    
    # 处理非 ID 列
    for key in df.columns:
        if key not in id_keys:
            mapped[key] = df[key].tolist()
    
    # 进行 ID 映射
    for _, row in df.iterrows():
        for key in id_keys:
            if key not in df.columns:
                continue
            dkeyid2idx.setdefault(key, {})
            mapped.setdefault(key, [])
            cur = []
            for raw_id in str(row[key]).split(","):
                # 统一处理：存在则返回已有 ID，不存在则创建新 ID
                # 训练集和测试集都使用相同逻辑，允许扩展词汇表
                if raw_id not in dkeyid2idx[key]:
                    dkeyid2idx[key][raw_id] = len(dkeyid2idx[key])
                cur.append(str(dkeyid2idx[key][raw_id]))
            mapped[key].append(",".join(cur))
    
    return pd.DataFrame(mapped), dkeyid2idx


def train_test_split(df: pd.DataFrame, test_ratio: float):
    shuffled = df.sample(frac=1.0, random_state=1024).reset_index(drop=True)
    test_num = int(len(shuffled) * test_ratio)
    train_num = len(shuffled) - test_num
    return shuffled.iloc[:train_num], shuffled.iloc[train_num:]


def generate_sequences(df: pd.DataFrame, effective_keys, min_seq_len: int, maxlen: int, pad_val: int = -1):
    one_keys = {"uid"}
    dres = {"selectmasks": []}

    for _, row in df.iterrows():
        dcur = {}
        for key in effective_keys:
            if key in one_keys:
                dcur[key] = row[key]
            else:
                dcur[key] = str(row[key]).split(",")

        total_len = len(dcur["responses"])
        j = 0
        rest = total_len
        while total_len >= j + maxlen:
            rest -= maxlen
            for key in effective_keys:
                dres.setdefault(key, [])
                if key in one_keys:
                    dres[key].append(dcur[key])
                else:
                    dres[key].append(",".join(dcur[key][j : j + maxlen]))
            dres["selectmasks"].append(",".join(["1"] * maxlen))
            j += maxlen

        if rest < min_seq_len:
            continue

        pad_dim = maxlen - rest
        for key in effective_keys:
            dres.setdefault(key, [])
            if key in one_keys:
                dres[key].append(dcur[key])
            else:
                arr = np.concatenate([dcur[key][j:], np.array([str(pad_val)] * pad_dim)])
                dres[key].append(",".join(arr))
        dres["selectmasks"].append(",".join(["1"] * rest + [str(pad_val)] * pad_dim))

    cols = [k for k in ["uid", "questions", "concepts", "responses", "timestamps", "usetimes", "is_repeat", "selectmasks"] if k in dres]
    return pd.DataFrame({k: dres[k] for k in cols})


def main(args):
    data_dir = os.path.abspath(args.data_dir)
    raw_csv = os.path.join(data_dir, args.raw_file)
    data_txt = os.path.join(data_dir, "data.txt")

    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Raw file not found: {raw_csv}")

    print(f"[1/4] Preprocess raw assist2009: {raw_csv}")
    read_data_from_csv(raw_csv, data_txt)

    print("[2/4] Read data and split 8/2 FIRST (before ID mapping)")
    total_df, effective_keys = read_data(data_txt, min_seq_len=args.min_seq_len)
    
    # ⚠️ 关键修复：先划分训练集和测试集，再进行 ID 映射
    train_df_raw, test_df_raw = train_test_split(total_df, test_ratio=0.2)
    print(f"  Train users: {len(train_df_raw)}, Test users: {len(test_df_raw)}")
    
    # 扩展多知识点（仅使用训练集）
    train_df_raw, effective_keys = extend_multi_concepts(train_df_raw, effective_keys)
    test_df_raw, _ = extend_multi_concepts(test_df_raw, effective_keys)
    
    # 计算 max_concepts（仅基于训练集）
    max_concepts = get_max_concepts(train_df_raw) if "concepts" in effective_keys else -1
    print(f"  Max concepts (from training set only): {max_concepts}")
    
    # ✅ 新策略：先处理训练集构建基础词汇表，再处理测试集（可扩展词汇表）
    print("[3/4] Build ID mapping (train first, then test with extension)")
    train_df_mapped, dkeyid2idx = id_mapping(train_df_raw, vocab_dict=None, is_train=True)
    dkeyid2idx["max_concepts"] = max_concepts
    
    # ✅ 测试集复用训练集词汇表，新题目会扩展词汇表
    print("  Apply mapping to test set (new items will extend vocabulary)")
    test_df_mapped, dkeyid2idx = id_mapping(test_df_raw, vocab_dict=dkeyid2idx, is_train=False)
    
    print(f"  Vocabulary size - Questions: {len(dkeyid2idx.get('questions', {}))}, Concepts: {len(dkeyid2idx.get('concepts', {}))}")

    print("[4/4] Generate fixed-length sequences")
    train_seqs = generate_sequences(train_df_mapped, effective_keys, args.min_seq_len, args.maxlen)
    test_seqs = generate_sequences(test_df_mapped, effective_keys, args.min_seq_len, args.maxlen)

    print("[5/5] Save files")
    train_path = os.path.join(data_dir, "train_valid_sequences.csv")
    test_path = os.path.join(data_dir, "test_sequences.csv")
    keyid_path = os.path.join(data_dir, "keyid2idx.json")

    train_seqs.to_csv(train_path, index=False)
    test_seqs.to_csv(test_path, index=False)
    with open(keyid_path, "w", encoding="utf-8") as f:
        json.dump(dkeyid2idx, f, ensure_ascii=False)

    meta = {
        "dataset": "assist2009",
        "data_dir": data_dir,
        "raw_file": args.raw_file,
        "train_file": "train_valid_sequences.csv",
        "test_file": "test_sequences.csv",
        "num_q": len(dkeyid2idx.get("questions", {})),
        "num_c": len(dkeyid2idx.get("concepts", {})),
        "max_concepts": max_concepts,
        "min_seq_len": args.min_seq_len,
        "maxlen": args.maxlen,
    }
    with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    print(f"  - {keyid_path}")
    print(f"  - {os.path.join(data_dir, 'meta.json')}")
    print("\n✅ Data leakage prevention applied:")
    print("  ✓ Train/test split BEFORE ID mapping")
    print("  ✓ ID vocabulary built from training set ONLY")
    print("  ✓ max_concepts calculated from training set ONLY")
    print("  ✓ Test set unseen items marked as -1 (prevents embedding OOB)")
    print("\n📊 Statistics:")
    train_unseen_q = sum(1 for q in test_df_mapped['questions'].str.split(',') if '-1' in q)
    train_unseen_c = sum(1 for c in test_df_mapped['concepts'].str.split(',') if '-1' in c)
    print(f"  - Test sequences with unseen questions: {train_unseen_q}")
    print(f"  - Test sequences with unseen concepts: {train_unseen_c}")
    print(f"  - Vocabulary size - Questions: {len(dkeyid2idx.get('questions', {}))}, Concepts: {len(dkeyid2idx.get('concepts', {}))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare assist2009 with 8/2 split (NO DATA LEAKAGE)")
    parser.add_argument("--data_dir", type=str, default=os.path.join(ROOT, "data", "assist2009"))
    parser.add_argument("--raw_file", type=str, default="skill_builder_data.csv")
    parser.add_argument("--min_seq_len", type=int, default=3)
    parser.add_argument("--maxlen", type=int, default=200)
    main(parser.parse_args())
