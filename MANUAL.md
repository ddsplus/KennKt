# KeenKT 极简版操作手册

## 1. 目标
本项目已简化为仅支持 `assist2009` 的训练流程，核心特性如下：
- 单数据集：`assist2009`
- 单切分：`8/2`（训练/测试）
- 单训练入口：`train/simple_train.py`
- 每个 epoch 直接输出测试集 `AUC/ACC`
- 自动保存最佳测试集 AUC 对应模型

---

## 2. 目录与脚本说明
- 预处理脚本：[preprocess/prepare_assist2009.py](/E:/desk_file/KennKt/preprocess/prepare_assist2009.py)
- 通用注册预处理入口：[preprocess/prepare_registered_datasets.py](/E:/desk_file/KennKt/preprocess/prepare_registered_datasets.py)
- 训练脚本：[train/simple_train.py](/E:/desk_file/KennKt/train/simple_train.py)
- 模型定义：[models/KeenKT.py](/E:/desk_file/KennKt/models/KeenKT.py)

---

## 3. 环境要求
- Python 3.8+
- 已安装依赖：
  - `torch`
  - `numpy`
  - `pandas`
  - `scikit-learn`

示例安装命令：
```bash
pip install torch numpy pandas scikit-learn
```

---

## 4. 数据准备
将 `assist2009` 原始文件放到数据目录，默认文件名：
- `skill_builder_data.csv`

默认目录示例：
```text
<项目根目录>/data/assist2009/skill_builder_data.csv
```

如果你的文件路径不同，可以用 `--data_dir` 和 `--raw_file` 指定。

---

## 5. 预处理（8/2 划分）
在项目根目录执行：

```bash
python preprocess/prepare_assist2009.py --data_dir ./data/assist2009
```

可选参数：
- `--raw_file`：原始 csv 文件名，默认 `skill_builder_data.csv`
- `--min_seq_len`：最短序列长度，默认 `3`
- `--maxlen`：序列截断/填充长度，默认 `200`

预处理输出文件：
- `train_valid_sequences.csv`
- `test_sequences.csv`
- `keyid2idx.json`
- `meta.json`

## 5.1 注册数据集预处理（推荐）
如果要处理多个数据集，使用统一注册入口：

```bash
python preprocess/prepare_registered_datasets.py --dataset <dataset_name> --data_dir <dataset_dir>
```

当前已注册：
- `assist2009`
- `assist2017`
- `statics2011`
- `xes3g5m`

统一输出同一套文件（供训练脚本直接读取）：
- `train_valid_sequences.csv`
- `test_sequences.csv`
- `keyid2idx.json`
- `meta.json`

### A. assist2009
```bash
python preprocess/prepare_registered_datasets.py --dataset assist2009 --data_dir ./data/assist2009 --raw_file skill_builder_data.csv
```

### B. assist2017
把 `anonymized_full_release_competition_dataset.csv` 放到 `./data/assist2017/` 后执行：
```bash
python preprocess/prepare_registered_datasets.py --dataset assist2017 --data_dir ./data/assist2017 --raw_file anonymized_full_release_competition_dataset.csv --encoding utf-8
```

说明：
- 处理逻辑与 `assist2009` 一致，只是字段映射不同
- 使用字段：`studentId`、`skill`、`problemId`、`correct`、`action_num`
- 每个学生按 `action_num` 排序后构造序列，再做 8/2 切分

### C. statics2011
把 `AllData_student_step_2011F.csv` 放到 `./data/statics2011/` 后执行：
```bash
python preprocess/prepare_registered_datasets.py --dataset statics2011 --data_dir ./data/statics2011 --raw_file AllData_student_step_2011F.csv --skill_col "KC (F2011)" --encoding utf-8-sig
```

说明：
- 默认用 `First Attempt` 转 `responses`（`correct=1`，其他为 `0`）
- `questions` 由 `Problem Name::Step Name` 组成
- `concepts` 取 `KC (F2011)` 的首个 KC

### D. xes3g5m
方式 1：你有官方拆分 `train.csv` + `test.csv`（推荐）
```bash
python preprocess/prepare_registered_datasets.py --dataset xes3g5m --data_dir ./data/xes3g5m --train_file train.csv --test_file test.csv
```

方式 2：你只有单文件 `all.csv`，脚本自动按 8/2 切分
```bash
python preprocess/prepare_registered_datasets.py --dataset xes3g5m --data_dir ./data/xes3g5m --raw_file all.csv
```

`xes3g5m` 输入格式要求（每行一个用户）：
- `uid`
- `questions`（逗号分隔）
- `concepts`（逗号分隔）
- `responses`（逗号分隔）

---

## 6. 训练
在项目根目录执行：

```bash
python train/simple_train.py --data_dir ./data/<dataset_name> --save_dir ./saved_model_simple --epochs 100 --batch_size 64 --learning_rate 1e-4
```

训练日志每轮都会打印：
- `train_loss`
- `train_bce`
- `test_auc`
- `test_acc`

---

## 7. 模型保存规则（最佳测试 AUC）
训练过程中，只要当前 epoch 的 `test_auc` 超过历史最佳，就会保存两份：

1. 带标签版本（保留历史最佳快照）
```text
best_epoch{EPOCH}_auc{AUC}.pth
```
示例：
```text
best_epoch037_auc0.823456.pth
```

2. 固定文件名版本（始终指向当前最佳）
```text
best_model.pth
```

训练结束后还会写一份摘要文件：
```text
best_metrics.json
```
内容包括：
- `best_epoch`
- `best_test_auc`
- `best_test_acc`
- `best_model_path`

---

## 8. 常用参数
`train/simple_train.py` 关键参数：
- `--seed`：随机种子，默认 `3407`
- `--epochs`：训练轮数，默认 `100`
- `--batch_size`：批大小，默认 `64`
- `--learning_rate`：学习率，默认 `1e-4`
- `--weight_decay`：权重衰减，默认 `1e-5`
- `--emb_type`：默认 `stoc_qid`
- `--use_cl`：是否启用对比学习（`0/1`）
- `--cl_weight`：对比损失权重
- `--use_diffusion`：是否启用扩散重建（`0/1`）
- `--diffusion_weight`：扩散损失权重
- `--noise_level`：扩散噪声强度

查看完整参数：
```bash
python train/simple_train.py --help
```

---

## 9. 常见问题
1. 报错 `meta.json not found`
- 原因：还没跑预处理
- 处理：先执行第 5 节命令

2. 报错 `Raw file not found`
- 原因：原始 csv 路径或文件名不对
- 处理：检查 `--data_dir` 和 `--raw_file`

3. 报错 `ModuleNotFoundError: torch`
- 原因：未安装 PyTorch
- 处理：先安装 `torch`

---

## 10. 最小执行流程
只需要两条命令：

```bash
python preprocess/prepare_registered_datasets.py --dataset assist2009 --data_dir ./data/assist2009
python train/simple_train.py --data_dir ./data/assist2009 --save_dir ./saved_model_simple
```
