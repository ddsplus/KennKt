# KeenKT 推理脚本使用说明

## 功能说明

`infer.py` 是一个用于评估训练好的 KeenKT 模型的推理脚本。它可以在测试集上计算模型的 AUC（ROC-AUC）和 ACC（准确率）。

## 使用方法

### 基本用法

```bash
cd /mnt/sda/hds/2026_spring/KeenKT-main/KeenKT_Code

python infer.py \
    --model_path saved_model_simple/assist2009/best_model.pth \
    --dataset_name assist2009
```

### 完整参数说明

```bash
python infer.py \
    --model_path <模型文件路径> \
    --dataset_name <数据集名称> \
    [--data_dir <数据目录>] \
    [--batch_size <批次大小>]
```

#### 必需参数

- `--model_path`: 训练好的模型文件路径（.pth 格式）
  - 示例：`saved_model_simple/assist2009/best_model.pth`
  - 或指定具体 epoch 的模型：`saved_model_simple/assist2009/assist2009_best_20260502_234329_epoch100_auc0.811410.pth`

- `--dataset_name`: 数据集名称
  - 可选值：`assist2009`, `assist2017`, `statics2011`, `xes3g5m`

#### 可选参数

- `--data_dir`: 数据目录路径
  - 默认：自动根据 `dataset_name` 定位到 `data/<dataset_name>`
  - 如果数据在其他位置，可以手动指定

- `--batch_size`: 评估时的批次大小
  - 默认：64
  - 可根据显存大小调整

## 输出示例

```
Using device: cuda
Dataset: assist2009
Meta info: num_q=17751, num_c=149, maxlen=200
Loading test data from: /path/to/data/assist2009/test_sequences.csv
Test samples: 1234

Initializing model with hyperparameters:
  emb_type: stoc_qid
  use_uncertainty_aug: False
  n_blocks: 4
  dropout: 0.2
  d_model: 256
  ...

Loading model from: saved_model_simple/assist2009/best_model.pth
Model loaded successfully!

Evaluating on test set...

============================================================
Results for assist2009:
  AUC: 0.811410
  ACC: 0.756234
============================================================
```

## 注意事项

### 1. 模型超参数

由于 `.pth` 文件只保存了模型的权重（state_dict），不包含超参数信息，因此推理脚本使用以下默认超参数（与训练脚本保持一致）：

```python
{
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
```

**重要**：如果你训练时使用了不同的超参数，需要修改 `infer.py` 中的 `hyperparams` 字典，或者添加命令行参数来指定。

### 2. 数据集准备

在运行推理之前，确保已经完成了数据预处理：

```bash
# 以 assist2009 为例
cd preprocess
python prepare_assist2009.py
```

这会生成必要的文件：
- `train_valid_sequences.csv`
- `test_sequences.csv`
- `meta.json`

### 3. 常见错误

#### 错误 1：找不到 meta.json
```
FileNotFoundError: meta.json not found: /path/to/meta.json
```
**解决**：先运行数据预处理脚本。

#### 错误 2：维度不匹配
```
RuntimeError: size mismatch for ...
```
**解决**：检查训练时使用的超参数是否与推理脚本中的默认值一致。如果不一致，需要修改 `infer.py` 中的 `hyperparams`。

#### 错误 3：找不到测试数据
```
FileNotFoundError: Test sequences file not found: ...
```
**解决**：确认数据目录中存在 `test_sequences.csv` 文件。

## 批量评估多个模型

如果需要评估多个模型，可以使用 shell 脚本：

```bash
#!/bin/bash

MODEL_DIR="saved_model_simple/assist2009"
DATASET="assist2009"

for model in $MODEL_DIR/*.pth; do
    echo "Evaluating: $model"
    python infer.py --model_path "$model" --dataset_name "$DATASET"
    echo ""
done
```

## 进阶：自定义超参数

如果你的模型使用了非默认的超参数，可以修改 `infer.py` 中的 `infer_model` 函数，添加额外的命令行参数：

```python
parser.add_argument("--d_model", type=int, default=256)
parser.add_argument("--n_blocks", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.2)
# ... 其他参数
```

然后在初始化模型时使用这些参数。

## 技术细节

### 评估流程

1. **数据加载**：使用与训练相同的 `SequenceDataset` 类加载测试数据
2. **数据预处理**：
   - 将序列向前移动一位（预测下一个作答）
   - 创建 mask 过滤无效位置
   - Padding 处理不同长度的序列
3. **模型预测**：对每个批次进行前向传播
4. **指标计算**：
   - 只考虑 `smasks=True` 的有效预测位置
   - 计算 ROC-AUC 和 Accuracy

### 关键代码片段

```python
# 预测下一个时间步
preds = model(batch)[:, 1:]

# 获取有效位置的 mask
sm = batch["smasks"]

# 提取有效预测和真实标签
y_pred = preds[sm].cpu().numpy()
y_true = target[sm].cpu().numpy()

# 计算指标
auc = roc_auc_score(y_true, y_pred)
acc = accuracy_score(y_true, (y_pred >= 0.5).astype(np.int64))
```

## 联系与支持

如有问题，请检查：
1. 数据预处理是否正确完成
2. 模型超参数是否与训练时一致
3. 模型文件路径是否正确
