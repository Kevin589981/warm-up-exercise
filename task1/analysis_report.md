# Task-1 基于机器学习的文本分类 — 完整分析报告

---

## 一、实验目的：这个实验想让你学什么？

### 1.1 核心教学目标

本实验的本质是一个 **"从零手写"** 的线性分类器训练流程，重点不是拿到最高准确率，而是让你通过动手实现理解以下链条：

```
原始文本 → 分词 → 特征表示(BoW/N-gram) → 线性变换(Wx+b) → 损失函数 → 梯度下降 → 参数更新 → 评估
```

### 1.2 需要掌握的知识点细节

| 知识模块 | 具体知识点 | 为什么重要 |
|---------|-----------|-----------|
| **文本特征表示** | Bag of Words (词袋模型) | 最基础的文本表示，理解"忽略词序、只看词频"的思路 |
| | N-gram (unigram/bigram/trigram) | 在 BoW 基础上引入局部词序信息，理解特征工程的递进关系 |
| | Hash Trick (哈希技巧) | 用固定大小的哈希桶替代词典，避免 OOV 问题，控制内存 |
| | Binary vs Count 特征 | 理解"有没有这个词"和"这个词出现了几次"的区别 |
| **线性模型** | 公式: `logits = X @ W + b` | 理解线性层的本质就是矩阵乘法+偏置 |
| | Softmax 的数学原理 | `p_i = exp(z_i) / Σ exp(z_j)`，理解概率归一化 |
| | 为什么用 `logsumexp` 稳定计算 | 防止 exp 溢出，这是数值计算的基本功 |
| **损失函数** | 交叉熵 (Cross Entropy) | 分类任务的标准损失，理解 `−log(p_correct)` 的含义 |
| | MSE on Softmax | 对照实验用，让学生体会"为什么分类不用 MSE" |
| **优化器** | 手写 SGD 更新 | `W -= lr * W.grad`，理解梯度下降的每一步 |
| | Mini-batch 训练 | 理解"一次看一个样本"vs"一次看一批样本"的权衡 |
| | L2 正则化 (Weight Decay) | 防止过拟合的基础手段 |
| **模型评估** | 训练集/验证集/测试集划分 | 理解数据泄漏的概念 |
| | 学习曲线 (Loss/Acc vs Epoch) | 诊断过拟合/欠拟合的基本工具 |
| | 最佳模型选择 (Best Checkpoint) | 以验证集指标为准，而非训练集 |
| **PyTorch 基础** | Tensor 操作 | `@`, `torch.zeros`, `torch.randn`, `.to(device)` |
| | 自动求导 | `requires_grad=True`, `.backward()`, `.grad.zero_()` |
| | `torch.no_grad()` | 推理时不构建计算图，节省内存 |

### 1.3 实验设计的核心思路

实验要求你回答 **3 个问题**：

1. **BoW vs N-gram 哪个更好？** → 控制其他变量，只改 `ngram_range`
2. **交叉熵 vs MSE 哪个更适合分类？** → 控制其他变量，只改 `loss_type`
3. **学习率怎么选？** → 控制其他变量，只改 `lr`

这就是科学实验的 **控制变量法** —— 一次只改一个因素，观察结果差异。

---

## 二、你的代码 vs 要求对比分析

### 2.1 已经做好的部分 ✅

| 要求 | 你的实现 | 评价 |
|------|---------|------|
| 用 PyTorch 做矩阵运算 | ✅ `linear_forward` 用 `@` 运算 | 正确 |
| 不调用 `torch.nn` | ✅ 手写 `cross_entropy_manual`, `mse_on_softmax_manual` | 完全符合 |
| Mini-batch 训练 | ✅ `for start in range(0, len(shuffled), cfg.batch_size)` | 正确 |
| BoW 实现 | ✅ `ngram_min=1, ngram_max=1` | 正确 |
| N-gram 实现 | ✅ `build_ngram_tokens` 函数 | 正确 |
| 不同损失函数实验 | ✅ `ce` 和 `mse` 两组 | 正确 |
| 不同学习率实验 | ✅ `lr=0.10` 和 `lr=0.05` | 有，但不够 |
| 可视化 | ✅ `plot_histories` 输出两张图 | 正确 |
| 训练/验证/测试划分 | ✅ `train_test_split` + stratify | 正确 |

### 2.2 代码层面的问题 ⚠️

#### 问题 1: `evaluate_model` 函数签名与实际调用不一致

```python
# 函数签名写的是 List[Dict[int, float]]
def evaluate_model(
    ...
    sparse_features: List[Dict[int, float]],  # ← 这里
    ...
) -> Tuple[float, float]:

# 但 main() 里传入的是 csr_matrix
sparse_train, sparse_test = feature_cache[cache_key]  # 这是 csr_matrix
test_logits_acc = evaluate_model(
    ...
    sparse_features=sparse_test,  # ← 实际传的是 csr_matrix
    ...
)
```

虽然 Python 不做运行时类型检查所以不会报错，但类型标注是错的，影响代码可读性和后续维护。

#### 问题 2: 残留的注释代码

第 134-144 行和第 563-583 行有大段注释掉的旧实现。如果已决定用 CSR 稀疏矩阵方案，应该清理掉。

#### 问题 3: `predict_labels` 仍使用旧接口

```python
def predict_labels(
    ...
    sparse_features: List[Dict[int, float]],  # ← 签名是旧的
    ...
) -> np.ndarray:
    ...
    x = build_dense_batch(batch_idx, sparse_features, ...)  # ← 传入 csr_matrix
```

同问题 1，类型标注与实际传入类型不符。

#### 问题 4: 缺少学习率调度

代码中学习率在整个训练过程中是固定的。对于这种手写 SGD 的实验，缺少 learning rate schedule（如 step decay、cosine annealing）是一个可以扩展的方向，但这不是硬性要求。

---

## 2.3 缺失的对照实验 🔴

这是最关键的部分。你的代码虽然能跑，但 **实验矩阵覆盖不足**，以下是要求你应该做但没做的对照：

### 缺失 1: Binary 特征 vs Count 特征的对比

你的代码有 `binary` 参数（第 32 行），但在 `build_default_experiments` 中 **所有实验都设 `binary=False`**。

要求没有明确说要测 binary，但这是一个经典的对照实验：
- Binary: 只看"有没有这个词"，忽略词频
- Count: 考虑"这个词出现了几次"

**应该补的实验：**
```python
ExperimentConfig(
    name="bow_ce_lr0.10_binary",  # 对比 bow_ce_lr0.10
    ngram_min=1, ngram_max=1,
    loss_type="ce", lr=0.10,
    binary=True,  # ← 改这里
    ...
)
```

### 缺失 2: 学习率实验不够系统

你的学习率实验只有两组：
- `bow_ce_lr0.10`
- `bow_ce_lr0.05`

**应该补的实验（控制变量法要求更细的梯度）：**

| 实验名 | ngram | loss | lr | 目的 |
|--------|-------|------|----|-----|
| `bow_ce_lr0.01` | (1,1) | ce | 0.01 | 测试过小学习率 |
| `bow_ce_lr0.05` | (1,1) | ce | 0.05 | 已有 |
| `bow_ce_lr0.10` | (1,1) | ce | 0.10 | 已有 |
| `bow_ce_lr0.20` | (1,1) | ce | 0.20 | 测试较大学习率 |
| `bow_ce_lr0.50` | (1,1) | ce | 0.50 | 测试大学习率是否震荡 |

这样才能画出"学习率 vs 最终准确率"的曲线。

### 缺失 3: N-gram + MSE 的实验不全

你有：
- `ngram12_mse_lr0.10` ✅

但缺少：
- `bow_mse_lr0.10` ← BoW + MSE 的对照
- `ngram13_mse_lr0.10` ← N-gram(1,3) + MSE 的对照

没有这些对照，你无法回答"是 N-gram 本身更好，还是只是 N-gram+CE 这个组合好"。

### 缺失 4: 不同 num_features 的实验

你的代码默认 `num_features=100000`，但没有测试不同特征维度的影响。

**应该补的实验：**
- `num_features=10000` (过小，可能欠拟合)
- `num_features=50000`
- `num_features=100000` (已有)
- `num_features=200000` (过大，可能过拟合或无提升)

这能回答"特征空间多大才合适"的问题。

### 缺失 5: 不同 Weight Decay 的实验

你设了 `weight_decay=1e-4`，但没测试其他值。

**应该补的实验：**
- `weight_decay=0` (无正则化，作为 baseline)
- `weight_decay=1e-4` (已有)
- `weight_decay=1e-3`
- `weight_decay=1e-2`

这能回答"正则化有没有用"的问题。

### 缺失 6: 没有 BoW vs N-gram 的系统对比图表

你把所有实验画在同一张图上，但缺少一个 **专门回答"BoW vs N-gram"问题** 的汇总表或柱状图。

**建议补充：**
```python
# 生成一个汇总表
summary_table = pd.DataFrame([
    {
        "特征类型": "BoW",
        "ngram_range": "(1,1)",
        "最佳val_acc": bow_result["best_val_acc"],
        "测试集acc": bow_result["test_acc_reference"],
    },
    {
        "特征类型": "N-gram(1,2)",
        "ngram_range": "(1,2)",
        "最佳val_acc": ngram12_result["best_val_acc"],
        "测试集acc": ngram12_result["test_acc_reference"],
    },
    ...
])
```

---

## 三、建议补全的完整实验矩阵

以下是按照"控制变量法"设计的完整实验矩阵：

### 维度 1: 特征表示 (BoW vs N-gram)
| 实验 | ngram_range | 说明 |
|------|-------------|------|
| A1 | (1,1) | BoW baseline |
| A2 | (1,2) | + bigram |
| A3 | (1,3) | + trigram |
| A4 | (1,4) | + 4-gram |

### 维度 2: 损失函数
| 实验 | loss_type | 说明 |
|------|-----------|------|
| B1 | ce | 交叉熵 (标准选择) |
| B2 | mse | MSE on softmax |

### 维度 3: 学习率
| 实验 | lr | 说明 |
|------|----|----|
| C1 | 0.01 | 小 |
| C2 | 0.05 | 中小 |
| C3 | 0.10 | 中 |
| C4 | 0.20 | 中大 |
| C5 | 0.50 | 大 |

### 维度 4: Binary vs Count
| 实验 | binary | 说明 |
|------|--------|------|
| D1 | False | 词频 (已有) |
| D2 | True | 二值 |

### 维度 5: 正则化
| 实验 | weight_decay | 说明 |
|------|-------------|------|
| E1 | 0 | 无正则化 |
| E2 | 1e-4 | 轻度 |
| E3 | 1e-3 | 中度 |
| E4 | 1e-2 | 强 |

### 维度 6: 特征维度
| 实验 | num_features | 说明 |
|------|-------------|------|
| F1 | 10000 | 小 |
| F2 | 50000 | 中 |
| F3 | 100000 | 大 (已有) |

**最小可行实验集（至少应该有这些）：**

| # | 实验名 | ngram | loss | lr | binary | wd | 目的 |
|---|--------|-------|------|----|--------|----|----|
| 1 | baseline | (1,1) | ce | 0.10 | False | 1e-4 | 基线 |
| 2 | +bigram | (1,2) | ce | 0.10 | False | 1e-4 | N-gram 效果 |
| 3 | +trigram | (1,3) | ce | 0.10 | False | 1e-4 | N-gram 效果 |
| 4 | +mse | (1,1) | mse | 0.10 | False | 1e-4 | 损失函数对比 |
| 5 | +lr_small | (1,1) | ce | 0.01 | False | 1e-4 | 学习率影响 |
| 6 | +lr_large | (1,1) | ce | 0.30 | False | 1e-4 | 学习率影响 |
| 7 | +binary | (1,1) | ce | 0.10 | True | 1e-4 | 特征类型对比 |
| 8 | +no_reg | (1,1) | ce | 0.10 | False | 0 | 正则化效果 |

---

## 四、其他改进建议

### 4.1 错误分析 (Error Analysis)

你的代码只报告了准确率数字，但没有做 **错误分析** —— 即模型到底错在哪？

建议在最佳模型确定后，输出：
1. 错误样本的文本内容
2. 预测标签 vs 真实标签
3. 混淆矩阵 (Confusion Matrix)

这能帮你理解模型的弱点（比如可能在 label=2（中性）上表现差，因为中性评论的特征不明显）。

### 4.2 TF-IDF 特征

你的实现只用了 raw count 和 binary。TF-IDF 是另一个经典的文本特征表示方法，可以作为一个额外实验：
- TF = 词频
- IDF = 逆文档频率 (惩罚常见词)
- TF-IDF = TF × IDF

这虽然不是硬性要求，但能展示更完整的特征工程知识。

### 4.3 收敛判断

你的代码用固定 epoch 数 (默认 100)。更优雅的做法是：
- 如果连续 N 个 epoch 验证准确率不再提升，提前停止 (Early Stopping)
- 防止过拟合的同时节省计算时间

---

## 五、总结

| 维度 | 评分 | 说明 |
|------|------|------|
| 代码质量 | ⭐⭐⭐⭐ | 结构清晰，有缓存优化，类型标注需修正 |
| 功能完整性 | ⭐⭐⭐⭐ | 核心功能齐全，有可视化 |
| 实验覆盖度 | ⭐⭐⭐ | 缺少 binary 对照、学习率梯度实验、特征维度实验 |
| 对照实验设计 | ⭐⭐ | 实验矩阵不够系统，难以回答"到底哪个因素影响最大" |
| 错误分析 | ⭐ | 完全缺失 |

**最核心的问题：** 你的代码能跑，但实验设计的科学性不足。一个好的实验报告需要能回答"为什么这个好"，而不仅仅是"这个好"。补全缺失的对照实验，你就能做到这一点。
