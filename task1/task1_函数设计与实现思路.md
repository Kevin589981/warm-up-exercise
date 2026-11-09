# Task1 函数设计与实现思路（非常详细版）

## 1. 任务目标与约束

本任务实现“基于机器学习的文本分类”，核心约束如下：

1. 数据使用 `new_train.tsv`、`new_test.tsv`。
2. 训练时不能使用测试集。
3. 采用 mini-batch 训练。
4. 允许使用 PyTorch 的张量和自动求导，但不允许调用 `torch.nn` 中封装好的网络层和损失函数。
5. 对比 Bag of Words 与 N-gram 的效果。
6. 支持实验：不同损失函数、学习率等超参数对结果影响。

因此，代码将采用“手写线性分类器 + 手写损失函数 + 手写优化步骤”的方式完成。

---

## 2. 总体流程设计

程序分为 10 个阶段：

1. 固定随机种子（保证可复现实验）。
2. 读取 TSV 数据。
3. 文本清洗与分词。
4. 将文本转换为哈希特征（BoW/N-gram）。
5. 从训练集切分验证集。
6. 初始化参数（`W`, `b`）。
7. mini-batch 训练（前向、损失、反向、参数更新）。
8. 每个 epoch 在验证集评估准确率与 loss。
9. 在测试集做推理并保存预测。
10. 汇总多组实验结果并绘图。

---

## 3. 函数级设计

下面按“输入 -> 处理 -> 输出 -> 关键细节”的格式说明每个函数。

### 3.1 `set_seed(seed: int) -> None`

#### 输入
- `seed`：随机种子整数。

#### 处理
- 固定 `random`、`numpy`、`torch` 的随机状态。
- 如果使用 GPU，再固定 `cuda` 的随机状态。

#### 输出
- 无返回值。

#### 为什么需要
- 不固定种子时，每次打乱数据、参数初始化、训练过程都会变化，不利于对比实验（例如 BoW vs N-gram）。

---

### 3.2 `load_tsv(path: Path) -> pd.DataFrame`

#### 输入
- `path`：TSV 文件路径。

#### 处理
- 用 `pandas.read_csv` 读取，指定 `sep='\t'`。
- 显式指定两列列名：`text`, `label`。
- 把文本转成字符串，标签转成整型。
- 对空值做基础处理（空文本填空串）。

#### 输出
- 仅包含 `text` 和 `label` 的 DataFrame。

#### 为什么这样设计
- 数据文件无表头，若不指定列名容易在下游混乱。
- 统一 dtype 可以减少训练时的隐式转换问题。

---

### 3.3 `tokenize(text: str) -> list[str]`

#### 输入
- 原始句子字符串。

#### 处理
- 转小写。
- 通过正则切分出词和简单符号（例如英文单词与数字）。

#### 输出
- token 列表。

#### 关键思考
- 任务要求是传统机器学习文本分类，不必须使用复杂 tokenizer。
- 简单 tokenizer 速度快，且对 BoW/N-gram 已足够。

---

### 3.4 `build_ngram_tokens(tokens: list[str], ngram_range: tuple[int, int]) -> list[str]`

#### 输入
- `tokens`：一元 token 序列。
- `ngram_range`：如 `(1,1)` 表示 BoW；`(1,2)` 表示 unigram + bigram。

#### 处理
- 对每个 `n` 从 `min_n` 到 `max_n`：
  - 滑动窗口生成 n-gram。
  - 通过固定连接符拼接（如 `good_movie`）。

#### 输出
- 该句子的 n-gram 字符串列表。

#### 关键思考
- BoW 可看作 `n=1` 特例。
- 统一函数接口后，实验只需改配置而无需改主流程。

---

### 3.5 `hash_features_for_text(text: str, num_features: int, ngram_range: tuple[int, int], binary: bool) -> dict[int, float]`

#### 输入
- `text`：句子。
- `num_features`：哈希空间大小（例如 50000）。
- `ngram_range`：n-gram 设置。
- `binary`：是否二值特征。

#### 处理
1. 调用 `tokenize`。
2. 调用 `build_ngram_tokens`。
3. 对每个 token 做稳定哈希（`md5`），再 `% num_features` 映射到特征维度。
4. 若 `binary=True`，该维度只记 0/1；否则统计词频。

#### 输出
- 稀疏形式字典：`{feature_index: value}`。

#### 为什么用哈希技巧
- 不用构建词典，代码简洁。
- 可直接支持 N-gram，且控制内存上限。
- 哈希冲突会带来噪声，但在基线模型中可接受。

---

### 3.6 `vectorize_corpus_sparse(texts: list[str], ...) -> list[dict[int, float]]`

#### 输入
- 全部文本列表和特征参数。

#### 处理
- 对每条文本调用 `hash_features_for_text`。

#### 输出
- 稀疏样本列表：长度等于样本数，每个元素是一个字典。

#### 关键思考
- 不直接构建全量稠密矩阵，避免内存峰值过高。
- 稀疏格式在 Python 层便于缓存，可重复用于多轮训练。

---

### 3.7 `build_dense_batch(indices: list[int], sparse_features: list[dict[int, float]], num_features: int, device: torch.device) -> torch.Tensor`

#### 输入
- 一个 batch 的样本下标。
- 稀疏特征列表。
- 特征维度。
- 设备（CPU/GPU）。

#### 处理
- 创建形状 `(batch_size, num_features)` 的零矩阵。
- 将稀疏字典中的键值写入对应位置。

#### 输出
- 当前 batch 的稠密特征张量。

#### 为什么这样做
- 训练时矩阵乘法需要稠密输入；按 batch 动态构建比一次性构建全量稠密矩阵更省内存。

---

### 3.8 `linear_forward(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor`

#### 输入
- `X`: `(B, D)`
- `W`: `(D, C)`
- `b`: `(C,)`

#### 处理
- 计算 `logits = X @ W + b`

#### 输出
- `logits`: `(B, C)`

#### 数学意义
- 这是多分类线性模型（Softmax Regression）在 softmax 前的打分。

---

### 3.9 `cross_entropy_manual(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor`

#### 输入
- `logits`: `(B, C)`
- `targets`: `(B,)`，类别编号。

#### 处理
- 使用 `logsumexp` 计算稳定版 softmax 交叉熵：
  - `loss_i = logsumexp(logits_i) - logits_i[y_i]`
- 对 batch 求平均。

#### 输出
- 标量 loss。

#### 为什么手写
- 任务要求不调用 `torch.nn` 的现成损失函数。
- `logsumexp` 数值稳定，避免 `exp` 溢出。

---

### 3.10 `mse_on_softmax_manual(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor`

#### 输入
- 同上。

#### 处理
1. 手写 softmax（`exp(logits - max) / sum(exp)`）。
2. 构建 one-hot 标签矩阵。
3. 计算均方误差。

#### 输出
- 标量 loss。

#### 设计目的
- 用于“不同损失函数对性能影响”的实验对照。

---

### 3.11 `compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float`

#### 输入
- `logits`, `targets`。

#### 处理
- `argmax` 得到预测类别。
- 比较并计算比例。

#### 输出
- 准确率（0 到 1）。

---

### 3.12 `evaluate_model(...) -> tuple[float, float]`

#### 输入
- 模型参数、验证集索引、稀疏特征等。

#### 处理
- 分批构建 `X`。
- 前向计算。
- 计算 loss 与 accuracy（不反向传播）。

#### 输出
- `(mean_loss, mean_accuracy)`。

#### 注意点
- 验证时用 `torch.no_grad()`。
- 统计时按样本数加权平均。

---

### 3.13 `train_one_experiment(config: dict, data_bundle: dict, device: torch.device) -> dict`

#### 输入
- 单组实验超参数，如：
  - `ngram_range`
  - `num_features`
  - `loss_type`
  - `lr`
  - `epochs`
  - `batch_size`
- 数据包（训练文本、标签、划分索引）。

#### 处理
1. 构建（或读取缓存）稀疏特征。
2. 初始化参数 `W`, `b`（`requires_grad=True`）。
3. epoch 循环：
  - 打乱训练索引。
  - mini-batch 训练。
  - 手动参数更新（SGD）。
  - 每轮记录训练 loss、验证 loss、验证准确率。
4. 保存 best checkpoint（以验证准确率为准）。

#### 输出
- 一个字典，包含：
  - 最佳验证准确率
  - 各轮指标曲线
  - 最佳 `W`, `b`
  - 实验配置

---

### 3.14 `predict_labels(...) -> np.ndarray`

#### 输入
- 训练好的 `W`, `b` 与待预测样本的稀疏特征。

#### 处理
- 分批前向，取 `argmax`。

#### 输出
- 标签数组。

---

### 3.15 `save_predictions(path: Path, preds: np.ndarray) -> None`

#### 输入
- 输出路径与预测数组。

#### 处理
- 保存成 CSV：`id,label`。

#### 输出
- 文件写盘。

---

### 3.16 `plot_histories(all_results: list[dict], out_dir: Path) -> None`

#### 输入
- 多组实验结果。

#### 处理
- 绘制两张图：
  - `val_accuracy.png`
  - `train_val_loss.png`
- 每条曲线对应一组实验。

#### 输出
- 图像文件。

---

### 3.17 `main()`

#### 处理顺序
1. 解析命令行参数。
2. 读取数据、划分验证集。
3. 生成实验配置列表（BoW 与 N-gram，损失函数、学习率组合）。
4. 逐实验训练并收集结果。
5. 选最佳模型在测试集推理。
6. 保存预测、图表、实验汇总 JSON。

#### 最终产物
- `outputs/predictions_best.csv`
- `outputs/metrics_summary.json`
- `outputs/val_accuracy.png`
- `outputs/train_val_loss.png`

---

## 4. 复杂度与资源评估

设：
- 样本数 `N`
- 特征维度 `D`
- 类别数 `C=5`
- 每个样本平均有效特征数 `k`（通常 `k << D`）

每个 batch 前向计算复杂度约为 `O(B * D * C)`（稠密矩阵乘法），构建 batch 的稠密输入约为 `O(B * k)`。

在本任务中，`C` 很小，主要开销在 `B * D`。因此建议：

1. 控制 `D` 在 20k~80k 区间探索。
2. 合理设置 `B`（如 64 或 128）平衡速度与内存。

---

## 5. 实验建议（可直接复现）

建议起步实验矩阵：

1. BoW：`ngram_range=(1,1)`，`loss=ce`，`lr=0.1`
2. BoW：`ngram_range=(1,1)`，`loss=ce`，`lr=0.05`
3. N-gram：`ngram_range=(1,2)`，`loss=ce`，`lr=0.1`
4. N-gram：`ngram_range=(1,2)`，`loss=mse`，`lr=0.10`

观察：

1. N-gram 是否稳定优于 BoW。
2. CE 与 MSE 的收敛速度差异。
3. 学习率过大是否导致震荡。

---

## 6. 结果解读模板

最终报告建议包含：

1. 各实验最佳验证准确率表格。
2. 学习曲线图（loss / accuracy）。
3. 结论：
   - 哪种特征最好。
   - 哪个损失函数更合适。
   - 学习率推荐区间。
4. 失败案例分析（可选）：对错误预测样本进行文本层面分析。

---

## 7. 为什么该实现满足课程要求

1. 使用了 PyTorch 张量与矩阵运算进行 mini-batch 训练。
2. 没有依赖 `torch.nn` 的现成层和损失。
3. 明确对比了 BoW 与 N-gram。
4. 提供不同损失、学习率实验接口。
5. 产出可视化结果，便于实验报告撰写。
