# Task-2 基于深度学习的文本分类 — 完整实现方案

---

## 一、实验目的：这个实验想让你学什么？

### 1.1 核心教学目标

与 Task-1 的"从零手写"不同，Task-2 的核心是 **"理解深度学习 NLP 工具链"**。你要掌握：

```
原始文本 → 分词 → 序列编码(整数ID) → Embedding层(稠密向量) → 特征提取器(CNN/RNN/Transformer) → 分类头
```

### 1.2 需要掌握的知识点细节

| 知识模块 | 具体知识点 | 为什么重要 |
|---------|-----------|-----------|
| **Word Embedding** | `torch.nn.Embedding` 的原理 | 理解"离散 token → 连续向量"的映射 |
| | 词表大小 (vocab_size) 和嵌入维度 (embed_dim) | 决定模型容量和内存占用 |
| | 预训练 embedding (GloVe) 的加载方法 | 理解迁移学习在 NLP 中的应用 |
| | padding_idx 的作用 | 处理不同长度句子的关键技术 |
| **序列处理** | Padding (补零) 的原理 | 神经网络需要固定长度输入 |
| | 最大序列长度 (max_len) 的选择 | 太短丢失信息，太长浪费计算 |
| | 句子 → token → ID 序列的转换流程 | 深度学习 NLP 的标准预处理管线 |
| **TextCNN** | Yoon Kim 2014 论文的核心思想 | 用 CNN 提取局部 n-gram 特征 |
| | 多尺度卷积核 (如 [2,3,4]) | 不同大小的卷积核捕获不同长度的 n-gram |
| | 1D 卷积在文本上的应用 | `Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)` |
| | Max-over-time pooling | 从每个卷积核输出中取最大值，保留最强信号 |
| **RNN** | LSTM/GRU 的基本原理 | 处理序列的时序依赖 |
| | `torch.nn.LSTM` 的输入输出格式 | `(seq_len, batch, input_size)` vs `(batch, seq_len, input_size)` |
| | 双向 RNN (bidirectional) | 同时利用前后文信息 |
| | 隐藏状态 (hidden state) 的提取 | 用最后一个时间步的输出作为句子表示 |
| **Transformer** | `torch.nn.TransformerEncoder` 的使用 | 现代 NLP 的基础架构 |
| | Positional Encoding | 给 Transformer 提供位置信息 |
| | `nn.TransformerEncoderLayer` 的参数 | `nhead`, `dim_feedforward`, `dropout` |
| **Dropout** | `torch.nn.Dropout` 的作用 | 防止过拟合 |
| | 在 embedding 后和全连接层前使用 | 标准位置 |
| **GPU 加速** | `.to(device)` 的使用 | 模型、数据都要移到 GPU |
| | `torch.cuda.is_available()` 检查 | 环境适配 |
| **优化器** | Adam vs SGD vs RMSprop | 不同优化器的收敛特性 |
| | 学习率与优化器的配合 | Adam 通常用较小学习率 (0.001) |

---

## 二、总体架构设计

### 2.1 程序流程

```
1. 固定随机种子
2. 读取 TSV 数据
3. 构建词表 (Vocabulary)
   - 分词 → 统计词频 → 分配 ID
   - 特殊 token: <PAD>, <UNK>
4. 句子 → ID 序列 → Padding
5. 划分训练集/验证集/测试集
6. 构建模型 (TextCNN / LSTM / Transformer)
7. 训练 (mini-batch, GPU)
8. 评估 (val_acc, test_acc)
9. 多组实验对比
10. 可视化
```

### 2.2 模块划分

| 模块 | 文件/函数 | 职责 |
|------|----------|------|
| 数据加载 | `load_tsv()` | 读取 TSV |
| 词表构建 | `build_vocab()` | 分词 → 词表 → ID 映射 |
| 序列处理 | `encode_sentences()` | 句子 → padded ID 序列 |
| 数据集 | `TextDataset` (torch.utils.data.Dataset) | 封装 (X, y) 对 |
| DataLoader | `torch.utils.data.DataLoader` | mini-batch 迭代 |
| 模型 - TextCNN | `TextCNN(nn.Module)` | TextCNN 模型 |
| 模型 - LSTM | `TextLSTM(nn.Module)` | 双向 LSTM 模型 |
| 模型 - Transformer | `TextTransformer(nn.Module)` | Transformer 模型 |
| 训练循环 | `train_one_epoch()` | 一个 epoch 的训练 |
| 评估 | `evaluate()` | 在验证/测试集上评估 |
| GloVe 加载 | `load_glove_embeddings()` | 加载预训练词向量 |
| 实验管理 | `build_experiments()` | 定义实验矩阵 |
| 可视化 | `plot_results()` | 绘制对比图表 |

---

## 三、核心实现细节

### 3.1 词表构建

```python
def build_vocab(texts: List[str], min_freq: int = 1, max_size: int = 20000) -> dict:
    """
    构建词表: token → ID 映射
    
    关键设计:
    - 特殊 token: <PAD>=0, <UNK>=1
    - 按词频排序，取 top max_size 个词
    - min_freq 过滤低频词，减少词表噪声
    """
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in counter.most_common(max_size - 2):
        if counter[token] >= min_freq:
            vocab[token] = len(vocab)
    
    return vocab
```

**为什么需要词表：** 深度学习模型不能直接处理文本，需要先将 token 映射为整数 ID，然后通过 Embedding 层转为稠密向量。

### 3.2 序列编码与 Padding

```python
def encode_sentences(texts: List[str], vocab: dict, max_len: int = 100) -> torch.Tensor:
    """
    句子列表 → padded ID 序列
    
    返回: (num_samples, max_len) 的 LongTensor
    """
    sequences = []
    for text in texts:
        tokens = tokenize(text)[:max_len]  # 截断
        ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
        # Padding
        if len(ids) < max_len:
            ids += [vocab["<PAD>"]] * (max_len - len(ids))
        sequences.append(ids)
    
    return torch.tensor(sequences, dtype=torch.long)
```

**关键参数：**
- `max_len=100`：超过截断，不足补 `<PAD>`。需要根据数据分布选择（可以画个句子长度分布图来选）

### 3.3 TextCNN 模型 (Yoon Kim 2014)

```python
class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_filters: int,
        kernel_sizes: List[int],
        num_classes: int,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        
        # 多尺度卷积: 每个 kernel_size 对应一组卷积核
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
            ) for k in kernel_sizes
        ])
        
        # 分类头: 输入 = num_filters * len(kernel_sizes)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        emb = self.dropout(emb)
        
        # Conv1d 需要 (batch, embed_dim, seq_len)
        emb = emb.transpose(1, 2)
        
        # 多尺度卷积 + Max-over-time pooling
        pooled = []
        for conv in self.convs:
            conv_out = F.relu(conv(emb))  # (batch, num_filters, seq_len - k + 1)
            pooled.append(F.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2))
            # max_pool1d 后: (batch, num_filters, 1) → squeeze → (batch, num_filters)
        
        # 拼接所有尺度的特征
        cat = torch.cat(pooled, dim=1)  # (batch, num_filters * len(kernel_sizes))
        cat = self.dropout(cat)
        
        return self.fc(cat)  # (batch, num_classes)
```

**架构图解：**
```
输入: "This movie is great"
  ↓ Embedding (dim=128)
  (4, 128) 矩阵
  ↓ transpose → (128, 4)
  ↓ Conv1d kernel_size=2 → 提取 bigram 特征 → Max Pool → 128-d 向量
  ↓ Conv1d kernel_size=3 → 提取 trigram 特征 → Max Pool → 128-d 向量
  ↓ Conv1d kernel_size=4 → 提取 4-gram 特征 → Max Pool → 128-d 向量
  ↓ Concat → 384-d 向量
  ↓ Dropout(0.5)
  ↓ Linear(384, 5)
  输出: logits (5,)
```

**为什么 TextCNN 有效：** 不同大小的卷积核相当于自动学习不同长度的 n-gram 特征，比 Task-1 中手动设计的 BoW/N-gram 更灵活。

### 3.4 LSTM 模型

```python
class TextLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.5,
        bidirectional: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # 输入格式: (batch, seq_len, embed_dim)
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        lstm_out, (hidden, cell) = self.lstm(emb)
        # lstm_out: (batch, seq_len, hidden_dim * direction_factor)
        # hidden: (num_layers * direction_factor, batch, hidden_dim)
        
        # 取最后一个时间步的输出
        # 双向 LSTM: 拼接前向最后一个和后向第一个
        if self.lstm.bidirectional:
            forward_last = lstm_out[:, -1, :self.lstm.hidden_size]
            backward_first = lstm_out[:, 0, self.lstm.hidden_size:]
            hidden_cat = torch.cat([forward_last, backward_first], dim=1)
        else:
            hidden_cat = lstm_out[:, -1, :]
        
        hidden_cat = self.dropout(hidden_cat)
        return self.fc(hidden_cat)
```

### 3.5 Transformer 模型

```python
class TextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        max_len: int = 100,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        emb = emb + self.pos_encoding[:, :x.size(1), :]  # 加位置编码
        emb = self.dropout(emb)
        
        # 创建 padding mask (True = 忽略)
        pad_mask = (x == 0)
        
        out = self.transformer(emb, src_key_padding_mask=pad_mask)
        # out: (batch, seq_len, embed_dim)
        
        # 用 [CLS] 位置 (第一个非 padding token) 或 mean pooling
        # 这里用 mean pooling (对非 padding 位置求平均)
        mask = (~pad_mask).unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        return self.fc(pooled)
```

### 3.6 GloVe 预训练 Embedding 加载

```python
def load_glove_embeddings(glove_path: str, vocab: dict, embed_dim: int = 100) -> torch.Tensor:
    """
    加载 GloVe 预训练词向量，构建 embedding 矩阵
    
    返回: (vocab_size, embed_dim) 的 Tensor
    未匹配的词用随机初始化
    """
    # 1. 读取 GloVe 文件
    glove_vectors = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            token = parts[0]
            vector = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
            glove_vectors[token] = vector
    
    # 2. 构建 embedding 矩阵
    vocab_size = len(vocab)
    embeddings = torch.randn(vocab_size, embed_dim) * 0.1  # 随机初始化
    
    matched = 0
    for token, idx in vocab.items():
        if token in glove_vectors:
            embeddings[idx] = glove_vectors[token]
            matched += 1
    
    print(f"GloVe 匹配率: {matched}/{vocab_size} ({matched/vocab_size:.1%})")
    return embeddings
```

**在模型中使用：**
```python
# 加载预训练 embedding
glove_matrix = load_glove_embeddings("glove.6B.100d.txt", vocab, embed_dim=100)

# 创建模型
model = TextCNN(vocab_size, embed_dim=100, ...)

# 替换 embedding 层的权重
model.embedding.weight.data.copy_(glove_matrix)

# 可选: 冻结 embedding 层 (不更新)
model.embedding.weight.requires_grad = False
```

### 3.7 训练循环 (GPU 版)

```python
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch_y)
        total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        total_samples += len(batch_y)
    
    return total_loss / total_samples, total_correct / total_samples
```

### 3.8 Dataset 和 DataLoader

```python
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, sequences: torch.Tensor, labels: np.ndarray):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# 使用
train_dataset = TextDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

---

## 四、完整实验矩阵设计

### 4.1 维度 1: 损失函数

| 实验 | loss | 说明 |
|------|------|------|
| L1 | CrossEntropyLoss | 标准选择 |
| L2 | MSELoss (配合 softmax) | 对照实验 |

### 4.2 维度 2: 学习率

| 实验 | lr | 说明 |
|------|----|------|
| LR1 | 0.001 | Adam 默认 |
| LR2 | 0.005 | 中等 |
| LR3 | 0.01 | 较大 |

### 4.3 维度 3: 卷积核配置 (TextCNN)

| 实验 | num_filters | kernel_sizes | 说明 |
|------|-------------|-------------|------|
| C1 | 64 | [2, 3, 4] | 标准配置 |
| C2 | 128 | [2, 3, 4] | 更多卷积核 |
| C3 | 64 | [3, 4, 5] | 更大卷积核 |
| C4 | 64 | [2, 3, 4, 5] | 更多尺度 |

### 4.4 维度 4: 优化器

| 实验 | optimizer | lr | 说明 |
|------|-----------|----|------|
| O1 | Adam | 0.001 | 默认推荐 |
| O2 | SGD + momentum | 0.01 | 传统优化器 |
| O3 | RMSprop | 0.001 | 自适应学习率 |

### 4.5 维度 5: GloVe 预训练

| 实验 | embedding | 说明 |
|------|-----------|------|
| G1 | 随机初始化 | Baseline |
| G2 | GloVe (不冻结) | 预训练 + 微调 |
| G3 | GloVe (冻结) | 预训练 + 固定 |

### 4.6 维度 6: 模型架构

| 实验 | model | 说明 |
|------|-------|------|
| M1 | TextCNN | 卷积网络 |
| M2 | BiLSTM | 双向 LSTM |
| M3 | Transformer | 自注意力 |

### 4.7 最小可行实验集（至少跑这些）

| # | 实验名 | model | lr | optimizer | embedding | 目的 |
|---|--------|-------|----|-----------|-----------|------|
| 1 | cnn_baseline | TextCNN | 0.001 | Adam | 随机 | 基线 |
| 2 | cnn_more_filters | TextCNN | 0.001 | Adam | 随机 | 卷积核数量 |
| 3 | cnn_wide_kernels | TextCNN | 0.001 | Adam | 随机 | 卷积核大小 |
| 4 | cnn_lr_high | TextCNN | 0.01 | Adam | 随机 | 学习率 |
| 5 | cnn_sgd | TextCNN | 0.01 | SGD | 随机 | 优化器 |
| 6 | cnn_glove | TextCNN | 0.001 | Adam | GloVe | 预训练 |
| 7 | lstm_baseline | BiLSTM | 0.001 | Adam | 随机 | 模型对比 |
| 8 | transformer_baseline | Transformer | 0.001 | Adam | 随机 | 模型对比 |

---

## 五、与 Task-1 的关键差异

| 维度 | Task-1 | Task-2 |
|------|--------|--------|
| 特征工程 | 手动 (BoW/N-gram + Hash) | 自动 (Embedding 层学习) |
| 模型 | 手写线性层 (Wx+b) | 调库 (nn.Conv1d, nn.LSTM, nn.Transformer) |
| 损失函数 | 手写 CE/MSE | 调 `nn.CrossEntropyLoss` |
| 优化器 | 手写 SGD | 调 `torch.optim.Adam/SGD/RMSprop` |
| 运行设备 | CPU | GPU (CUDA) |
| 核心能力 | 理解底层数学 | 理解架构设计和调参 |
| 代码量 | 较大 (手写一切) | 较小 (调库为主) |

---

## 六、GPU 使用要点

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 模型移到 GPU
model = model.to(device)

# 2. 数据移到 GPU (在训练循环中)
batch_x = batch_x.to(device)
batch_y = batch_y.to(device)

# 3. 预训练 embedding 也要在 GPU 上
glove_matrix = glove_matrix.to(device)
model.embedding.weight.data.copy_(glove_matrix)
```

---

## 七、可视化设计

### 7.1 学习曲线

```python
def plot_learning_curves(results: List[dict], out_dir: Path):
    """绘制所有实验的 val_acc 曲线"""
    plt.figure(figsize=(12, 6))
    for r in results:
        plt.plot(r["history"]["val_acc"], label=r["config"]["name"])
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.savefig(out_dir / "val_accuracy.png", dpi=180)
```

### 7.2 柱状对比图

```python
def plot_bar_comparison(results: List[dict], out_dir: Path):
    """绘制各实验最佳 val_acc 的柱状图"""
    names = [r["config"]["name"] for r in results]
    val_accs = [r["best_val_acc"] for r in results]
    test_accs = [r["test_acc"] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, val_accs, width, label="Val Acc", color="steelblue")
    ax.bar(x + width/2, test_accs, width, label="Test Acc", color="coral")
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Best Validation & Test Accuracy by Experiment")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "bar_comparison.png", dpi=180)
```

---

## 八、项目文件结构

```
task2/
├── run_task2.py              # 主程序
├── task2_函数设计与实现思路.md  # 设计文档
├── outputs/                   # 输出目录
│   ├── val_accuracy.png       # 学习曲线
│   ├── bar_comparison.png     # 柱状对比
│   ├── metrics_summary.json   # 实验结果
│   └── predictions_best.csv   # 最佳模型预测
└── data/
    └── glove.6B.100d.txt     # GloVe 预训练向量 (可选)
```

---

## 九、关键注意事项

### 9.1 词表大小 vs Embedding 维度

- 词表太大 (如 30000+) → 内存大，训练慢，且低频词学不好
- 词表太小 (如 5000) → OOV (Out-of-Vocabulary) 太多，信息丢失
- 建议: `max_size=15000~20000`, `min_freq=2`

### 9.2 Embedding 维度选择

| embed_dim | 适用场景 |
|-----------|---------|
| 50 | 小词表，快速实验 |
| 100 | 标准 (GloVe 6B) |
| 200 | 中等词表 |
| 300 | GloVe 840B, 最佳质量 |

### 9.3 防止过拟合

- Dropout: TextCNN 用 0.5, Transformer 用 0.1
- Early stopping: 连续 5 个 epoch val_acc 不提升就停止
- 数据增强: 同义词替换 (可选)

### 9.4 序列长度选择

建议先统计数据集的句子长度分布：
```python
lengths = [len(tokenize(text)) for text in texts]
print(f"p50={np.percentile(lengths, 50)}, p90={np.percentile(lengths, 90)}, p99={np.percentile(lengths, 99)}")
```
选择覆盖 90%~95% 句子的长度作为 `max_len`。

---

## 十、预期结果方向

基于 TextCNN 论文和类似任务的经验：

1. **TextCNN 应该显著优于 Task-1 的 BoW 线性模型** — 因为 embedding 能捕获语义相似性
2. **GloVe 预训练应该有帮助** — 尤其在训练数据有限时
3. **BiLSTM 和 TextCNN 性能接近** — 不同任务各有优劣
4. **Transformer 在小数据集上可能不如 CNN/RNN** — 因为需要更多数据才能学到好的注意力模式
5. **Adam 通常优于 SGD** — 除非精心调学习率和 schedule
