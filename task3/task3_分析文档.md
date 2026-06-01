# Task-3 实现 Transformer 的基础结构 — 完整分析

---

## 一、这个实验想让你学什么？

### 1.1 核心教学目标

Task-3 是整个系列中最难的一个。前两个任务是"用模型"，这个任务是 **"造模型"**。你要从零实现 Transformer 的每个组件，理解它为什么能工作。

**与 Task-1/2 的本质区别：**

| 维度 | Task-1/2 | Task-3 |
|------|----------|--------|
| 目标 | 调库完成分类任务 | 手写 Transformer 架构 |
| 数据 | 现成的电影评论数据集 | 自己生成（加法题 + 语料） |
| 训练方式 | 句子级分类（teacher forcing） | 序列到序列 / 自回归生成 |
| 评估方式 | 准确率 | 生成是否正确、perplexity |
| 难度 | 中等 | 高 |

### 1.2 需要掌握的知识点细节

| 知识模块 | 具体知识点 | 为什么重要 |
|---------|-----------|-----------|
| **Multi-Head Attention** | Q, K, V 的计算 | Transformer 的核心机制 |
| | Scaled Dot-Product Attention: `softmax(QK^T / √d_k) V` | 理解为什么除以 √d_k（防止 softmax 饱和） |
| | 多头的作用 | 每个 head 关注不同维度的关系 |
| | attention mask 的应用 | 控制哪些位置可以 attend 到哪些位置 |
| **Positional Encoding** | 为什么需要位置编码 | Attention 本身没有位置信息 |
| | sin/cos 编码 vs 可学习编码 | 原始论文用 sin/cos，实际两者都可以 |
| **残差连接 (Residual Connection)** | `x + Sublayer(x)` 的作用 | 解决深层网络的梯度消失问题 |
| **Layer Normalization** | 与 Batch Normalization 的区别 | LN 对序列长度不敏感，适合 NLP |
| | LN 的公式: `y = γ * (x - μ) / σ + β` | 对每个样本独立做归一化 |
| **Padding Mask** | 为什么需要 | 不同长度的句子 padding 后，不能让模型 attend 到 `<PAD>` |
| | 实现方式: 将 padding 位置的 attention score 设为 -∞ | softmax 后这些位置权重为 0 |
| **Subsequent Mask (Causal Mask)** | 为什么需要 | 训练 decoder 时不能看到未来的 token |
| | 上三角矩阵 mask | 位置 i 只能 attend 到 ≤ i 的位置 |
| **Encoder-Decoder 架构** | Encoder 做什么 | 把输入序列编码为上下文表示 |
| | Decoder 做什么 | 自回归地生成输出序列 |
| | Cross-Attention | Decoder 的每个位置 attend 到 Encoder 的所有输出 |
| **训练 vs 推理** | 训练: teacher forcing（一次性输入整个目标序列） | 利用 ground truth 加速训练 |
| | 推理: autoregressive（逐个生成 token） | 每次只能基于已生成的内容预测下一个 |
| **序列到序列任务** | 输入和输出可以是不同长度 | 如 "123+456=" → "579" |
| | 需要 `<EOS>` token 标记结束 | 推理时知道什么时候停止生成 |

---

## 二、实验的两个子任务

### 2.1 子任务 1: 多位数加法

**任务描述：** 让 Transformer 学习做加法，如 "123+456=" → "579"

**为什么选这个任务？**

1. **确定性** — 加法有唯一正确答案，容易评估
2. **需要推理** — 不是简单的模式匹配，模型需要理解进位逻辑
3. **可以测试泛化性** — 训练 3+3 位数，测试 4+4 位数，看模型能否外推

**数据生成方式：**

```python
def generate_addition_data(num_samples, max_digits=3):
    """
    生成加法题目和答案
    格式: "123+456=" → "579"
    """
    pairs = []
    for _ in range(num_samples):
        a = random.randint(10**(max_digits-1), 10**max_digits - 1)
        b = random.randint(10**(max_digits-1), 10**max_digits - 1)
        src = f"{a}+{b}="
        tgt = f"{a+b}"
        pairs.append((src, tgt))
    return pairs
```

**词表设计：**

```
词表 = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '=', '<PAD>', '<EOS>', '<GO>'}
共 15 个 token
```

- `<GO>`: decoder 的起始 token
- `<EOS>`: 标记答案结束
- `<PAD>`: padding

**关键实验设计：**

| 实验 | 训练集 | 测试集 | 测试什么 |
|------|--------|--------|---------|
| 同分布 | 3+3 位数 | 3+3 位数 | 模型能不能学会 |
| 外推测试 | 3+3 位数 | 4+4 位数 | 能否泛化到更长的序列 |
| 外推测试 | 3+3 位数 | 5+5 位数 | 外推能力有多强 |
| 混合训练 | 2+2 ~ 5+5 混合 | 各种长度 | 混合训练能否提升泛化 |

**预期结果：**
- 同分布：应该能达到 95%+ 准确率
- 外推：会明显下降，因为模型没见过更长的序列（位置编码外推 + 进位逻辑外推都很困难）
- 这是 Transformer 的已知弱点 — 难以外推到训练时未见过的序列长度

### 2.2 子任务 2: 语言模型

**任务描述：** 让 Transformer 学习预测下一个 token（language modeling）

**语料选择建议：**

| 语料 | 优点 | 缺点 |
|------|------|------|
| 莎士比亚文本 | 经典，数据量适中 | 古英语，词表大 |
| 中文古诗 | 有趣，规则性强 | 需要处理中文 |
| 代码片段 | 结构清晰 | 需要特殊 token |
| 新闻标题 | 简单，句子短 | 信息量有限 |

**推荐：用英文新闻标题或莎士比亚文本**，因为：
- 纯英文，tokenizer 简单（空格分词或字符级）
- 数据量可控
- 有现成的数据集

**两种实现方式：**

#### 方式 A: Decoder-only（类似 GPT）

```
输入: "the cat sat on the"
目标: "cat sat on the mat"

训练: 用 subsequent mask，每个位置只能看到前面的 token
推理: 逐个生成下一个 token
```

#### 方式 B: Encoder-Decoder（类似原始 Transformer）

```
输入: "the cat sat on"
目标: "the mat"  # 预测下一个词

这种方式不太适合语言建模，通常用 Decoder-only
```

**评估指标：**

- **Perplexity (困惑度)**: `exp(cross_entropy_loss)`，越低越好
- **生成样本质量**: 人工看生成的文本是否通顺

---

## 三、模型架构设计

### 3.1 模块层次

```
Transformer
├── Embedding + PositionalEncoding
├── TransformerEncoder
│   └── TransformerEncoderLayer × N
│       ├── MultiHeadAttention (self-attention)
│       ├── Add & LayerNorm
│       ├── FeedForward
│       └── Add & LayerNorm
└── TransformerDecoder
    └── TransformerDecoderLayer × N
        ├── MultiHeadAttention (self-attention, with causal mask)
        ├── Add & LayerNorm
        ├── MultiHeadAttention (cross-attention, attend to encoder)
        ├── Add & LayerNorm
        ├── FeedForward
        └── Add & LayerNorm
```

### 3.2 MultiHeadAttention 详解

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead  # 每个 head 的维度
        
        # Q, K, V 的线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # 输出变换
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. 线性变换 + 分头
        Q = self.W_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        # Q, K, V: (batch, nhead, seq_len, d_k)
        
        # 2. Scaled Dot-Product Attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        # scores: (batch, nhead, seq_len_q, seq_len_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 3. 加权求和
        output = attn_weights @ V
        # output: (batch, nhead, seq_len, d_k)
        
        # 4. 合并头 + 输出变换
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output, attn_weights
```

**关键细节：**

1. **为什么除以 √d_k？**
   - 当 d_k 很大时，Q·K 的点积方差会很大
   - 大的点积会让 softmax 进入梯度极小的饱和区
   - 除以 √d_k 把方差归一化到 1

2. **为什么分多头？**
   - 单个 attention head 只能关注一种关系
   - 多个 head 可以并行关注不同类型的关系（语法、语义、位置等）
   - 8 个 head 的 64 维 ≠ 1 个 head 的 512 维，因为每个 head 有独立的 Q/K/V 变换

### 3.3 Positional Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

**为什么用 sin/cos？**
- 可以让模型学到相对位置关系：`PE(pos+k)` 可以表示为 `PE(pos)` 的线性函数
- 可以外推到比训练时更长的序列（虽然效果有限）

### 3.4 训练 vs 推理

**训练（Teacher Forcing）：**

```
Encoder 输入: "123+456="
Decoder 输入: "<GO>579<EOS>"   # 完整的目标序列
Decoder 输出: "579<EOS><PAD>"  # 预测下一个 token

损失: 每个位置的预测 vs 真实下一个 token 的交叉熵
```

**推理（Autoregressive）：**

```
Encoder 输入: "123+456="
Decoder 输入: "<GO>"
→ 预测: "5"

Decoder 输入: "<GO>5"
→ 预测: "7"

Decoder 输入: "<GO>57"
→ 预测: "9"

Decoder 输入: "<GO>579"
→ 预测: "<EOS>"  # 停止
```

---

## 四、实验设计

### 4.1 子任务 1: 加法实验矩阵

| # | 实验名 | 训练数据 | 测试数据 | 目的 |
|---|--------|---------|---------|------|
| 1 | baseline_3plus3 | 3+3 位数 | 3+3 位数 | 基线：能不能学会 |
| 2 | extrapolate_4plus4 | 3+3 位数 | 4+4 位数 | 外推能力 |
| 3 | extrapolate_5plus5 | 3+3 位数 | 5+5 位数 | 外推极限 |
| 4 | mixed_training | 2+2 ~ 5+5 混合 | 各种长度 | 混合训练能否提升泛化 |
| 5 | decoder_only | 3+3 位数 | 3+3 位数 | Decoder-only 变体对比 |

### 4.2 子任务 2: 语言模型实验矩阵

| # | 实验名 | 模型 | 词表 | 目的 |
|---|--------|------|------|------|
| 1 | lm_decoder_only | Decoder-only | 字符级 | 基线 |
| 2 | lm_word_level | Decoder-only | 词级 | 词级 vs 字符级 |
| 3 | lm_deeper | 更多层 | 字符级 | 深度影响 |
| 4 | lm_wider | 更多 head | 字符级 | 宽度影响 |

### 4.3 超参数搜索

| 参数 | 范围 | 说明 |
|------|------|------|
| d_model | 64, 128, 256 | 模型维度 |
| nhead | 2, 4, 8 | 注意力头数 |
| num_layers | 1, 2, 4 | encoder/decoder 层数 |
| dropout | 0.0, 0.1, 0.3 | 正则化强度 |
| learning_rate | 0.0001, 0.0005, 0.001 | 学习率 |
| batch_size | 32, 64, 128 | batch 大小 |

---

## 五、实现难点与注意事项

### 5.1 Mask 的正确实现

**Padding Mask（Encoder 和 Decoder 都用）：**

```python
def make_pad_mask(seq, pad_idx=0):
    # seq: (batch, seq_len)
    # 返回: (batch, 1, 1, seq_len) 的 mask，1 表示有效，0 表示 padding
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask  # (batch, 1, 1, seq_len)
```

**Subsequent Mask（仅 Decoder self-attention 用）：**

```python
def make_subsequent_mask(size):
    # 返回: (1, 1, size, size) 的上三角 mask
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
    return mask  # 下三角为 1，上三角为 0
```

**组合使用：**

```python
# Decoder self-attention 需要同时应用两种 mask
def make_decoder_mask(tgt, pad_idx=0):
    pad_mask = make_pad_mask(tgt, pad_idx)
    sub_mask = make_subsequent_mask(tgt.size(1))
    # 两种 mask 取 AND
    return pad_mask & sub_mask
```

### 5.2 梯度裁剪

Transformer 训练时梯度可能爆炸，建议加 gradient clipping：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 5.3 学习率调度

原始论文用 warmup + 衰减：

```python
class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model ** -0.5 * min(
            self.step_num ** -0.5,
            self.step_num * self.warmup_steps ** -1.5
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

### 5.4 标签平滑

原始论文用了 label smoothing (ε=0.1) 防止过拟合：

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## 六、项目文件结构

```
task3/
├── run_task3.py                  # 主程序
├── task3_函数设计与实现思路.md    # 设计文档
├── outputs_addition/             # 加法实验输出
│   ├── metrics_summary.json
│   ├── accuracy_curves.png
│   └── extrapolation_results.png
├── outputs_lm/                   # 语言模型实验输出
│   ├── metrics_summary.json
│   ├── perplexity_curves.png
│   └── generated_samples.txt
└── models/                       # 保存的模型
    ├── transformer_addition.pt
    └── transformer_lm.pt
```

---

## 七、预期结果

### 7.1 加法任务

| 实验 | 预期准确率 | 说明 |
|------|-----------|------|
| 3+3 → 3+3 | 95%+ | 同分布，应该轻松学会 |
| 3+3 → 4+4 | 30%~60% | 外推困难，但部分能泛化 |
| 3+3 → 5+5 | 10%~30% | 外推更困难 |
| 混合训练 → 各种 | 70%~90% | 混合训练显著提升泛化 |

### 7.2 语言模型

- 字符级语言模型的 perplexity 预计在 2~5 之间（取决于语料复杂度）
- 生成的文本应该能学到一些局部模式（如英文的拼写规则）
- 但很难学到长距离的语义连贯性（数据量和模型规模都不够）

---

## 八、与 Task-1/2 的关系

| 维度 | Task-1 | Task-2 | Task-3 |
|------|--------|--------|--------|
| 模型来源 | 手写线性层 | 调 PyTorch 库 | 手写 Transformer |
| 数据 | 现成 | 现成 | 自己生成 |
| 任务类型 | 分类 | 分类 | 序列生成 |
| 训练方式 | 标准 mini-batch | 标准 mini-batch | Teacher forcing + Autoregressive |
| 核心能力 | 理解梯度下降 | 理解深度学习工具链 | 理解 Transformer 架构 |
| 难度 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

Task-3 是整个系列的"毕业设计"，需要综合运用前两个任务学到的所有知识，同时深入理解 Transformer 的内部机制。
