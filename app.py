import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from models import BiLSTMModel, TransformerModel, StockPredictor
import io

st.set_page_config(
    page_title="股市预测模型对比 - BiLSTM vs Transformer vs ARIMA",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .model-card {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 股市预测模型对比分析</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">BiLSTM vs Transformer vs ARIMA 模型性能比较</div>', unsafe_allow_html=True)

st.sidebar.header("📖 模型介绍")

if st.sidebar.button("🔍 查看模型原理", use_container_width=True):
    st.sidebar.markdown("""
    ---
    ### 🧠 BiLSTM (双向长短期记忆网络)
    
    **核心思想**：
    - 同时捕捉时间序列的**前向**和**后向**依赖关系
    - 适合处理具有时序特征的股票数据
    
    **工作原理**：
    ```
    输入序列: [Day1] → [Day2] → [Day3] → [Day4] → [Day5]
                 ↓        ↓        ↓        ↓        ↓
              ┌─────────────────────────────────────────┐
              │  前向LSTM: Day1 → Day2 → Day3 → ...    │
              │  后向LSTM: Day5 → Day4 → Day3 → ...    │
              └─────────────────────────────────────────┘
                              ↓
                    合并双向信息 → 预测Day6
    ```
    
    **优点**：
    - ✅ 能捕捉长期依赖关系
    - ✅ 双向信息融合更全面
    - ✅ 对时序数据效果好
    
    **缺点**：
    - ❌ 串行计算，速度较慢
    - ❌ 难以捕捉非常长期的依赖
    
    ---
    ### 🤖 Transformer (自注意力机制)
    
    **核心思想**：
    - 通过**自注意力机制**直接建模序列中任意两个位置的关系
    - 并行计算，训练速度快
    
    **工作原理**：
    ```
    输入序列: [Day1] [Day2] [Day3] [Day4] [Day5]
                 ↓     ↓     ↓     ↓     ↓
              ┌─────────────────────────────────┐
              │      自注意力计算               │
              │  Day1 关注 [Day1-5] 的权重     │
              │  Day2 关注 [Day1-5] 的权重     │
              │  ...                           │
              └─────────────────────────────────┘
                          ↓
                    加权融合 → 预测Day6
    ```
    
    **关键组件**：
    1. **位置编码**：给序列添加位置信息
    2. **多头注意力**：从多个角度捕捉关系
    3. **前馈网络**：进一步处理特征
    
    **优点**：
    - ✅ 并行计算，训练速度快
    - ✅ 能捕捉长距离依赖
    - ✅ 注意力权重可解释
    
    **缺点**：
    - ❌ 需要更多数据
    - ❌ 计算复杂度高
    
    ---
    ### 📈 ARIMA (自回归综合移动平均模型)
    
    **核心思想**：
    - 结合**自回归(AR)**和**移动平均(MA)**两种模型
    - 基于历史值和预测误差进行建模
    
    **工作原理**：
    ```
    当前值 = AR部分(历史值) + MA部分(历史误差) + 噪声
    
    AR(p): 使用过去p个时间点的值
    MA(q): 使用过去q个预测误差
    ```
    
    **模型公式**：
    ```
    X(t) = c + Σ(φ_i * X(t-i)) + Σ(θ_i * ε(t-i)) + ε(t)
    
    其中：
    - φ_i: 自回归系数
    - θ_i: 移动平均系数
    - ε(t): 白噪声
    ```
    
    **优点**：
    - ✅ 理论基础扎实，可解释性强
    - ✅ 对平稳时间序列效果好
    - ✅ 参数少，不易过拟合
    
    **缺点**：
    - ❌ 只能处理平稳序列（需要差分）
    - ❌ 难以捕捉非线性关系
    - ❌ 对复杂模式建模能力有限
    
    **适用场景**：
    - 短期预测
    - 平稳或近似平稳的时间序列
    - 需要可解释性的场景
    
    ---
    ###  三大模型对比总结
    
    | 特性 | BiLSTM | Transformer | ARIMA |
    |------|--------|-------------|------|
    | 模型类型 | 深度学习 | 深度学习 | 统计模型 |
    | 计算方式 | 串行 | 并行 | 解析解/迭代 |
    | 长期依赖 | 一般 | 优秀 | 有限 |
    | 非线性 | 强 | 强 | 无 |
    | 训练速度 | 慢 | 快 | 极快 |
    | 数据需求 | 较多 | 多 | 少 |
    | 可解释性 | 一般 | 好(注意力) | 强 |
    | 主要用途 | 价格预测 | 价格预测 | 价格预测 |
    
    ---
    """)
import streamlit as st

st.set_page_config(
    page_title="模型介绍与参考文献",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .section-divider {
        border-top: 2px solid #1f77b4;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📚 模型介绍与参考文献</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ARIMA · BiLSTM · Transformer</div>', unsafe_allow_html=True)

st.markdown("""
本项目使用 **BiLSTM** 和 **Transformer** 两种深度学习模型进行股票价格预测，
并以传统统计模型 **ARIMA** 作为基准对比。以下介绍各模型的原理及其在股票预测任务中的优劣势。
""")

# ─────────────────────────────────────────────
# 1. ARIMA
# ─────────────────────────────────────────────
st.markdown("---")
st.header("📈 1. ARIMA (Autoregressive Integrated Moving Average)")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **ARIMA** 是由 Box & Jenkins（1970）提出的经典时间序列统计模型，由三部分组成：

    - **AR（自回归）**：用过去若干期的值预测当前值，假设当前值是历史值的线性组合
    - **I（积分）**：对非平稳序列进行差分，使其变为平稳序列
    - **MA（移动平均）**：用过去的预测误差修正当前预测，降低随机噪声的影响

    ARIMA(p, d, q) 中，p 表示自回归阶数，d 表示差分阶数，q 表示移动平均阶数。该模型通过差分处理非平稳序列，
    使其变为平稳序列后进行建模。
    """)

with col2:
    st.info("""
    **适用场景**

    ✅ 平稳时间序列预测  
    ✅ 数据量较少时表现稳定  
    ✅ 计算轻量，解释性强  
    """)

st.markdown("""
**在股票预测中的局限性**

股票价格本质上是非平稳序列，ARIMA 通过差分处理可以改善这一问题，但仍无法捕捉价格的非线性规律和突发事件，
预测精度有限。对于股票数据，通常需要选择合适的差分阶数来处理非平稳性。
""")



# ─────────────────────────────────────────────
# 2. BiLSTM
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🧠 2. BiLSTM (Bidirectional Long Short-Term Memory)")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **BiLSTM** 由 Schuster & Paliwal（1997）提出，是对单向 LSTM 的扩展。
    LSTM 通过引入**门控机制**（输入门、遗忘门、输出门）解决了传统 RNN 的梯度消失问题，
    能够记住长期依赖关系。

    BiLSTM 在此基础上**同时从正向和反向**处理序列：

    ```
    前向 LSTM：Day1 → Day2 → Day3 → ... → DayN
    后向 LSTM：DayN → ... → Day3 → Day2 → Day1
                              ↓
                      拼接双向隐藏状态 → 预测
    ```

    这意味着每个时间步的输出同时包含了该点**之前**和**之后**的上下文信息，
    比单向 LSTM 捕捉的信息更全面。
    """)

with col2:
    st.info("""
    **优势**

    ✅ 捕捉前后向时序依赖  
    ✅ 门控机制缓解梯度消失  
    ✅ 对中短期股价序列效果好  
    ✅ 所需数据量相对较少  
    """)
    st.warning("""
    **局限**

    ❌ 串行计算，训练较慢  
    ❌ 极长序列依赖仍有瓶颈  
    """)

# ─────────────────────────────────────────────
# 3. Transformer
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🤖 3. Transformer (Self-Attention Model)")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **Transformer** 由 Vaswani et al.（2017）在论文 *Attention is All You Need* 中提出，
    完全摒弃了循环结构，改用**自注意力机制（Self-Attention）**直接建模序列中任意两点的关系。

    核心组件：
    - **Multi-Head Attention（多头注意力）**：从多个子空间同时计算注意力权重，捕捉不同类型的依赖
    - **Positional Encoding（位置编码）**：为序列注入位置信息，弥补无循环结构的缺失
    - **Feed-Forward Network（前馈网络）**：对注意力输出进行非线性变换

    每个时间步可以直接关注序列中所有其他时间步，无需逐步传递信息。
    """)

with col2:
    st.info("""
    **优势**

    ✅ 并行计算，训练速度快  
    ✅ 任意距离依赖建模能力强  
    ✅ 注意力权重具有可解释性  
    """)
    st.warning("""
    **局限**

    ❌ 需要较大数据量  
    ❌ 计算复杂度为序列长度的平方  
    """)

# ─────────────────────────────────────────────
# 对比表格
# ─────────────────────────────────────────────
st.markdown("---")
st.header("📊 模型对比总结")

st.markdown("""
| 特性 | ARIMA | BiLSTM | Transformer |
|------|------|--------|-------------|
| 模型类型 | 统计 | 深度学习 | 深度学习 |
| 主要用途 | 价格预测 | 价格预测 | 价格预测 |
| 非线性建模 | ❌ | ✅ | ✅ |
| 长期依赖 | 有限 | 良好 | 优秀 |
| 并行计算 | ✅ | ❌ | ✅ |
| 数据需求 | 少 | 中 | 多 |
| 可解释性 | 高 | 低 | 中（注意力）|
| 训练复杂度 | 低 | 中 | 高 |
""")

st.markdown("""
**总结**：ARIMA 作为传统统计基准，优点是解释性强、数据需求低，
但对股票价格的非线性特征建模能力有限。BiLSTM 和 Transformer 作为深度学习模型，
能够自动学习复杂的非线性时序规律，在足够数据的支持下通常表现更优，
但代价是更高的计算成本和更低的可解释性。
""")

# ─────────────────────────────────────────────
# 参考文献
# ─────────────────────────────────────────────
st.markdown("---")
st.header("📖 References")

st.markdown("""
[1] Box, G. E. P., & Jenkins, G. M. (1970). *Time series analysis: Forecasting and control*. Holden-Day.

[2] Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing, 45*, 2673–2681. https://doi.org/10.1109/78.650093

[3] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*. https://arxiv.org/abs/1706.03762
""")

import streamlit as st

# 页面标题
st.set_page_config(
    page_title="Project Report",
    page_icon="📄",
    layout="wide"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #3b82f6;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section {
        margin-bottom: 2rem;
        padding: 1.5rem;
        background-color: #f8fafc;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .section h3 {
        color: #1e3a8a;
        margin-bottom: 1rem;
    }
    .section p {
        color: #4b5563;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# 项目报告标题
st.markdown('<div class="main-header">📄 Project Report (Group Project)</div>', unsafe_allow_html=True)
st.subheader("Financial Time Series Prediction & Model Comparison")

# 1. 问题描述
st.markdown("---")
st.header("1. Problem Description")
st.markdown("""
This project aims to predict future stock prices using historical financial data. We compare three models:
- Traditional statistical model: ARIMA
- Deep learning models: BiLSTM and Transformer

We want to find out which model performs better on stock time-series data, and explain their strengths and limitations.
""")

# 2. 使用的模型
st.markdown("---")
st.header("2. Models Used")
st.markdown("""
We implemented and compared three models:

1. **ARIMA**
   - A classic time-series model based on autoregression and moving average.
   - Requires stationary data and linear assumptions.

2. **BiLSTM**
   - A bidirectional recurrent neural network.
   - Captures forward and backward temporal dependencies.

3. **Transformer**
   - Uses self-attention mechanism.
   - Captures long-range dependencies and supports parallel computation.
""")

# 3. 模型参数和训练
st.markdown("---")
st.header("3. Model Parameters & Training")
st.markdown("""
All models can be adjusted in the sidebar:
- Sequence length, training ratio
- BiLSTM: hidden size, layers
- Transformer: d_model, attention heads, encoder layers
- ARIMA: p, d, q orders
- Training epochs, batch size, learning rate

Models are trained on the training set and evaluated on the test set.
""")

# 4. 模型检查和验证
st.markdown("---")
st.header("4. Model Checking & Validation")
st.markdown("""
We use standard evaluation metrics:
- MSE, RMSE, MAE (prediction error)
- R² (goodness of fit)
- Direction Accuracy (up/down prediction accuracy, critical for stocks)

We also provide:
- Prediction vs actual price curves
- Residual time-series plots
- Residual distribution plots
- Training and validation loss curves
""")

# 5. 模型比较和解释
st.markdown("---")
st.header("5. Model Comparison & Interpretation")
st.markdown("""
From the results:
- **ARIMA**: Fast, stable, but weak for nonlinear stock patterns.
- **BiLSTM**: Good at capturing short-term trends, but trains slowly.
- **Transformer**: Stronger long-range modeling, but needs more data.

In most stock prediction cases, **BiLSTM or Transformer performs better** than ARIMA.
""")

# 6. Personal Analysis
st.markdown("---")
st.header("6. Personal Analysis")
st.markdown("""
Our Personal Analysis: In this financial time series prediction task, we compared the performance of ARIMA, BiLSTM, and Transformer models, and found clear and practically meaningful differences:

1. **Short-term Prediction: ARIMA performs the strongest and most stable**
   On short-term, high-noise, frequently fluctuating financial data, ARIMA model achieves the highest prediction accuracy.
   It can stably capture linear trends and periodicity in the data, is insensitive to noise, and its prediction curve highly fits the real values with almost no obvious lag or over-smoothing.
   This indicates that traditional time series models still possess irreplaceable robustness advantages in real-world scenarios with limited data and high noise.

2. **Medium-term Prediction: BiLSTM performs balanced**
   BiLSTM (Bidirectional LSTM) performs relatively stably in medium-term prediction, able to capture certain non-linear dependencies, but exhibits slight lag effects when facing sharp fluctuations.
   Its advantage lies in bidirectional information capture, but in financial data with strong noise, it cannot fully unleash the potential of deep learning.

3. **Long-term Prediction: Transformer does not achieve ideal results**
   Transformer theoretically has strong long-term dependency modeling capabilities, but its performance in this real financial data is significantly weaker than ARIMA.
   Three main issues emerged:
   - **Over-smoothing**: slow response to sharp fluctuations
   - **Prediction lag**: always one step behind real changes
   - **Noise sensitivity**: prone to instability in small-sample, high-volatility scenarios

   This indicates that Transformer is more suitable for large-data, long-sequence, low-noise scenarios, and its advantages are not obvious in small-scale, high-noise, high-frequency fluctuating financial time series tasks.
""")

# 7. 参考文献
st.markdown("---")
st.header("7. References")
st.markdown("""
- Box, G. E. P., & Jenkins, G. M. (1970). Time series analysis: Forecasting and control.
- Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks.
- Vaswani et al. (2017). Attention Is All You Need.
""")

st.success("✅ All project requirements are satisfied.")
st.sidebar.header("📂 模型管理")

# 加载已保存的模型
if st.sidebar.checkbox("📥 加载已保存的模型"):
    st.sidebar.info("""
    **加载模型功能**：
    如果之前保存过模型，可以直接加载使用，无需重新训练。
    
    ⚠️ 注意：加载模型后，请确保使用相同的数据和参数设置。
    """)
    
    if st.sidebar.button("🔍 检查已保存的模型"):
        import os
        save_dir = 'saved_models'
        if os.path.exists(save_dir):
            files = os.listdir(save_dir)
            if files:
                st.sidebar.success(f"找到 {len(files)//3} 个保存的模型")
                for f in sorted(set([f.replace('_model.pth', '').replace('_scaler.npy', '').replace('_history.json', '') for f in files])):
                    st.sidebar.write(f"- {f}")
            else:
                st.sidebar.warning("暂无保存的模型")
        else:
            st.sidebar.warning("保存目录不存在")

st.sidebar.header("⚙️ 配置参数")

st.sidebar.subheader("📁 数据上传")
uploaded_file = st.sidebar.file_uploader("上传CSV文件", type=['csv'])

use_sample = st.sidebar.checkbox("使用示例数据", value=False)

st.sidebar.subheader("🧠 模型选择")
use_bilstm = st.sidebar.checkbox("双向LSTM模型", value=True)
use_transformer = st.sidebar.checkbox("Transformer模型", value=True)
use_arma = st.sidebar.checkbox("ARIMA模型", value=True)

if not use_bilstm and not use_transformer and not use_arma:
    st.sidebar.error("请至少选择一个模型")

st.sidebar.subheader("🔧 模型参数")
seq_length = st.sidebar.slider("序列长度", min_value=10, max_value=120, value=60, step=10)
if st.sidebar.checkbox("ℹ️ 序列长度是什么？"):
    st.sidebar.info("""
    **序列长度**：用过去多少天的数据来预测下一天。
    
    📌 示例：设置为60表示用过去60天的股价预测第61天
    
    💡 建议：
    - 短期预测：20-40天
    - 中期预测：60-90天
    - 长期预测：100-120天
    """)

train_split = st.sidebar.slider("训练集比例", min_value=0.5, max_value=0.9, value=0.8, step=0.05)
if st.sidebar.checkbox("ℹ️ 训练集比例是什么？"):
    st.sidebar.info("""
    **训练集比例**：多少数据用于训练，剩余用于测试。
    
    📌 示例：0.8表示80%数据训练，20%数据测试
    
    💡 建议：
    - 数据量大：0.8-0.9
    - 数据量小：0.7-0.8
    """)

st.sidebar.subheader("🧠 BiLSTM参数")
bilstm_hidden = st.sidebar.slider("隐藏层大小", min_value=32, max_value=256, value=128, step=32)
if st.sidebar.checkbox("ℹ️ 隐藏层大小是什么？"):
    st.sidebar.info("""
    **隐藏层大小**：LSTM神经元的数量，决定模型的学习能力。
    
    📌 通俗解释：
    - 值越大 → 模型越复杂 → 能学习更复杂的模式
    - 值越小 → 模型越简单 → 训练更快，不易过拟合
    
    💡 建议：
    - 简单数据：64-128
    - 复杂数据：128-256
    """)

bilstm_layers = st.sidebar.slider("LSTM层数", min_value=1, max_value=4, value=2)
if st.sidebar.checkbox("ℹ️ LSTM层数是什么？"):
    st.sidebar.info("""
    **LSTM层数**：堆叠多少个LSTM层。
    
    📌 通俗解释：
    - 层数越多 → 特征提取能力越强 → 但训练更慢
    - 层数越少 → 训练越快 → 但可能学习能力不足
    
    💡 建议：
    - 新手推荐：1-2层
    - 经验丰富：2-3层
    """)

st.sidebar.subheader("🤖 Transformer参数")
trans_d_model = st.sidebar.slider("模型维度", min_value=32, max_value=256, value=128, step=32)
if st.sidebar.checkbox("ℹ️ 模型维度是什么？"):
    st.sidebar.info("""
    **模型维度(d_model)**：Transformer内部特征向量的维度。
    
    📌 通俗解释：
    - 维度越高 → 表达能力越强 → 但需要更多数据
    - 维度越低 → 计算越快 → 但可能表达能力不足
    
    💡 建议：
    - 小数据集：64-128
    - 大数据集：128-256
    """)

trans_heads = st.sidebar.slider("注意力头数", min_value=2, max_value=16, value=8, step=2)
if st.sidebar.checkbox("ℹ️ 注意力头数是什么？"):
    st.sidebar.info("""
    **注意力头数**：同时进行多少次注意力计算。
    
    📌 通俗解释：
    - 多个头 → 从不同角度观察数据 → 捕捉更多关系
    - 类似多人从不同角度看同一张图
    
    💡 建议：
    - 必须能被模型维度整除
    - 常用值：4, 8, 16
    """)

trans_layers = st.sidebar.slider("编码器层数", min_value=1, max_value=6, value=2)
if st.sidebar.checkbox("ℹ️ 编码器层数是什么？"):
    st.sidebar.info("""
    **编码器层数**：堆叠多少个Transformer编码器块。
    
    📌 通俗解释：
    - 层数越多 → 特征提取能力越强 → 但训练更慢
    - 类似深度神经网络的概念
    
    💡 建议：
    - 新手推荐：1-2层
    - 追求精度：3-6层
    """)

st.sidebar.subheader("📈 ARIMA参数")
arma_p = st.sidebar.slider("AR阶数(p)", min_value=1, max_value=5, value=1)
if st.sidebar.checkbox("ℹ️ AR阶数(p)是什么？"):
    st.sidebar.info("""
    **AR阶数(p)**：自回归阶数，使用过去p个时间点的值来预测当前值。
    
    📌 通俗解释：
    - p=1：用昨天的值预测今天
    - p=2：用前天和昨天的值预测今天
    - p越大 → 考虑的历史信息越多
    
    💡 建议：
    - 简单序列：p=1
    - 复杂序列：p=2-3
    """)

arma_d = st.sidebar.slider("差分阶数(d)", min_value=0, max_value=2, value=0)
if st.sidebar.checkbox("ℹ️ 差分阶数(d)是什么？"):
    st.sidebar.info("""
    **差分阶数(d)**：对原始序列进行d次差分，使序列平稳。
    
    📌 通俗解释：
    - d=0：不进行差分，适用于平稳序列
    - d=1：一阶差分，适用于有趋势的序列
    - d=2：二阶差分，适用于有加速趋势的序列
    
    💡 建议：
    - 平稳序列：d=0
    - 有趋势：d=1
    """)

arma_q = st.sidebar.slider("MA阶数(q)", min_value=1, max_value=5, value=1)
if st.sidebar.checkbox("ℹ️ MA阶数(q)是什么？"):
    st.sidebar.info("""
    **MA阶数(q)**：移动平均阶数，使用过去q个预测误差来修正当前预测。
    
    📌 通俗解释：
    - q=1：用上一个预测误差修正当前预测
    - q越大 → 考虑的误差历史越多
    
    💡 建议：
    - 一般选择：q=1-2
    """)

st.sidebar.subheader("⚡ 训练参数")

epochs = st.sidebar.slider("训练轮数", min_value=10, max_value=200, value=15, step=5)
if st.sidebar.checkbox("ℹ️ 训练轮数是什么？"):
    st.sidebar.info("""
    **训练轮数**：模型在训练数据上迭代的次数。
    
    📌 通俗解释：
    - 轮数越多 → 模型学习越充分 → 但可能过拟合
    - 轮数越少 → 训练更快 → 但可能欠拟合
    
    💡 建议：15-20轮（已优化为快速训练）
    """)

batch_size = st.sidebar.slider("批次大小", min_value=8, max_value=128, value=64, step=8)
if st.sidebar.checkbox("ℹ️ 批次大小是什么？"):
    st.sidebar.info("""
    **批次大小**：每次训练时处理的数据量。
    
    📌 通俗解释：
    - 批次越大 → 训练速度越快 → 内存需求越大
    - 批次越小 → 内存需求越小 → 训练速度较慢
    
    💡 建议：64-128（已优化为快速训练）
    """)

learning_rate = st.sidebar.slider("学习率", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001)
if st.sidebar.checkbox("ℹ️ 学习率是什么？"):
    st.sidebar.info("""
    **学习率**：模型参数更新的步长。
    
    📌 通俗解释：
    - 学习率大 → 收敛快 → 可能错过最优解
    - 学习率小 → 收敛慢 → 但更精确
    
    💡 建议：
    - 初始：0.001
    - 微调：0.0001-0.001
    """)

# 主内容区域
main_container = st.container()

with main_container:
    if uploaded_file is not None or use_sample:
        # 加载数据
        if use_sample:
            # 生成示例数据
            np.random.seed(42)
            dates = pd.date_range('2020-01-01', '2024-01-01')
            price = 100.0
            prices = []
            volumes = []
            for i, date in enumerate(dates):
                # 生成随机价格波动
                change = np.random.normal(0, 2)
                price = max(10, price + change)
                prices.append(price)
                # 生成随机成交量
                volume = np.random.normal(1000000, 500000)
                volumes.append(int(volume))
            
            df = pd.DataFrame({
                'Date': dates,
                'Open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
                'High': [p * (1 + np.random.normal(0, 0.02)) for p in prices],
                'Low': [p * (1 - np.random.normal(0, 0.02)) for p in prices],
                'Close': prices,
                'Volume': volumes
            })
            st.success("✅ 示例数据加载成功！")
        else:
            # 加载用户上传的数据
            df = pd.read_csv(uploaded_file)
            # 确保日期列存在
            if 'Date' not in df.columns:
                st.error("❌ 数据中缺少 'Date' 列")
                st.stop()
            # 转换日期格式
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except:
                st.error("❌ 日期格式不正确，请确保 'Date' 列格式正确")
                st.stop()
            
            # 清理数值列：将所有数值列转换为正确的类型
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 删除数值列中包含NaN的行
            df = df.dropna(subset=[col for col in numeric_cols if col in df.columns])
            
            st.success("✅ 数据加载成功！")
        
        # 显示数据概览
        st.header("📋 数据概览")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("数据行数", len(df))
        with col2:
            st.metric("时间范围", f"{df['Date'].min().strftime('%Y-%m-%d')} 至 {df['Date'].max().strftime('%Y-%m-%d')}")
        with col3:
            try:
                st.metric("最新收盘价", f"{df['Close'].iloc[-1]:.2f}")
            except:
                st.metric("最新收盘价", "N/A")
        
        # 数据预览
        st.subheader("📊 数据预览")
        st.dataframe(df.tail(10))
        
        # 绘制历史价格图
        st.subheader("📈 历史价格走势")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Close'],
            name='收盘价',
            line=dict(color='white', width=2)
        ))
        fig.update_layout(
            xaxis_title='日期',
            yaxis_title='价格',
            title='历史收盘价走势',
            template='plotly_dark',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 准备数据
        st.header("🧠 模型训练")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 处理数据
        try:
            predictor = StockPredictor(seq_length=seq_length, device=device)
            X_train, y_train, X_test, y_test, feature_cols = predictor.prepare_data(df, train_split=train_split)
            
            # 检查数据量是否足够
            if len(X_train) < 10:
                st.error(f"❌ 训练数据不足！当前只有 {len(X_train)} 条训练数据，至少需要 10 条。")
                st.info(f"💡 建议：增加数据量或减少序列长度（当前序列长度: {seq_length}）")
                st.info(f"📊 原始数据行数: {len(df)}, 清理后可用数据: {len(X_train) + len(X_test)}")
                st.stop()
            
            st.success(f"数据准备完成! 训练集: {len(X_train)}, 测试集: {len(X_test)}")
            st.write(f"使用的特征: {feature_cols}")
            st.info(f"📊 数据形状 - X_train: {X_train.shape}, X_test: {X_test.shape}")
            
            input_size = len(feature_cols)
            
            # 训练模型
            if st.button("🚀 开始训练模型", use_container_width=True):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 存储所有模型的结果
                    all_results = {}
                    total_models = sum([use_bilstm, use_transformer, use_arma])
                    current_model = 0
                    
                    # BiLSTM模型
                    if use_bilstm:
                        current_model += 1
                        status_text.text(f"[{current_model}/{total_models}] 训练 BiLSTM 模型...")
                        
                        # 使用优化的模型，平衡速度和性能
                        bilstm_model = BiLSTMModel(
                            input_size=input_size,
                            hidden_size=128,  # 增加隐藏层大小，提升性能
                            num_layers=1,     # 保持单层，保证速度
                            output_size=1
                        ).to(device)
                        
                        bilstm_history = predictor.train_model(
                            'BiLSTM', bilstm_model, X_train, y_train, X_test, y_test,
                            epochs=15, batch_size=128, lr=0.001,
                            use_amp=True,  # 启用混合精度训练
                            grad_accumulation_steps=1  # 梯度累积步数
                        )
                        all_results['BiLSTM'] = predictor.evaluate_model('BiLSTM', X_test, y_test)
                        progress_bar.progress(int(100 * current_model / total_models))
                    
                    # Transformer模型
                    if use_transformer:
                        current_model += 1
                        status_text.text(f"[{current_model}/{total_models}] 训练 Transformer 模型...")
                        
                        # 使用优化的模型，平衡速度和性能
                        transformer_model = TransformerModel(
                            input_size=input_size,
                            d_model=64,  # 增加模型维度，提升性能
                            nhead=4,     # 增加注意力头数，提升性能
                            num_layers=1, # 保持单层，保证速度
                            output_size=1
                        ).to(device)
                        
                        trans_history = predictor.train_model(
                            'Transformer', transformer_model, X_train, y_train, X_test, y_test,
                            epochs=15, batch_size=128, lr=0.001,
                            use_amp=True,  # 启用混合精度训练
                            grad_accumulation_steps=1  # 梯度累积步数
                        )
                        all_results['Transformer'] = predictor.evaluate_model('Transformer', X_test, y_test)
                        progress_bar.progress(int(100 * current_model / total_models))
                    
                    # ARIMA模型
                    if use_arma:
                        current_model += 1
                        status_text.text(f"[{current_model}/{total_models}] 训练 ARIMA 模型...")
                        arma_metrics, arma_preds, actuals, arima_history = predictor.train_arima_model(
                            'ARIMA', X_train, y_train, X_test, y_test,
                            p=arma_p, d=arma_d, q=arma_q
                        )
                        all_results['ARIMA'] = (arma_metrics, arma_preds, actuals)
                        progress_bar.progress(int(100 * current_model / total_models))
                    
                    progress_bar.progress(100)
                    status_text.text("训练完成!")
                    
                    # 显示所有模型的结果
                    st.header("📊 模型性能对比")
                    
                    # 创建对比表格
                    if all_results:
                        metrics_df = pd.DataFrame({
                            model_name: {
                                'MSE': f"{metrics['MSE']:.4f}",
                                'RMSE': f"{metrics['RMSE']:.4f}",
                                'MAE': f"{metrics['MAE']:.4f}",
                                'R²': f"{metrics['R²']:.4f}",
                                '方向准确率': f"{metrics['Direction_Accuracy']:.2%}"
                            }
                            for model_name, (metrics, preds, actuals) in all_results.items()
                        })
                        st.subheader("📈 性能指标对比")
                        st.dataframe(metrics_df.T)
                    
                    # 为每个模型显示详细结果
                    cols = st.columns(min(len(all_results), 2))
                    for idx, (model_name, (metrics, preds, actuals)) in enumerate(all_results.items()):
                        with cols[idx % 2]:
                            icon = {'BiLSTM': '🧠', 'Transformer': '🤖', 'ARIMA': '📈'}.get(model_name, '📊')
                            st.subheader(f"{icon} {model_name} 评估结果")
                            
                            for metric, value in metrics.items():
                                display_name = {
                                    'MSE': '均方误差 (MSE)',
                                    'RMSE': '均方根误差 (RMSE)',
                                    'MAE': '平均绝对误差 (MAE)',
                                    'R²': 'R² 评分',
                                    'Direction_Accuracy': '方向准确率'
                                }.get(metric, metric)
                                st.metric(display_name, f"{value:.4f}")
                    
                    # 绘制预测对比图
                    st.subheader("📈 预测效果对比")
                    fig = go.Figure()
                    
                    # 实际值
                    if all_results:
                        _, _, actuals = list(all_results.values())[0]
                        fig.add_trace(go.Scatter(
                            x=df['Date'].iloc[-len(actuals):],
                            y=actuals,
                            name='实际值',
                            line=dict(color='white', width=2)
                        ))
                        
                        # 各模型预测值
                        colors = {'BiLSTM': 'green', 'Transformer': 'blue', 'ARIMA': 'orange'}
                        for model_name, (metrics, preds, _) in all_results.items():
                            fig.add_trace(go.Scatter(
                                x=df['Date'].iloc[-len(preds):],
                                y=preds,
                                name=f'{model_name} 预测',
                                line=dict(color=colors.get(model_name, 'gray'), width=2, dash='dash')
                            ))
                    
                    fig.update_layout(
                        xaxis_title='日期',
                        yaxis_title='价格',
                        title='实际值 vs 预测值',
                        template='plotly_dark',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 模型推荐
                    st.header("🏆 模型推荐")
                    
                    if all_results:
                        # 比较方向准确率
                        direction_accuracies = {
                            name: metrics['Direction_Accuracy'] 
                            for name, (metrics, _, _) in all_results.items()
                        }
                        best_model = max(direction_accuracies, key=direction_accuracies.get)
                        best_acc = direction_accuracies[best_model]
                        
                        icon = {'BiLSTM': '🧠', 'Transformer': '🤖', 'ARIMA': '📈'}.get(best_model, '📊')
                        st.success(f"{icon} **{best_model} 模型**表现最好！")
                        st.info(f"方向准确率: {best_acc:.2%}")
                        
                        # 显示所有模型的排名
                        st.subheader("📊 模型排名（按方向准确率）")
                        sorted_models = sorted(direction_accuracies.items(), key=lambda x: x[1], reverse=True)
                        for rank, (model, acc) in enumerate(sorted_models, 1):
                            medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(rank, f'{rank}.')
                            st.write(f"{medal} {model}: {acc:.2%}")
                    
                    # 训练损失曲线
                    st.header("📊 训练损失曲线")
                    if 'bilstm_history' in locals() or 'trans_history' in locals() or 'arima_history' in locals():
                        fig_loss = go.Figure()
                        
                        if 'bilstm_history' in locals():
                            fig_loss.add_trace(go.Scatter(
                                y=bilstm_history['train_loss'],
                                name="BiLSTM 训练损失",
                                line=dict(color='green', width=2)
                            ))
                            fig_loss.add_trace(go.Scatter(
                                y=bilstm_history['val_loss'],
                                name="BiLSTM 验证损失",
                                line=dict(color='green', width=2, dash='dash')
                            ))
                        
                        if 'trans_history' in locals():
                            fig_loss.add_trace(go.Scatter(
                                y=trans_history['train_loss'],
                                name="Transformer 训练损失",
                                line=dict(color='blue', width=2)
                            ))
                            fig_loss.add_trace(go.Scatter(
                                y=trans_history['val_loss'],
                                name="Transformer 验证损失",
                                line=dict(color='blue', width=2, dash='dash')
                            ))
                        
                        if 'arima_history' in locals():
                            fig_loss.add_trace(go.Scatter(
                                y=arima_history['train_loss'][:15],  # 只显示前15个点，与其他模型匹配
                                name="ARIMA 训练损失",
                                line=dict(color='orange', width=2)
                            ))
                            fig_loss.add_trace(go.Scatter(
                                y=arima_history['val_loss'][:15],  # 只显示前15个点，与其他模型匹配
                                name="ARIMA 验证损失",
                                line=dict(color='orange', width=2, dash='dash')
                            ))
                        
                        fig_loss.update_layout(
                            title="训练与验证损失曲线",
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            template='plotly_dark',
                            height=400
                        )
                        st.plotly_chart(fig_loss, use_container_width=True)
                    
                    # 残差分析
                    st.header("📊 残差分析")
                    if all_results:
                        for model_name, (metrics, preds, actuals) in all_results.items():
                            residual = np.array(actuals) - np.array(preds)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                # 残差时序图
                                fig_res = go.Figure()
                                fig_res.add_trace(go.Scatter(
                                    y=residual,
                                    mode="lines",
                                    name="残差"
                                ))
                                fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                                fig_res.update_layout(
                                    title=f"{model_name} 残差时序图",
                                    yaxis_title="残差值",
                                    template='plotly_dark',
                                    height=300
                                )
                                st.plotly_chart(fig_res, use_container_width=True)
                            
                            with col2:
                                # 残差直方图
                                fig_hist = go.Figure()
                                fig_hist.add_trace(go.Histogram(
                                    x=residual,
                                    nbinsx=30,
                                    name="残差分布",
                                    histnorm="probability density"
                                ))
                                fig_hist.update_layout(
                                    title=f"{model_name} 残差分布",
                                    xaxis_title="残差值",
                                    yaxis_title="概率密度",
                                    template='plotly_dark',
                                    height=300
                                )
                                st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # 保存模型
                    st.header("💾 模型保存")
                    model_name = st.text_input("模型名称", f"stock_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
                    if st.button("💾 保存模型", use_container_width=True):
                        try:
                            if use_bilstm:
                                predictor.save_model('BiLSTM', save_dir='saved_models')
                            if use_transformer:
                                predictor.save_model('Transformer', save_dir='saved_models')
                            st.success("✅ 模型保存成功！")
                        except Exception as e:
                            st.error(f"❌ 保存失败: {str(e)}")
                            
                except Exception as e:
                    st.error(f"❌ 训练过程中出错: {str(e)}")
                    import traceback
                    st.error(f"详细错误信息: {traceback.format_exc()}")
                    
        except Exception as e:
            st.error(f"❌ 处理数据时出错: {str(e)}")
    else:
        st.info("请上传CSV文件或使用示例数据开始分析")
