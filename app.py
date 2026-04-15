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

# ====================== 模型介绍按钮（左侧最前面） ======================
st.sidebar.header("📖 模型介绍")
if st.sidebar.button("📘 模型介绍 Model Intro", use_container_width=True, key="btn_intro"):
    with st.expander("模型介绍", expanded=True):
        st.markdown("""
# 📚 模型介绍与参考文献
### ARIMA · BiLSTM · Transformer
本项目使用 **BiLSTM** 和 **Transformer** 两种深度学习模型进行股票价格预测,
并以传统统计模型 **ARIMA** 作为基准对比。以下介绍各模型的原理及其在股票预测任务中的优劣势。

---
## 📈 1. ARIMA (Autoregressive Integrated Moving Average)
**ARIMA** 是由 Box & Jenkins (1970) 提出的经典时间序列统计模型,由三部分组成：
- **AR(自回归)**：用过去若干期的值预测当前值,假设当前值是历史值的线性组合
- **I(积分)**：对非平稳序列进行差分,使其变为平稳序列
- **MA(移动平均)**：用过去的预测误差修正当前预测,降低随机噪声的影响
ARIMA(p, d, q) 中,p 表示自回归阶数,d 表示差分阶数,q 表示移动平均阶数。该模型通过差分处理非平稳序列,
使其变为平稳序列后进行建模。

**适用场景**
✅ 平稳时间序列预测
✅ 数据量较少时表现稳定
✅ 计算轻量,解释性强

**在股票预测中的局限性**
股票价格本质上是非平稳序列,ARIMA 通过差分处理可以改善这一问题,但仍无法捕捉价格的非线性规律和突发事件,
预测精度有限。对于股票数据,通常需要选择合适的差分阶数来处理非平稳性。

---
## 🧠 2. BiLSTM (Bidirectional Long Short-Term Memory)
**BiLSTM** 是由 Schuster & Paliwal (1997) 提出,是对单向 LSTM 的扩展。
LSTM 通过引入**门控机制**（输入门、遗忘门、输出门）解决了传统 RNN 的梯度消失问题,
能够记住长期依赖关系。
BiLSTM 在此基础上**同时从正向和反向**处理序列：
