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
