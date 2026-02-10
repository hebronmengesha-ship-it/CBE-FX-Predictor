import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import date, timedelta
import warnings

# Force web-safe chart rendering
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore")

# --- 1. CORE INTERFACE SETTINGS ---
st.set_page_config(
    page_title="ML-1 PRICE PREDICTOR | RIDGE AR",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. HIGH-CONTRAST MONOCHROME CSS (VISIBILITY FIX) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    h1, h2, h3, h4, p, span, label, div {
        color: #000000 !important;
        background-color: transparent !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 2px solid #000000 !important;
    }
    
    /* CRITICAL FIX: Number Input Visibility */
    div[data-testid="stNumberInput"] input {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        border: 2px solid #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }

    div[data-testid="stSlider"] *, div[data-testid="stSelectSlider"] * {
        background-color: transparent !important;
        color: #000000 !important;
    }
    div[data-baseweb="slider"] > div:first-child > div { background-color: #000000 !important; }
    div[role="slider"] { background-color: #000000 !important; border: 2px solid #000000 !important; }

    div[data-testid="stMetric"] {
        border: 2px solid #000000 !important;
        padding: 15px;
        background-color: #FFFFFF !important;
    }
    label[data-testid="stMetricLabel"] {
        text-transform: uppercase;
        font-size: 0.85rem !important;
        font-weight: 900 !important;
    }
    .analysis-box {
        border: 2px solid #000000;
        padding: 15px;
        font-family: 'Inter', sans-serif;
        color: #000000;
        background-color: #F8F9FA;
        margin-bottom: 20px;
    }
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. DASHBOARD HEADER ---
st.markdown("### **ML-1 PRICE PREDICTOR** / RIDGE AR")
st.markdown(f"<p style='font-size: 13px; font-weight: 700; margin-top:-15px;'>SYSTEM CLOCK: {date.today().strftime('%Y-%m-%d')} | ARCHITECTURE: HYPER-HYBRID ENSEMBLE</p>", unsafe_allow_html=True)

st.markdown(f"""
<div class="analysis-box">
    <strong>Market Dynamics:</strong> This model integrates real-time market noise with a <strong>10% semi-annual devaluation constant</strong>. 
    By hard-coding this structural pressure into the ML recursive loop, the projection accounts for the ETB's historical downward trend of ~10% every six months.
</div>
""", unsafe_allow_html=True)

# --- 4. THE HYBRID ENGINE (10% PER 6 MONTHS) ---
@st.cache_data
def get_market_data(ticker):
    return yf.download(ticker, period="5y", interval="1d")

def hybrid_devaluation_engine(df, horizon):
    data = df['Close'].values.flatten()
    vol = float(np.std(np.diff(data)))
    
    # 10% over 180 days = ~0.053% daily compounded
    daily_deval_constant = (1.10 ** (1/180)) - 1
    
    df_ml = pd.DataFrame({'C': data})
    for l in [1, 2, 3, 7, 14, 30]: df_ml[f'L{l}'] = df_ml['C'].shift(l)
    df_ml = df_ml.dropna()
    X, y = df_ml.drop('C', axis=1), df_ml['C']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    
    preds = []
    curr = X.iloc[-1].values.reshape(1, -1)
    
    for i in range(horizon):
        base_pred = float(model.predict(curr)[0])
        # Force the 10% rule into every step
        hybrid_step = base_pred * (1 + daily_deval_constant)
        noise = float(np.random.normal(0, vol * 0.4))
        final_step = hybrid_step + noise
        
        preds.append(final_step)
        new_feats = [final_step] + list(curr[0][:-1])
        curr = np.array([new_feats])
        
    return preds, vol

# --- 5. SIDEBAR ---
st.sidebar.markdown("**SYSTEM INPUTS**")
pairs = {"USD/ETB": "ETB=X", "EUR/ETB": "EURETB=X", "GBP/ETB": "GBPETB=X", "CNY/ETB": "CNYETB=X"}
selected = st.sidebar.selectbox("INSTRUMENT TICKET", list(pairs.keys()))
look_ahead = st.sidebar.slider("FORECAST RANGE (DAYS)", 30, 180, 180)

# --- 6. EXECUTION ---
try:
    df_raw = get_market_data(pairs[selected])
    if not df_raw.empty:
        df = df_raw[['Close']].copy()
        last_price = float(df['Close'].iloc[-1].item())
        
        with st.spinner("CALCULATING HYBRID CURVE..."):
            forecast, vol = hybrid_devaluation_engine(df, look_ahead)
        
        f_dates = [df.index[-1] + timedelta(days=i) for i in range(1, look_ahead+1)]
        
        # --- HYBRID TARGET BENCHMARKS ---
        target_tomorrow = last_price * (1.10 ** (1/180))
        target_3mo = last_price * (1.10 ** (90/180))
        target_6mo = last_price * 1.10

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("HYBRID: TOMORROW", f"{target_tomorrow:.2f}")
        col_b.metric("HYBRID: 3 MONTHS", f"{target_3mo:.2f}")
        col_c.metric("HYBRID: 6 MONTHS", f"{target_6mo:.2f}")

        # --- GRAPHING ---
        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4.2))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        ax.plot(df.index[-180:], df['Close'].tail(180).values.flatten(), color='#000000', linewidth=1.5, label='HISTORICAL')
        
        # Bridge & Projection
        ax.plot([df.index[-1], f_dates[0]], [last_price, forecast[0]], color='#2962FF', linewidth=2.5)
        ax.plot(f_dates, forecast, color='#2962FF', linewidth=2.5, label='HYBRID ML PROJECTION')
        
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=9, colors='#000000')
        ax.grid(True, linestyle='-', alpha=0.1, color='#000000')
        ax.legend(frameon=False, fontsize=9)
        st.pyplot(fig)

        # Strategic Calculator
        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        with c1:
            # FIX: Forced visibility for the number input
            t_day = st.number_input("DAYS TO SETTLEMENT", min_value=1, max_value=look_ahead, value=30, step=1)
        with c2:
            st.markdown(f'<div style="border: 2px solid #000000; padding: 15px; font-family: monospace; font-weight: 800; color: #000000; background-color: #FFFFFF;">EXECUTION RATE [{f_dates[t_day-1].strftime("%Y-%m-%d")}]: {forecast[t_day-1]:.2f} ETB</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"SYSTEM_EXCEPTION: {str(e)}")
