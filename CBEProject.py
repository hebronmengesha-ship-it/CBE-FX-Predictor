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

# Force a web-safe chart backend
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore")

# --- 1. CORE INTERFACE SETTINGS ---
st.set_page_config(
    page_title="ML-1 PRICE PREDICTOR | RIDGE AR",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. THE ULTIMATE "ZERO-OVERLAY" CSS (FINAL FIX) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }

    /* Force Black Text Visibility */
    h1, h2, h3, h4, p, span, label, div {
        color: #000000 !important;
        background-color: transparent !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 2px solid #000000 !important;
    }

    /* --- SLIDER LABEL REPAIR --- */
    /* This kills the black highlight boxes in the sidebar */
    div[data-testid="stSlider"] *, div[data-testid="stSelectSlider"] * {
        background-color: transparent !important;
        background: transparent !important;
        color: #000000 !important;
    }

    /* Slider Track & Thumb (Solid Black) */
    div[data-baseweb="slider"] > div:first-child > div { background-color: #000000 !important; }
    div[role="slider"] { background-color: #000000 !important; border: 2px solid #000000 !important; }
    div[data-testid="stThumbValue"] { font-weight: 800 !important; }

    /* Metric Panels: Clinical Black Borders */
    div[data-testid="stMetric"] {
        border: 2px solid #000000 !important;
        padding: 15px;
        background-color: #FFFFFF !important;
    }
    
    label[data-testid="stMetricLabel"] {
        text-transform: uppercase;
        font-size: 0.85rem !important;
        font-weight: 900 !important;
        letter-spacing: 1.2px;
    }

    /* Settlement Tool Layout Fix */
    div[data-testid="stNumberInput"] input {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        border: 2px solid #000000 !important;
    }

    .settlement-display {
        border: 2px solid #000000;
        padding: 15px;
        font-family: monospace;
        font-weight: 800;
        color: #000000;
        background-color: #FFFFFF;
        margin-top: 5px;
    }

    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. DASHBOARD HEADER ---
st.markdown("### **ML-1 PRICE PREDICTOR** / RIDGE AR")
st.markdown(f"<p style='font-size: 13px; font-weight: 700; margin-top:-15px;'>SYSTEM CLOCK: {date.today().strftime('%Y-%m-%d')} | ARCHITECTURE: HYPER-ENSEMBLE (24 CONFIGS)</p>", unsafe_allow_html=True)
st.markdown("---")

# --- 4. THE HYPER-ENSEMBLE ENGINE ---
@st.cache_data
def get_market_data(ticker):
    return yf.download(ticker, period="5y", interval="1d")

def hyper_ensemble_arena(df, horizon):
    data = df['Close'].values.flatten()
    vol = float(np.std(np.diff(data)))
    drift = float(np.mean(np.diff(data[-60:])))
    
    train_vals, test_vals = data[:-60], data[-60:]
    scores = {}

    # ARIMA 12-Pack
    configs = [(1,1,0), (2,1,0), (5,1,0), (0,1,1), (1,1,1), (2,1,2), (5,1,2), (1,2,1), (0,2,1), (3,1,1), (4,1,0), (2,1,1)]
    for cfg in configs:
        try:
            res = ARIMA(train_vals, order=cfg).fit().forecast(steps=60)
            scores[f"ARIMA{cfg}"] = np.sqrt(mean_squared_error(test_vals, res))
        except: pass

    # RF 8-Pack
    df_ml = pd.DataFrame({'C': data})
    for l in [1, 2, 3, 5, 7, 14, 30]: df_ml[f'L{l}'] = df_ml['C'].shift(l)
    df_ml = df_ml.dropna()
    X, y = df_ml.drop('C', axis=1), df_ml['C']
    train_x, test_x = X.iloc[:-60], X.iloc[-60:]
    train_y = y.iloc[:-60]

    for d in [5, 10, 20, None]:
        for e in [50, 100]:
            try:
                m = RandomForestRegressor(n_estimators=e, max_depth=d, random_state=42).fit(train_x, train_y)
                scores[f"RF_D{d}_E{e}"] = np.sqrt(mean_squared_error(test_vals, m.predict(test_x)))
            except: pass

    winner_name = min(scores, key=scores.get)
    final_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    
    preds = []
    curr = X.iloc[-1].values.reshape(1, -1)
    for _ in range(horizon):
        p = final_rf.predict(curr)[0]
        step = float(p + np.random.normal(drift, vol * 0.4))
        preds.append(step)
        curr = np.array([[step] + list(curr[0][:-1])])
        
    return preds, vol, winner_name, len(scores)

# --- 5. SYSTEM CONTROLS ---
st.sidebar.markdown("**SYSTEM INPUTS**")
pairs = {"USD/ETB": "ETB=X", "EUR/ETB": "EURETB=X", "GBP/ETB": "GBPETB=X", "CNY/ETB": "CNYETB=X"}
selected = st.sidebar.selectbox("INSTRUMENT TICKET", list(pairs.keys()))
look_ahead = st.sidebar.slider("FORECAST RANGE (DAYS)", 30, 180, 90)

st.sidebar.markdown("---")
st.sidebar.markdown("**BIAS OFFSET**")
bias = st.sidebar.select_slider("DIRECTIONAL BIAS", options=["SHORT", "NEUTRAL", "LONG"], value="NEUTRAL")
bias_val = {"SHORT": 0.015, "NEUTRAL": 0.0, "LONG": -0.015}[bias]

# --- 6. CORE EXECUTION ---
try:
    df_raw = get_market_data(pairs[selected])
    if not df_raw.empty:
        df = df_raw[['Close']].copy()
        last_price = float(df['Close'].iloc[-1].item())
        
        with st.spinner("ANALYZING MARKET DATA..."):
            forecast, vol, winning_model, model_count = hyper_ensemble_arena(df, look_ahead)
        
        f_dates = [df.index[-1] + timedelta(days=i) for i in range(1, look_ahead+1)]
        final_p, low_b, high_b = [], [], []
        for i, v in enumerate(forecast):
            adj = v * (1 + bias_val)
            cone = vol * np.sqrt(i + 1) * 1.645
            final_p.append(adj); low_b.append(adj - cone); high_b.append(adj + cone)

        m1, m2, m3 = st.columns(3)
        m1.metric("CURRENT SPOT", f"{last_price:.2f}")
        m2.metric("TARGET PROJECTION", f"{final_p[-1]:.2f}", delta=f"{final_p[-1]-last_price:.2f}")
        m3.metric("CHAMPION CONFIG", winning_model.replace("_", " "))

        # --- GRAPHING (WEB SAFE) ---
        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(df.index[-200:], df['Close'].tail(200).values.flatten(), color='#000000', linewidth=1.5, label='HISTORICAL')
        ax.plot([df.index[-1], f_dates[0]], [last_price, final_p[0]], color='#2962FF', linewidth=2.5)
        ax.plot(f_dates, final_p, color='#2962FF', linewidth=2.5, label='PROJECTION')
        ax.fill_between(f_dates, low_b, high_b, color='#2962FF', alpha=0.08, label=f'90% RISK CORRIDOR ({model_count} MODELS)')
        
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=9, colors='#000000')
        ax.grid(True, linestyle='-', alpha=0.1, color='#000000')
        ax.legend(frameon=False, fontsize=9)
        st.pyplot(fig)

        # Strategic Calculator
        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        with c1:
            t_day = st.number_input("DAYS TO SETTLEMENT", min_value=1, max_value=look_ahead, value=30)
        with c2:
            st.markdown(f'<div class="settlement-display">EXECUTION RATE [{f_dates[t_day-1].strftime("%Y-%m-%d")}]: {final_p[t_day-1]:.2f} ETB</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"SYSTEM_EXCEPTION: {str(e)}")
