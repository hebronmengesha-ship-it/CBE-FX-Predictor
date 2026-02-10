import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
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

# --- 2. HIGH-CONTRAST MONOCHROME CSS ---
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

    /* Slider Fix: No Black Highlighting */
    div[data-testid="stSlider"] *, div[data-testid="stSelectSlider"] * {
        background-color: transparent !important;
        background: transparent !important;
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
st.markdown(f"<p style='font-size: 13px; font-weight: 700; margin-top:-15px;'>SYSTEM CLOCK: {date.today().strftime('%Y-%m-%d')} | ARCHITECTURE: FULL-HISTORY DEVALUATION-INFORMED ML</p>", unsafe_allow_html=True)
st.markdown("---")

# --- 4. THE HYBRID ENGINE (10Y TRAINING + 10% BIAS) ---
@st.cache_data
def get_market_data(ticker):
    # Training on 'max' to capture the 20s, 50s, and 100s
    return yf.download(ticker, period="max", interval="1d")

def informed_ensemble_arena(df, horizon):
    data = df['Close'].values.flatten()
    vol = float(np.std(np.diff(data[-365:]))) # volatility based on last year
    
    # 10% over 180 days = ~0.053% daily compounded adjustment
    daily_deval_bias = (1.10 ** (1/180)) - 1
    
    # ML Feature Prep
    df_ml = pd.DataFrame({'C': data})
    for l in [1, 2, 3, 7, 14, 30]: 
        df_ml[f'L{l}'] = df_ml['C'].shift(l)
    df_ml = df_ml.dropna()
    X, y = df_ml.drop('C', axis=1), df_ml['C']
    
    # Silent Battle for Accuracy on the last 60 days
    train_x, test_x = X.iloc[:-60], X.iloc[-60:]
    train_y = y.iloc[:-60]
    test_y = y.iloc[-60:]
    
    m_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(train_x, train_y)
    rmse_rf = np.sqrt(mean_squared_error(test_y, m_rf.predict(test_x)))
    winner = "INFORMED_RANDOM_FOREST"

    # Final Train on all data
    final_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    
    preds = []
    curr = X.iloc[-1].values.reshape(1, -1)
    
    for i in range(horizon):
        # 1. Prediction based on historical patterns (since the 20s)
        p = float(final_model.predict(curr)[0])
        
        # 2. Inform the ML with the 10% rule
        informed_step = p * (1 + daily_deval_bias)
        
        # 3. Add market intricacy (noise)
        step = float(informed_step + np.random.normal(0, vol * 0.4))
        preds.append(step)
        
        # Recursive update
        new_feats = [step] + list(curr[0][:-1])
        curr = np.array([new_feats])
        
    return preds, vol, winner

# --- 5. SYSTEM CONTROLS ---
st.sidebar.markdown("**SYSTEM INPUTS**")
pairs = {"USD/ETB": "ETB=X", "EUR/ETB": "EURETB=X", "GBP/ETB": "GBPETB=X", "CNY/ETB": "CNYETB=X"}
selected = st.sidebar.selectbox("INSTRUMENT TICKET", list(pairs.keys()))
look_ahead = st.sidebar.slider("FORECAST RANGE (DAYS)", 30, 180, 90)

# --- 6. CORE EXECUTION ---
try:
    df_raw = get_market_data(pairs[selected])
    if not df_raw.empty:
        df = df_raw[['Close']].copy()
        last_price = float(df['Close'].iloc[-1].item())
        
        with st.spinner("PROCESSING FULL HISTORY..."):
            forecast, vol, winner = informed_ensemble_arena(df, look_ahead)
        
        f_dates = [df.index[-1] + timedelta(days=i) for i in range(1, look_ahead+1)]
        final_p, low_b, high_b = [], [], []
        for i, v in enumerate(forecast):
            cone = vol * np.sqrt(i + 1) * 1.645
            final_p.append(v); low_b.append(v - cone); high_b.append(v + cone)

        m1, m2, m3 = st.columns(3)
        m1.metric("CURRENT SPOT", f"{last_price:.2f}")
        m2.metric("TARGET PROJECTION", f"{final_p[-1]:.2f}", delta=f"{((final_p[-1]/last_price)-1)*100:.1f}%")
        m3.metric("MODEL STATUS", winner.replace("_", " "))

        # --- GRAPHING (TRAINED ON MAX, DISPLAY 2Y) ---
        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4.2))
        
        # DISPLAY: Slicing to the last 2 years (approx 730 days)
        h_x = df.index[-730:]
        h_y = df['Close'].tail(730).values.flatten()
        
        ax.plot(h_x, h_y, color='#000000', linewidth=1.5, label='HISTORICAL (LAST 2Y)')
        
        # Bridge & Projection
        ax.plot([df.index[-1], f_dates[0]], [last_price, final_p[0]], color='#2962FF', linewidth=2.5)
        ax.plot(f_dates, final_p, color='#2962FF', linewidth=2.5, label='INFORMED PROJECTION')
        ax.fill_between(f_dates, low_b, high_b, color='#2962FF', alpha=0.08, label='90% RISK CORRIDOR')
        
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
