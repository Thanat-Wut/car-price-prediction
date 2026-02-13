import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page config
st.set_page_config(
    page_title="🚗 Car Price Predictor Premium",
    page_icon="🚗",
    layout="wide"
)

# USD to THB conversion
USD_TO_THB = 31.07

# Premium CSS + Animations
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}
    
.main-header {
    font-size: 4rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: 700;
    text-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
    
.hero-section {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
}
    
.input-card {
    background: rgba(255,255,255,0.95);
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}
    
.input-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 25px 50px rgba(0,0,0,0.15);
}
    
.prediction-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 3rem;
    border-radius: 25px;
    color: white;
    box-shadow: 0 25px 50px rgba(240,147,251,0.4);
    text-align: center;
    animation: pulse 2s infinite;
}
    
@keyframes pulse {
    0% { box-shadow: 0 25px 50px rgba(240,147,251,0.4); }
    50% { box-shadow: 0 25px 60px rgba(240,147,251,0.6); }
    100% { box-shadow: 0 25px 50px rgba(240,147,251,0.4); }
}
    
.metric-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 2rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 15px 35px rgba(79,172,254,0.4);
    height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
    
.thb-price {
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(45deg, #FFD700, #FFA500);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
    
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 50px;
    padding: 1rem 3rem;
    font-size: 1.2rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 10px 30px rgba(102,126,234,0.4);
}
    
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(102,126,234,0.6);
}
    
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}
    
.stMetric > div > div > div {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    with open('car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
    with open('metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
    return model, encoders, features, metrics

model, encoders, features, metrics = load_models()

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 class="main-header">🚗 ทำนายราคารถยนต์มือสอง</h1>
    <p style="font-size: 1.4rem; margin: 0; opacity: 0.9;">
        AI Powered Multiple Linear Regression • ความแม่นยำ 82% 
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar: Model Info
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem; border-radius: 15px; background: rgba(255,255,255,0.1); margin-bottom: 1rem;">
        <h3 style="color: white; text-align: center;">📊 ประสิทธิภาพโมเดล</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.metric("Test R² Score", f"{metrics['test_r2']:.4f}")
    st.metric("RMSE", f"฿{(metrics['rmse'] * USD_TO_THB):,.0f}")
    st.metric("MAE", f"฿{(metrics['mae'] * USD_TO_THB):,.0f}")
    
    st.markdown("---")
    st.subheader("🔧 คุณสมบัติที่ใช้")
    for f in features:
        st.markdown(f"• **{f}**")

# Input Section
st.markdown("# 📝 กรอกข้อมูลรถของคุณ")
input_row1, input_row2 = st.columns(2)

with input_row1:
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🚙 รายละเอียดหลัก")
        year = st.number_input("🗓️ ปีที่ผลิต", min_value=2000, max_value=2025, value=2015, step=1)
        engine_size = st.slider("⚙️ ขนาดเครื่องยนต์ (ลิตร)", 1.0, 6.0, 2.5, 0.1)
        mileage = st.number_input("🛣️ เลขไมล์ (กม.)", min_value=0, max_value=300000, value=50000, step=1000)
        st.markdown('</div>', unsafe_allow_html=True)

with input_row2:
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("⚙️ สเปคเครื่องยนต์")
        
        fuel_categories = encoders['Fuel_Type'].classes_.tolist()
        fuel_type = st.selectbox("⛽ ประเภทเชื้อเพลิง", fuel_categories)
        
        transmission_categories = encoders['Transmission'].classes_.tolist()
        transmission = st.radio("🔧 ระบบเกียร์", transmission_categories)
        st.markdown('</div>', unsafe_allow_html=True)

# Predict Button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("💎 ทำนายราคาทันที", type="primary", use_container_width=True, key="predict_premium")

# Prediction Results
if predict_button:
    with st.spinner("🔮 AI กำลังวิเคราะห์..."):
        input_data = pd.DataFrame({
            'Year': [year],
            'Engine_Size': [engine_size],
            'Mileage': [mileage],
            'Fuel_Type': [fuel_type],
            'Transmission': [transmission]
        })
        
        for col in ['Fuel_Type', 'Transmission']:
            input_data[col] = encoders[col].transform(input_data[col])
        
        prediction_usd = model.predict(input_data)[0]
        prediction_thb = prediction_usd * USD_TO_THB
        
        # Main Prediction Card
        col_pred1, col_pred2 = st.columns([3, 1])
        with col_pred1:
            st.markdown(f"""
            <div class="prediction-card">
                <h2 style="margin-bottom: 1rem;">🏆 ราคาที่คาดการณ์</h2>
                <h1 class="thb-price">฿{prediction_thb:,.0f}</h1>
                <p style="font-size: 1.2rem; opacity: 0.9; margin: 0;">
                    (USD ${prediction_usd:,.0f})
                </p>
                <p style="font-size: 1.1rem; margin-top: 1rem;">
                    ±฿{(metrics['rmse'] * USD_TO_THB):,.0f} (ความคลาดเคลื่อน)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics Cards
        metric_row = st.columns(3)
        with metric_row[0]:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 0.5rem 0;">R² Score</h4>
                <h2>{metrics['test_r2']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_row[1]:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 0.5rem 0;">RMSE</h4>
                <h2>฿{(metrics['rmse'] * USD_TO_THB):,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_row[2]:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 0.5rem 0;">MAE</h4>
                <h2>฿{(metrics['mae'] * USD_TO_THB):,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Input Summary
        with st.expander("📋 ดูข้อมูลที่ใช้ทำนาย", expanded=True):
            summary_col1, summary_col2 = st.columns(2)
            with summary_col1:
                st.success(f"✅ **ปี:** {year}")
                st.success(f"✅ **เครื่องยนต์:** {engine_size} ลิตร")
                st.success(f"✅ **เลขไมล์:** {mileage:,} กม.")
            with summary_col2:
                st.success(f"✅ **เชื้อเพลิง:** {fuel_type}")
                st.success(f"✅ **เกียร์:** {transmission}")

# Footer
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #666;'>
    <p>💎 สร้างด้วย <strong>Streamlit Premium UI</strong> | Multiple Linear Regression</p>
    <p>🎓 มหาวิทยาลัย [ชื่อ] | 2569 | 1 USD = ฿{:.2f}</p>
</div>
""".format(USD_TO_THB), unsafe_allow_html=True)
