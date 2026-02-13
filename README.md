# 🚗 Car Price Prediction App

Web Application สำหรับทำนายราคารถยนต์มือสองด้วย Machine Learning (Multiple Linear Regression)
โปรเจกต์นี้เป็นส่วนหนึ่งของวิชา [รหัสวิชา/ชื่อวิชา]

🌐 **Live App:** [ใส่ Link Streamlit ของคุณตรงนี้]

## 📊 Model Performance
โมเดลมีความแม่นยำสูง โดยทดสอบกับข้อมูล Test Set 200 คัน:
- **R² Score:** 0.8210 (82.1%)
- **RMSE:** $2,212.99
- **MAE:** $1,788.27

## 🛠 Features
- **Input:** รับค่า Year, Engine Size, Mileage, Fuel Type, Transmission
- **Output:** ทำนายราคาเป็น USD ($) และแปลงเป็นเงินบาท (฿)
- **UI:** Premium Design รองรับการใช้งานบนมือถือ

## 📂 Project Structure
- `app.py`: ไฟล์หลักสำหรับรัน Web Application
- `train_model.py`: สคริปต์สำหรับ Train และ Evaluate Model
- `car_price_model.pkl`: โมเดลที่ Train เสร็จแล้ว
- `Car_Price_Prediction.csv`: Dataset

## 🚀 How to Run Locally
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run app: `streamlit run app.py`

---
Created by [Thanat Phumprasert]