import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# 1. โหลดข้อมูล
print("Loading data...")
df = pd.read_csv('Car_Price_Prediction.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# 2. Data Preprocessing
print("\n" + "="*50)
print("Data Preprocessing...")

# เช็ค missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# ลบ missing values (ถ้ามี)
df = df.dropna()

# ⭐ เปลี่ยนชื่อ columns ให้เป็น underscore (แก้ปัญหา)
df.columns = df.columns.str.replace(' ', '_')
print(f"\nColumns after renaming: {df.columns.tolist()}")

# เลือก features (ไม่เกิน 5 features) - ใช้ชื่อใหม่
features = ['Year', 'Engine_Size', 'Mileage', 'Fuel_Type', 'Transmission']
target = 'Price'

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Fuel_Type', 'Transmission']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"\n{col} categories encoded: {le.classes_}")

print(f"\nFeatures used: {features}")
print(f"Target variable: {target}")

# 3. Split data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 4. Train Model
print("\n" + "="*50)
print("Training Multiple Linear Regression Model...")

model = LinearRegression()
model.fit(X_train, y_train)

print("✓ Model trained successfully!")

# 5. Evaluate Model
print("\n" + "="*50)
print("Model Evaluation...")

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)

print(f"\nTraining R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")

# แสดง coefficients
print("\n" + "="*50)
print("Feature Importance (Coefficients):")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# 6. Save Model
print("\n" + "="*50)
print("Saving model and encoders...")

# บันทึก model
with open('car_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# บันทึก label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# บันทึก feature names
with open('features.pkl', 'wb') as f:
    pickle.dump(features, f)

# บันทึก metrics สำหรับ web app
metrics = {
    'test_r2': test_r2,
    'train_r2': train_r2,
    'mae': mae,
    'rmse': rmse
}
with open('metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

print("✓ Model saved as 'car_price_model.pkl'")
print("✓ Encoders saved as 'label_encoders.pkl'")
print("✓ Features saved as 'features.pkl'")
print("✓ Metrics saved as 'metrics.pkl'")

# 7. Test prediction
print("\n" + "="*50)
print("Testing prediction with sample data...")

sample = X_test.iloc[0:1]
prediction = model.predict(sample)[0]
actual = y_test.iloc[0]

print(f"\nSample input:")
for feature in features:
    print(f"  {feature}: {sample[feature].values[0]}")
print(f"\nPredicted Price: ${prediction:.2f}")
print(f"Actual Price: ${actual:.2f}")
print(f"Difference: ${abs(prediction - actual):.2f}")
print(f"Error %: {abs(prediction - actual) / actual * 100:.2f}%")

print("\n" + "="*50)
print("✓ Training completed successfully!")
print("\nNext step: Run 'streamlit run app.py'")
