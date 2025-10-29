# ============================================================
# Project UTS Machine Learning
# Judul: Klasifikasi Kelulusan Mahasiswa Berdasarkan Nilai Akademik dan Kehadiran
# Nama: Aldo Bagus Jiwantoro
# NIM: 231011400219
# ============================================================

# Import library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
data_path = os.path.join("data", "data_kelulusan.csv")
df = pd.read_csv(data_path)
print("=== Data Awal ===")
print(df.head(), "\n")

# ------------------------------------------------------------
# 2. Preprocessing
# ------------------------------------------------------------
le = LabelEncoder()
df['Status_Kelulusan'] = le.fit_transform(df['Status_Kelulusan'])

X = df[['Nilai_Akademik', 'Kehadiran']]
y = df['Status_Kelulusan']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------------
# 3. Training Model
# ------------------------------------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ------------------------------------------------------------
# 4. Evaluasi Model
# ------------------------------------------------------------
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

print("=== Hasil Evaluasi ===")
print(f"Akurasi Model: {akurasi * 100:.2f}%\n")
print("Laporan Klasifikasi:")
print(classification_report(y_test, y_pred), "\n")

# ------------------------------------------------------------
# 5. Simpan Model
# ------------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, os.path.join("models", "model_kelulusan.pkl"))
print("Model berhasil disimpan ke folder 'models'.\n")

# ------------------------------------------------------------
# 6. Prediksi Data Baru (Opsional)
# ------------------------------------------------------------
data_baru = pd.DataFrame({
    'Nilai_Akademik': [80, 60],
    'Kehadiran': [85, 55]
})
prediksi = model.predict(data_baru)
label_prediksi = le.inverse_transform(prediksi)
data_baru['Hasil_Prediksi'] = label_prediksi

print("=== Prediksi Data Baru ===")
print(data_baru)
