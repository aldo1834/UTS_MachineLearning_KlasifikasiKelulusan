# decision_tree.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load dataset
df = pd.read_csv('data_kelulusan.csv')

# 2. Informasi awal
print(df.info())
print(df.describe())

# 3. Visualisasi target
sns.countplot(x='Status_Kelulusan', data=df)
plt.title('Distribusi Status Kelulusan')
plt.show()

# 4. Label encoding
le = LabelEncoder()
df['Status_Kelulusan'] = le.fit_transform(df['Status_Kelulusan'])

# 5. Pilih fitur dan target
X = df[['Nilai_Akademik', 'Kehadiran']]
y = df['Status_Kelulusan']

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model Decision Tree
model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X_train, y_train)

# 8. Evaluasi
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))
print("\nMatriks Kebingungan:\n", confusion_matrix(y_test, y_pred))

# 9. Visualisasi pohon
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=X.columns, class_names=le.classes_, filled=True)
plt.title("Visualisasi Decision Tree (Dataset Kelulusan)")
plt.show()

# 10. Simpan model
joblib.dump(model, '../models/decision_tree_kelulusan.pkl')
print("Model Decision Tree berhasil disimpan di folder 'models'.")

# 11. Contoh prediksi baru
data_baru = pd.DataFrame({
    'Nilai_Akademik': [80, 60],
    'Kehadiran': [85, 55]
})
hasil_prediksi = model.predict(data_baru)
hasil_label = le.inverse_transform(hasil_prediksi)
data_baru['Hasil_Prediksi'] = hasil_label
print(data_baru)

# ============================================================
# 11. Tambahan Model Pembanding (Logistic Regression)
# ============================================================
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)

print("\n===== HASIL LOGISTIC REGRESSION =====")
print(f"Akurasi: {accuracy_score(y_test, y_pred_lr):.2f}")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))
print("Matriks Kebingungan:")
print(confusion_matrix(y_test, y_pred_lr))

# ============================================================
# 12. Perbandingan Akurasi Dua Model
# ============================================================
acc_tree = accuracy_score(y_test, y_pred)
acc_lr = accuracy_score(y_test, y_pred_lr)

print("\n===== PERBANDINGAN MODEL =====")
print(f"Decision Tree Accuracy: {acc_tree:.2f}")
print(f"Logistic Regression Accuracy: {acc_lr:.2f}")

if acc_tree > acc_lr:
    print("Model terbaik: Decision Tree 🌳")
else:
    print("Model terbaik: Logistic Regression 📈")
