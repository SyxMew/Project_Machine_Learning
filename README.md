# 🩺 Heart Disease Risk Prediction using Ensemble Machine Learning

## 📘 Deskripsi Proyek
Proyek ini bertujuan untuk membangun *model prediksi risiko penyakit jantung* berdasarkan data klinis pasien.
Sistem ini dikembangkan menggunakan kombinasi dua algoritma — `Decision Tree` dan `K-Nearest Neighbors (KNN)` — yang digabungkan dalam `Ensemble Voting Classifier` untuk meningkatkan akurasi prediksi.

Model ini kemudian diimplementasikan ke dalam aplikasi interaktif berbasis `Streamlit`, yang memungkinkan pengguna memasukkan data pasien dan mendapatkan hasil prediksi risiko penyakit jantung secara langsung.

---

## 🧠 Kompleksitas Masalah
Permasalahan utama dalam proyek ini adalah *prediksi penyakit jantung* berdasarkan berbagai parameter klinis yang bersifat multivariat.
Kompleksitas muncul dari:
- Banyaknya fitur numerik dan kategorikal yang memerlukan preprocessing berbeda.
- Potensi ketidakseimbangan data antara pasien sehat dan pasien berisiko.
- Hubungan non-linear antar variabel yang membuat satu model tunggal kurang optimal.

Untuk mengatasinya, digunakan pendekatan *ensemble learning*, yang mengombinasikan kekuatan `Decision Tree` dan `KNN` untuk hasil prediksi yang lebih stabil dan akurat.

---

## 🧩 Dataset yang Digunakan
- *Sumber:* [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- *Jumlah Data:* 1,025 sampel pasien
- *Jumlah Fitur:* 14 kolom atribut (13 fitur + 1 target)
- *Format Data:* CSV (`heart.csv`)
- *Label Target:*
  - 0 = Tidak berisiko
  - 1 = Berisiko penyakit jantung
- *Jenis Dataset:* Real-world (berasal dari 4 basis data medis: Cleveland, Hungary, Switzerland, dan Long Beach)

---

## 🔍 Exploratory Data Analysis (EDA)
Langkah-langkah eksplorasi data meliputi:
- Analisis *distribusi label*, menunjukkan data relatif seimbang (526 positif, 499 negatif).
- Visualisasi *histogram fitur numerik* seperti usia, kolesterol, tekanan darah, dan detak jantung maksimal.
- Analisis *korelasi antar fitur* menggunakan heatmap untuk mengidentifikasi atribut paling berpengaruh.
- Pemeriksaan nilai hilang dan tipe data (tidak ditemukan missing value).

---

## 🧹 Preprocessing Data
Tahapan preprocessing yang dilakukan:
1. *Pembersihan Data:* Menghapus atribut identitas dan memastikan tidak ada nilai kosong.
2. *Normalisasi:* Fitur numerik dinormalisasi dengan min-max scaling agar tiap fitur berada pada rentang [0, 1].
3. *Encoding:* Fitur kategorikal dikonversi menggunakan `OneHotEncoder`.
4. *Pembagian Data:* Dataset dibagi menjadi data latih dan uji dengan rasio *80:20*.

---

## 🤖 Model yang Digunakan
- *`Decision Tree Classifier`*
  - Criterion: Gini
  - max_depth: None
  - min_samples_split: 2

- *`K-Nearest Neighbors (KNN)`*
  - n_neighbors: 9
  - weights: distance
  - p: 2 (Euclidean distance)

- *`Ensemble Voting Classifier`*
  - Kombinasi hard voting antara `Decision Tree` dan `KNN`
  - Meningkatkan stabilitas hasil prediksi dan menekan risiko overfitting

---

## 📈 Hasil Evaluasi
- *Akurasi (`Decision Tree`):* 98.54%
- *Akurasi (`KNN`):* 100.00%
- *Akurasi (`Ensemble`):* 98.54%
- *F1-Score (Rata-rata):* 0.99
- *ROC-AUC (`Ensemble`):* 1.0000

Model menunjukkan performa tinggi dan generalisasi yang baik pada data uji.

---

## ⚙ Teknologi yang Digunakan
- *Bahasa Pemrograman:* Python 3.10
- *Lingkungan Pengembangan:* Google Colab
- *Library Utama:*
  - `pandas`, `numpy` — untuk manipulasi data
  - `matplotlib`, `seaborn` — untuk visualisasi
  - `scikit-learn` — untuk pemodelan dan evaluasi
  - `joblib` — untuk menyimpan model
  - `streamlit` — untuk pembuatan aplikasi interaktif

---

## 🚀 Deployment dan Demo

### 📦 Aplikasi Streamlit
Aplikasi dikembangkan dengan `Streamlit` menggunakan model ensemble yang telah disimpan (`ensemble_model.joblib`).
Pengguna dapat menginput data klinis pasien melalui antarmuka sederhana dan mendapatkan hasil prediksi risiko secara instan.

### 🔗 Repository & Deployment
- *GitHub Repository:* [https://github.com/username/heart-disease-prediction](#)
- *Demo Streamlit (opsional):* [https://heartpredictor.streamlit.app](#)

### 🎥 Video Demo (Penjelasan):
1. *Tampilan UI:* Form input berisi 13 parameter klinis pasien.
2. *Contoh Input:* Usia, tekanan darah, kolesterol, dan lain-lain.
3. *Output:* Prediksi risiko penyakit jantung beserta probabilitas dan progress bar.
4. *Tahapan Deployment:*
   - Model dilatih dan disimpan dalam format `.joblib`.
   - Aplikasi dibangun menggunakan `Streamlit`.
   - Repository diunggah ke GitHub.
   - (Opsional) Deployment dilakukan melalui Streamlit Cloud.

---

## 📊 Lampiran
Berikut beberapa grafik hasil evaluasi model:
- Confusion Matrix (Ensemble)
- ROC Curve (Ensemble)
- Feature Importance (Decision Tree)
- Cuplikan UI Aplikasi Streamlit

---

## 👥 Pembagian Tugas Kelompok

| *Nama Anggota* | *Tugas Utama* |
|:-----------------|:----------------|
| *Helga Yuliani Putri Aritonang* | EDA, pengembangan model KNN, penyusunan laporan akhir |
| *[Anggota 2]* | Pembuatan model Decision Tree, ensemble integration, analisis evaluasi |
| *[Anggota 3]* | Pembuatan UI Streamlit, deployment, dokumentasi GitHub & video demo |

---

## 📚 Daftar Pustaka
- World Health Organization (2023). Cardiovascular diseases (CVDs). [https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))
- Kaggle. Heart Disease Dataset. [https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825–2830.

---

## 🧾 Lisensi
Proyek ini dibuat untuk keperluan akademik.
Bebas digunakan untuk tujuan pembelajaran dan penelitian non-komersial dengan mencantumkan atribusi kepada pengembang.

---

*© 2025 — Heart Disease Prediction Project*