import streamlit as st
import pandas as pd
import joblib
import os

MODEL_FILE = 'ensemble_model.joblib'

@st.cache_resource
def load_model(filepath):
    """Memuat model joblib yang sudah dilatih."""
    if not os.path.exists(filepath):
        st.error(f"Error: File model '{filepath}' tidak ditemukan.")
        st.error("Pastikan Anda telah menjalankan 'complete_heart_pipeline.py' terlebih dahulu.")
        return None
    try:
        model = joblib.load(filepath)
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

model = load_model(MODEL_FILE)

st.set_page_config(page_title="Prediktor Penyakit Jantung", page_icon="🩺", layout="wide")
st.title("🩺 Prediktor Risiko Penyakit Jantung")
st.write("Aplikasi ini memprediksi risiko penyakit jantung menggunakan model *Ensemble* (Decision Tree + KNN).")
st.write("---")

st.sidebar.header("Masukkan Data Pasien")

st.sidebar.subheader("Data Klinis")
age = st.sidebar.slider("1. Usia (tahun)", min_value=18, max_value=100, value=55)
trestbps = st.sidebar.slider("2. Tekanan Darah Istirahat (mmHg)", min_value=80, max_value=220, value=120)
chol = st.sidebar.slider("3. Kolesterol Serum (mg/dL)", min_value=100, max_value=600, value=210)
thalach = st.sidebar.slider("4. Detak Jantung Maksimal (bpm)", min_value=60, max_value=220, value=150)
oldpeak = st.sidebar.slider("5. Depresi ST (oldpeak)", min_value=0.0, max_value=7.0, value=1.5, step=0.1)

st.sidebar.subheader("Data Kategorikal")

sex_map = {1: 'Pria', 0: 'Wanita'}
cp_map = {0: 'Angina Tipikal', 1: 'Angina Atipikal', 2: 'Nyeri Non-angina', 3: 'Asimtomatik'}
fbs_map = {1: 'Ya (> 120 mg/dL)', 0: 'Tidak (<= 120 mg/dL)'}
restecg_map = {0: 'Normal', 1: 'Abnormalitas ST-T', 2: 'Hipertrofi Ventrikel'}
exang_map = {1: 'Ya', 0: 'Tidak'}
slope_map = {0: 'Naik (Upsloping)', 1: 'Datar (Flat)', 2: 'Turun (Downsloping)'}
ca_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}
thal_map = {1: 'Normal', 2: 'Cacat Tetap (Fixed Defect)', 3: 'Cacat Reversibel (Reversible Defect)', 0: 'N/A'}

sex = st.sidebar.selectbox("6. Jenis Kelamin", options=list(sex_map.keys()), format_func=lambda x: sex_map[x])
cp = st.sidebar.selectbox("7. Tipe Nyeri Dada (cp)", options=list(cp_map.keys()), format_func=lambda x: cp_map[x])
fbs = st.sidebar.selectbox("8. Gula Darah Puasa > 120 mg/dL (fbs)", options=list(fbs_map.keys()), format_func=lambda x: fbs_map[x])
restecg = st.sidebar.selectbox("9. Hasil EKG Istirahat (restecg)", options=list(restecg_map.keys()), format_func=lambda x: restecg_map[x])
exang = st.sidebar.selectbox("10. Nyeri Dada Akibat Olahraga (exang)", options=list(exang_map.keys()), format_func=lambda x: exang_map[x])
slope = st.sidebar.selectbox("11. Kemiringan Segmen ST (slope)", options=list(slope_map.keys()), format_func=lambda x: slope_map[x])
ca = st.sidebar.selectbox("12. Jumlah Pembuluh Darah Utama (ca)", options=list(ca_map.keys()), format_func=lambda x: ca_map[x])
thal = st.sidebar.selectbox("13. Status Thalassemia (thal)", options=list(thal_map.keys()), format_func=lambda x: thal_map[x])

if st.sidebar.button("Prediksi Risiko", type="primary"):
    if model is not None:
        data = {
            'age': age,
            'trestbps': trestbps,
            'chol': chol,
            'thalach': thalach,
            'oldpeak': oldpeak,
            'sex': sex,
            'cp': cp,
            'fbs': fbs,
            'restecg': restecg,
            'exang': exang,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }

        input_df = pd.DataFrame([data])
        
        try:
            pred = model.predict(input_df)
            proba = model.predict_proba(input_df)
            
            prediction_value = pred[0]
            probability_value = proba[0][1] 

            st.subheader("--- Hasil Output Model ---")
            
            if prediction_value == 1:
                st.error(f"💡 Prediksi: Berisiko tinggi terhadap penyakit jantung")
            else:
                st.success(f"💡 Prediksi: Berisiko rendah terhadap penyakit jantung")

            st.metric(label="Probabilitas Berisiko", value=f"{probability_value*100:.0f}%")
            
            st.progress(probability_value)
            st.caption(f"Angka probabilitas mentah: {probability_value:.4f}")
            
            with st.expander("Lihat Data Input yang Digunakan"):
                st.dataframe(input_df)

        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
    else:
        st.error("Model tidak dapat dimuat. Prediksi dibatalkan.")
else:
    st.info("Silakan masukkan data pasien di sidebar kiri dan tekan tombol 'Prediksi Risiko'.")