import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt

# --- Fungsi Utilitas Pemrosesan Gambar ---

def calculate_rgb_grayscale_histograms(image):
    """Menghitung dan menampilkan Histogram RGB dan Grayscale."""
    
    # Konversi ke Grayscale untuk histogram Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    st.subheader("1. Histogram RGB dan Grayscale")
    
    # 1. Histogram dalam bentuk Grafik (menggunakan Matplotlib)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # RGB Histogram
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        axes[0].plot(hist, color=col)
    axes[0].set_title('Histogram RGB')
    axes[0].set_xlim([0, 256])
    
    # Grayscale Histogram
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
    axes[1].plot(hist_gray, color='gray')
    axes[1].set_title('Histogram Grayscale')
    axes[1].set_xlim([0, 256])
    
    st.pyplot(fig)
    
    # 2. Histogram dalam bentuk Angka (Nilai Puncak Maksimum)
    st.markdown("#### Data Angka Histogram (Tinggi Puncak Maksimum)")
    col1, col2 = st.columns(2)
    
    # Mendapatkan nilai maksimum untuk setiap channel
    b_max = np.max(cv2.calcHist([image], [0], None, [256], [0, 256]))
    g_max = np.max(cv2.calcHist([image], [1], None, [256], [0, 256]))
    r_max = np.max(cv2.calcHist([image], [2], None, [256], [0, 256]))
    gray_max = np.max(hist_gray)

    with col1:
        st.write("**RGB:**")
        st.code(f"Puncak Biru (B): {int(b_max)} piksel")
        st.code(f"Puncak Hijau (G): {int(g_max)} piksel")
        st.code(f"Puncak Merah (R): {int(r_max)} piksel")
    
    with col2:
        st.write("**Grayscale:**")
        st.code(f"Puncak Grayscale: {int(gray_max)} piksel")
    
    return gray

# --- FUNGSI BARU UNTUK MENCARI NILAI PUNCAK (Digunakan di Thresholding) ---
def find_histogram_peaks(hist, num_peaks=2):
    """Mencari N nilai puncak tertinggi (lokasi intensitas) pada histogram."""
    hist_smoothed = cv2.GaussianBlur(hist, (5, 5), 0) 
    peak_indices = np.argsort(hist_smoothed.flatten())[::-1]
    
    unique_peaks = []
    for idx in peak_indices:
        if idx not in unique_peaks:
            unique_peaks.append(idx)
        if len(unique_peaks) >= num_peaks:
            break
            
    return sorted(unique_peaks)

def threshold_and_binarize(gray_image):
    """Menghitung Nilai Threshold Otsu, menampilkan dua nilai puncak, citra Biner, dan Histogramnya."""
    st.subheader("2. Thresholding dan Binarisasi")
    
    hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    peaks = find_histogram_peaks(hist_gray, num_peaks=2)
    
    # 1. Tampilkan Nilai Puncak
    st.markdown("#### Nilai Puncak Histogram Grayscale (Intensitas)")
    if len(peaks) >= 2:
        st.info(
            f"Dua nilai intensitas puncak tertinggi (lokasi) pada histogram: **{peaks[0]}** dan **{peaks[1]}**."
        )
    
    # 2. Hitung Nilai Threshold Otsu
    ret, binary_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    st.markdown("#### Hasil Thresholding (Otsu)")
    st.code(f"Nilai Threshold Otomatis (Otsu): {ret:.2f}")
    
    # --- TAMPILKAN CITRA DAN HISTOGRAM BINER ---
    
    col_img, col_hist = st.columns(2)
    
    with col_img:
        st.markdown("##### Citra Biner")
        st.image(binary_img, caption=f"Threshold: {ret:.2f}", channels="GRAY")
        
    with col_hist:
        st.markdown("##### Histogram Citra Biner")
        # Histogram citra biner hanya memiliki dua bar: di 0 dan 255
        hist_binary = cv2.calcHist([binary_img], [0], None, [256], [0, 256])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hist_binary, color='black')
        ax.set_title('Histogram Biner')
        ax.set_xlim([0, 256])
        st.pyplot(fig)
    
    return binary_img

def histogram_equalization_analysis(gray_image):
    """Melakukan Histogram Equalization, menampilkan hasilnya, dan Histogramnya."""
    st.subheader("3. Histogram Equalization (Uniform)")
    
    # --- Perhitungan Statistik Awal (Sebelum Equalization) ---
    mean_before = np.mean(gray_image)
    std_dev_before = np.std(gray_image)

    # --- Proses Equalization ---
    equalized_img = cv2.equalizeHist(gray_image)
    hist_after = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])

    # --- Perhitungan Statistik Setelah Equalization ---
    mean_after = np.mean(equalized_img)
    std_dev_after = np.std(equalized_img)
    
    # --- TAMPILKAN CITRA DAN HISTOGRAM EQUALIZATION ---
    
    col_img, col_hist = st.columns(2)

    with col_img:
        st.markdown("##### Citra Hasil Equalization")
        st.image(equalized_img, caption="Citra setelah Equalization", channels="GRAY")
    
    with col_hist:
        st.markdown("##### Histogram Hasil Equalization")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hist_after, color='blue')
        ax.set_title('Histogram Equalized')
        ax.set_xlim([0, 256])
        st.pyplot(fig)
    
    # --- Tampilkan Nilai Mean dan Standard Deviasi ---
    st.markdown("#### Analisis Mean dan Standard Deviasi")
    
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        st.markdown("**Sebelum Equalization**")
        st.code(f"Mean: {mean_before:.2f}")
        st.code(f"Standard Deviasi: {std_dev_before:.2f}")
    
    with col_stat2:
        st.markdown("**Setelah Equalization**")
        st.code(f"Mean: {mean_after:.2f}")
        st.code(f"Standard Deviasi: {std_dev_after:.2f}")

# --- Bagian Utama Streamlit App ---

def main():
    st.title("Aplikasi Pemrosesan Citra Dasar (Tugas Thresholding & Equalization) 🖼️")
    st.sidebar.header("Input Citra")

    uploaded_file = st.sidebar.file_uploader(
        "Pilih Citra...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        st.header("Citra Asli")
        st.image(img_rgb, caption="Citra yang Diinput", use_column_width=True)
        st.markdown("---")
        
        # 1. Histogram RGB dan Grayscale
        gray_image = calculate_rgb_grayscale_histograms(img_bgr) 
        
        st.markdown("---")

        # 2. Thresholding dan Binarisasi
        threshold_and_binarize(gray_image)
        
        st.markdown("---")

        # 3. Histogram Equalization dan Analisis Mean/Std Dev
        histogram_equalization_analysis(gray_image)

    else:
        st.info("Silakan unggah citra (JPG, JPEG, atau PNG) untuk memulai pemrosesan.")

if __name__ == "__main__":
    main()
