import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt

# --- Fungsi Utilitas Pemrosesan Gambar ---

def calculate_rgb_grayscale_histograms(image):
    """Menghitung dan menampilkan Histogram RGB dan Grayscale."""
    
    # Konversi ke Grayscale untuk histogram Grayscale dan Mean/Std Dev
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
    
    # 2. Histogram dalam bentuk Angka (Angka yang relevan, e.g., sumbu y)
    st.markdown("#### Data Angka Histogram (Nilai Puncak)")
    col1, col2 = st.columns(2)
    
    # Mendapatkan nilai maksimum untuk setiap channel
    b_max = np.max(cv2.calcHist([image], [0], None, [256], [0, 256]))
    g_max = np.max(cv2.calcHist([image], [1], None, [256], [0, 256]))
    r_max = np.max(cv2.calcHist([image], [2], None, [256], [0, 256]))
    gray_max = np.max(hist_gray)

    with col1:
        st.write("**RGB:**")
        st.code(f"Puncak Biru (B): {int(b_max)}")
        st.code(f"Puncak Hijau (G): {int(g_max)}")
        st.code(f"Puncak Merah (R): {int(r_max)}")
    
    with col2:
        st.write("**Grayscale:**")
        st.code(f"Puncak Grayscale: {int(gray_max)}")
    
    return gray

def threshold_and_binarize(gray_image):
    """Menghitung Nilai Threshold Otsu dan menampilkan Citra Biner."""
    st.subheader("2. Thresholding dan Binarisasi")
    
    # Gunakan Otsu's Binarization untuk mendapatkan threshold optimal
    # Otsu bekerja paling baik pada citra bimodal (dua puncak)
    # _ adalah nilai threshold yang dihitung
    
    ret, binary_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Tampilkan Nilai Threshold
    st.write(f"**Nilai Threshold Otsu:** `{ret:.2f}`")
    st.caption("Nilai threshold otomatis didapatkan menggunakan metode Otsu's Binarization, yang mengasumsikan histogram memiliki dua puncak.")
    
    # Tampilkan Citra Biner
    st.markdown("#### Citra Biner Hasil Thresholding")
    # Karena citra biner adalah grayscale, gunakan st.image dengan cmap='gray'
    st.image(binary_img, caption="Citra Biner (0 atau 255)", channels="GRAY")
    
    return binary_img

def histogram_equalization_analysis(gray_image):
    """Melakukan Histogram Equalization dan menampilkan hasilnya."""
    st.subheader("3. Histogram Equalization (Uniform)")
    
    # --- Perhitungan Statistik Awal (Sebelum Equalization) ---
    mean_before = np.mean(gray_image)
    std_dev_before = np.std(gray_image)

    # --- Proses Equalization ---
    # cv2.equalizeHist hanya bekerja pada citra Grayscale/Single-channel
    equalized_img = cv2.equalizeHist(gray_image)
    
    # --- Perhitungan Statistik Setelah Equalization ---
    mean_after = np.mean(equalized_img)
    std_dev_after = np.std(equalized_img)
    
    # --- Tampilkan Citra Hasil Equalization ---
    st.markdown("#### Citra Hasil Equalization")
    st.image(equalized_img, caption="Citra setelah Histogram Equalization", channels="GRAY")
    
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
    st.title("Aplikasi Pemrosesan Citra Dasar (Streamlit)")
    st.sidebar.header("Input Citra")

    # Upload File
    uploaded_file = st.sidebar.file_uploader(
        "Pilih Citra...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Baca file yang diupload
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # OpenCV membaca dalam format BGR
        img_bgr = cv2.imdecode(file_bytes, 1)
        # Konversi ke RGB untuk tampilan Streamlit/Matplotlib (karena Streamlit/Matplotlib default-nya RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        st.header("Citra Asli")
        st.image(img_rgb, caption="Citra yang Diinput", use_column_width=True)
        st.markdown("---")
        
        # 1. Histogram RGB dan Grayscale
        # Fungsi ini juga mengembalikan citra grayscale untuk digunakan pada fungsi berikutnya
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