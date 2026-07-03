import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import sys

# ==========================================
# 1. PENGATURAN AWAL HALAMAN (PAGE SETUP)
# ==========================================
# Perintah ini harus dipanggil paling pertama untuk mengatur judul tab browser dan lebar halaman.
st.set_page_config(page_title="Lightweight Face Recognition", layout="wide")

# ==========================================
# 2. CUSTOM CSS (DESAIN TAMPILAN)
# ==========================================
# Menyuntikkan kode CSS untuk menghilangkan ruang kosong/putih (padding) bawaan Streamlit
# di bagian paling atas halaman agar aplikasi terlihat lebih padat dan rapi.
st.markdown("""
<style>
    /* Memotong ruang kosong di atas halaman utama */
    .block-container {
        padding-top: 2rem !important;
    }
    /* Memotong ruang kosong di atas menu samping (sidebar) */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. PENYIMPANAN DATA (DATABASE SEMENTARA)
# ==========================================
# Kita menggunakan `sys.modules` untuk menyimpan data wajah (database). 
# Kenapa tidak pakai st.session_state? Karena variabel di sys.modules akan benar-benar 
# bertahan selamanya di memori server meskipun halaman di-refresh, 
# dan lebih aman saat diakses oleh banyak proses (thread).
if "my_face_db" not in sys.modules:
    sys.modules["my_face_db"] = {}
# registered_faces adalah kamus (dictionary) yang berisi pasangan: {"Nama": [Vektor_Wajah]}
registered_faces = sys.modules["my_face_db"]

# ==========================================
# 4. MEMBANGUN MENU NAVIGASI (SIDEBAR)
# ==========================================
st.sidebar.markdown("##  IRIS ")
st.sidebar.caption("Sistem Pengenalan Wajah Berbasis AI")
st.sidebar.divider() # Membuat garis pembatas

st.sidebar.write("**Menu Navigasi:**")
# Membuat radio button untuk memilih halaman. Nilai yang dipilih disimpan di variabel 'menu'
menu = st.sidebar.radio("Pilih Halaman:", ["Snapshot", "Daftar Wajah"])

# ==========================================
# 5. MEMUAT MODEL AI (CACHE RESOURCE)
# ==========================================
# @st.cache_resource sangat penting! Ini mencegah Streamlit memuat (me-load) ulang 
# model AI berukuran besar setiap kali pengguna mengklik tombol di layar.
# Model hanya dimuat 1 kali saat server baru menyala.
@st.cache_resource
def load_models():
    # 1. Memuat model FaceNet (Pengenal Wajah) berformat ONNX
    facenet = ort.InferenceSession("models/facenet.onnx")
    
    # 2. Memuat model YuNet (Pendeteksi Lokasi Wajah) bawaan OpenCV
    yunet = cv2.FaceDetectorYN.create(
        model="models/yunet.onnx",
        config="",
        input_size=(320, 320), # Ukuran gambar awal yang diharapkan model
        score_threshold=0.8,   # Tingkat kepercayaan minimal (80%) agar dianggap sebagai wajah
        nms_threshold=0.3,
        top_k=5000
    )
    return facenet, yunet

# Panggil fungsi pemuatan model di atas
try:
    facenet_session, yunet_detector = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    models_loaded = False

# ==========================================
# 6. FUNGSI EKSTRAKSI CIRI WAJAH (FACENET)
# ==========================================
def extract_features(img, session):
    # Model FaceNet membutuhkan gambar berukuran tepat 160x160 piksel
    img_resized = cv2.resize(img, (160, 160))
    # OpenCV membaca gambar dalam format BGR (Biru-Hijau-Merah), kita ubah ke RGB (Merah-Hijau-Biru)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalisasi piksel: mengubah nilai warna dari 0-255 menjadi rentang -1.0 hingga 1.0
    img_normalized = (img_rgb.astype(np.float32) - 127.5) / 128.0
    
    # Mengubah urutan dimensi array dari (Tinggi, Lebar, Warna) menjadi (Warna, Tinggi, Lebar) sesuai standar PyTorch/ONNX
    img_tensor = np.transpose(img_normalized, (2, 0, 1))
    
    # Menambahkan dimensi Batch di depan, menjadi (1, Warna, Tinggi, Lebar)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    # Memasukkan matriks gambar ke dalam otak AI (FaceNet)
    inputs = {session.get_inputs()[0].name: img_tensor}
    
    # AI akan mengembalikan 512 angka (vektor 512 dimensi) yang merupakan "KTP/Sidik Jari" dari wajah tersebut
    return session.run(None, inputs)[0][0]

# ==========================================
# 7. FUNGSI UTAMA PENGENALAN WAJAH (LOGIC)
# ==========================================
# Fungsi ini menerima gambar utuh, mencari wajah, mencocokkannya, lalu menggambar kotak & nama
def process_frame_logic(img):
    if not models_loaded:
        return img
        
    height, width, _ = img.shape
    # Beri tahu YuNet ukuran gambar yang sedang diproses agar deteksi akurat
    yunet_detector.setInputSize((width, height))
    
    # Mencari wajah di dalam gambar
    _, faces = yunet_detector.detect(img)
    
    # Jika wajah ditemukan...
    if faces is not None:
        for face in faces:
            # Ambil koordinat kotak wajah: x (kiri), y (atas), w (lebar), h (tinggi)
            box = list(map(int, face[:4]))
            x, y, w, h = box
            
            # Mencegah error jika koordinat wajah keluar dari batas ukuran foto
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x+w), min(height, y+h)
            
            # Gambar kotak berwarna hijau (0, 255, 0) setebal 2 piksel
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Crop/potong gambar aslinya, ambil bagian wajahnya saja
            face_img = img[y1:y2, x1:x2]
            
            # Pastikan potongan wajah tidak terlalu kecil (mencegah error dimensi)
            if face_img.shape[0] > 10 and face_img.shape[1] > 10:
                try:
                    # Dapatkan 512 angka ciri wajah dari potongan gambar
                    live_feature = extract_features(face_img, facenet_session)
                    
                    name = "Unknown"
                    highest_sim = -1
                    
                    # PROSES PENCARIAN (COSINE SIMILARITY)
                    # Bandingkan vektor wajah orang ini dengan seluruh vektor wajah yang ada di database
                    for reg_name, reg_feat in registered_faces.items():
                        # Rumus Matematika (Cosine Similarity): Mencari seberapa mirip arah dua vektor (menghasilkan angka -1 hingga 1)
                        similarity = np.dot(live_feature, reg_feat) / (np.linalg.norm(live_feature) * np.linalg.norm(reg_feat))
                        
                        # Jika kemiripannya di atas 65% (0.65), maka dianggap orang yang sama
                        if similarity > 0.65: 
                            if similarity > highest_sim:
                                highest_sim = similarity
                                name = f"{reg_name} ({similarity*100:.1f}%)"
                    
                    # Logika penempatan teks: Jika posisi kotak terlalu di atas layar, pindahkan tulisan ke bawah kotak
                    text_y = y1 - 10 if y1 - 10 > 20 else y1 + 20
                    cv2.putText(img, name, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception as e:
                    # Jika gagal diproses, munculkan pesan error merah di layar (tidak membuat web crash)
                    err_msg = str(e)[:20]
                    cv2.putText(img, f"Err: {err_msg}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return img

# ==========================================
# 8. HALAMAN 1: SNAPSHOT (DETEKSI)
# ==========================================
if "Snapshot" in menu:
    st.title("Face Recognition")    
    if not registered_faces:
        st.warning(" Belum ada wajah terdaftar. Wajah Anda akan berlabel 'Unknown'.")
        
    # Mengambil gambar langsung dari webcam menggunakan elemen bawaan browser
    camera_image = st.camera_input("Ambil Foto untuk Dikenali")
    
    if camera_image is not None:
        # Konversi data gambar dari format RAM web (BytesIO) menjadi matriks angka yang dimengerti OpenCV (Numpy Array)
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1) # '1' berarti baca sebagai gambar berwarna (BGR)
        
        # Lempar gambar mentah ke otak logika kita di atas
        processed_img = process_frame_logic(img)
        
        # Tampilkan hasil gambarnya kembali ke halaman web (Jangan lupa ubah BGR ke format RGB standar Streamlit)
        st.image(processed_img, channels="BGR", caption="Hasil Pengenalan Wajah", use_column_width=True)


# ==========================================
# 9. HALAMAN 2: DAFTAR WAJAH (REGISTRASI)
# ==========================================
elif "Daftar Wajah" in menu:
    st.title("Registrasi Wajah")
    st.write("Silakan ambil foto untuk menyimpan data wajah Anda ke dalam sistem.")
    
    # Input kolom teks untuk nama orang
    reg_name = st.text_input("Masukkan Nama Anda:")
    reg_image = st.camera_input("Ambil Foto Wajah Anda")
    
    if st.button("Simpan Data Wajah"):
        # Pastikan kolom nama tidak kosong dan foto sudah dijepret
        if reg_name and reg_image:
            # Konversi foto dari Bytes web ke OpenCV Array
            file_bytes = np.asarray(bytearray(reg_image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            # Tahap Validasi: Kita wajib mengecek apakah di foto pendaftaran benar-benar ADA wajahnya
            height, width, _ = img.shape
            yunet_detector.setInputSize((width, height))
            _, faces = yunet_detector.detect(img)
            
            # Jika ada minimal 1 wajah yang terdeteksi
            if faces is not None:
                box = list(map(int, faces[0][:4])) # Fokus ambil wajah pertama saja (faces[0])
                x1, y1 = max(0, box[0]), max(0, box[1])
                x2, y2 = min(width, box[0]+box[2]), min(height, box[1]+box[3])
                
                # Menggambar kotak biru sebagai bukti ke layar bahwa wajah berhasil dideteksi
                img_with_box = img.copy()
                cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img_with_box, f"Wajah terdeteksi", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                st.image(img_with_box, channels="BGR", caption="Hasil Deteksi Wajah pada Pendaftaran")
                
                # Potong wajah dan ambil vektor ciri-cirinya
                face_crop = img[y1:y2, x1:x2]
                
                if face_crop.shape[0] > 10 and face_crop.shape[1] > 10:
                    feat = extract_features(face_crop, facenet_session)
                    
                    # SIMPAN KE DATABASE GLOBAL
                    registered_faces[reg_name] = feat
                    st.success(f"✅ Wajah '{reg_name}' berhasil didaftarkan! Silakan buka halaman Snapshot.")
                else:
                    st.error("Wajah terdeteksi tapi terlalu kecil/buram.")
            else:
                st.error("Wajah tidak terdeteksi di foto. Pastikan pencahayaan cukup.")
        else:
            st.error("Kolom Nama dan Foto wajib diisi!")
            
    st.divider()
    
    # Fitur tambahan: Menampilkan siapa saja yang sudah mendaftar di memori server saat ini
    st.write("### 🗃️ Data Wajah Tersimpan:")
    if registered_faces:
        for name in registered_faces.keys():
            st.info(f"👤 {name}")
    else:
        st.write("Belum ada data wajah.")
