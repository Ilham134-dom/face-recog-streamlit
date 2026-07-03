deploy ke streamlit sharing / cloud
1. buat repo di github
- git init 
- git add .
- git commit -m "first commit"
- git remote add origin https://github.com/Ilham134-dom/face-recog-streamlit.git
- git branch -M main
- git push -u origin main
2. push kodingan ke github
3. login di streamlit sharing
4. connect github ke streamlit sharing


kalau ada yang baru :
1. git add .
2. git commit -m "pesan baru"
3. git push -u origin main

---

# 📚 Penjelasan Logika & Struktur Kode (Materi Pengajaran)

Dokumentasi di bawah ini menjelaskan alur logika dari aplikasi `app.py`. Gunakan panduan ini untuk menjelaskan cara kerja aplikasi kepada penguji atau audiens.

## 1. Pengaturan Awal (Page Setup & CSS)
```python
st.set_page_config(page_title="Lightweight Face Recognition", layout="wide")
```
- **Fungsi:** Baris ini wajib berada di urutan paling atas. Tujuannya untuk memberikan judul pada *tab browser* dan membuat halaman mekar memenuhi layar (`layout="wide"`).
- **CSS Injeksi (`st.markdown`)**: Digunakan untuk memotong *padding* (ruang kosong warna putih) bawaan Streamlit agar tampilan aplikasi terlihat padat dan rapi.

## 2. Penyimpanan Data (Global Database)
```python
if "my_face_db" not in sys.modules:
    sys.modules["my_face_db"] = {}
registered_faces = sys.modules["my_face_db"]
```
- **Fungsi:** Aplikasi ini menggunakan struktur data *Dictionary* untuk menyimpan rekaman wajah.
- **Mengapa pakai `sys.modules`?** Umumnya Streamlit menggunakan `st.session_state` untuk menyimpan memori sementara. Namun, karena aplikasi ini menangani beban pemrosesan gambar yang berat, `sys.modules` digunakan agar data yang didaftarkan tidak hilang meskipun browser di-refresh secara tidak sengaja, serta aman dari bentrokan *Thread* memori.

## 3. Sistem *Caching* Model AI (@st.cache_resource)
```python
@st.cache_resource
def load_models():
    # ...
```
- **Fungsi Utama:** Ini adalah salah satu kunci efisiensi aplikasi ini. Jika fungsi pemuatan model (sebesar >20MB) dieksekusi terus-menerus setiap kali pengguna mengklik tombol, aplikasi akan menjadi lambat (*lag/crash*). 
- *Decorator* `@st.cache_resource` memastikan bahwa model **FaceNet** dan **YuNet** hanya dimuat **satu kali saja** ke dalam RAM (*Memory*) saat aplikasi pertama kali menyala.

## 4. Ekstraksi Ciri Wajah (FaceNet)
- Wajah yang sudah terpotong (*crop*) belum bisa langsung dikenali.
- Gambar potongan wajah tersebut di-*resize* secara paksa menjadi **160x160 piksel** (syarat mutlak dari model FaceNet).
- Selanjutnya dilakukan *Normalisasi Matematika* untuk mengubah nilai warna piksel (0 - 255) menjadi rentang `-1.0` hingga `1.0`. 
- Gambar dimasukkan ke dalam model (ONNX). Model akan "memeras" gambar tersebut dan mengeluarkan kumpulan angka array sepanjang **512 dimensi**. Angka 512 dimensi inilah yang disebut sebagai "Sidik Jari Digital" dari wajah tersebut.

## 5. Logika Pengenalan / Pencocokan Wajah
Proses pencocokan dilakukan di dalam fungsi `process_frame_logic(img)`:
1. **Deteksi (YuNet):** Pertama, YuNet menscan gambar dan mencari letak wajah (menghasilkan koordinat X, Y untuk digambar kotak hijaunya).
2. **Crop & Ekstrak:** Wajah dipotong dan dimasukkan ke FaceNet untuk mendapatkan angka 512 dimensi.
3. **Pencocokan (Cosine Similarity):** 
   - Komputer tidak bisa melihat gambar, mereka hanya melihat angka. Untuk mengetahui apakah wajah *live* sama dengan wajah di database, digunakan rumus matematika bernama **Cosine Similarity** (`np.dot`).
   - Rumus ini menghitung seberapa searah/mirip dua buah vektor wajah. Jika persentase kemiripannya di atas **65%**, maka sistem akan mendeklarasikannya sebagai orang yang sama dan menampilkan namanya di atas kotak hijau. Jika di bawah 65%, maka labelnya adalah "Unknown".

## 6. Integrasi dengan Streamlit Interface
Kode ini dipecah menjadi halaman (menu) yang diatur menggunakan `st.sidebar.radio`:
- **Halaman Snapshot:** Memanfaatkan fitur `st.camera_input` dari HTML5 bawaan browser. Gambar yang dijepret berbentuk data Bytes, kemudian di-decode kembali menjadi array matriks OpenCV (`cv2.imdecode`) agar bisa diproses AI.
- **Halaman Daftar Wajah:** Memiliki alur yang sama dengan Snapshot, hanya saja angka 512-dimensi yang dihasilkan tidak dipakai untuk "dicocokkan", melainkan disimpan/ditambahkan ke dalam *Dictionary* `registered_faces` beserta nama pengguna yang diinput.
