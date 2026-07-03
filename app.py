import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import av
import sys
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Lightweight Face Recognition", layout="wide")

# Menggunakan sys.modules untuk menyimpan database wajah agar aman diakses oleh semua thread
if "my_face_db" not in sys.modules:
    sys.modules["my_face_db"] = {}
registered_faces = sys.modules["my_face_db"]

# --- MENU NAVIGASI SAMPING ---
st.sidebar.title("Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", [" Halaman Live (WebRTC)", " Kamera IP (HP / CCTV)", " Daftar Wajah"])

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443", "turn:openrelay.metered.ca:443?transport=tcp"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]}
)

@st.cache_resource
def load_models():
    facenet = ort.InferenceSession("models/facenet.onnx")
    yunet = cv2.FaceDetectorYN.create(
        model="models/yunet.onnx",
        config="",
        input_size=(320, 320),
        score_threshold=0.8,
        nms_threshold=0.3,
        top_k=5000
    )
    return facenet, yunet

try:
    facenet_session, yunet_detector = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    models_loaded = False

def extract_features(img, session):
    img_resized = cv2.resize(img, (160, 160))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = (img_rgb.astype(np.float32) - 127.5) / 128.0
    img_tensor = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)
    inputs = {session.get_inputs()[0].name: img_tensor}
    return session.run(None, inputs)[0][0]

# --- FUNGSI UTAMA PEMROSESAN GAMBAR (Bisa dipakai WebRTC maupun IP Camera) ---
def process_frame_logic(img):
    if not models_loaded:
        return img
        
    height, width, _ = img.shape
    yunet_detector.setInputSize((width, height))
    
    _, faces = yunet_detector.detect(img)
    
    if faces is not None:
        for face in faces:
            box = list(map(int, face[:4]))
            x, y, w, h = box
            
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x+w), min(height, y+h)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            face_img = img[y1:y2, x1:x2]
            
            if face_img.shape[0] > 10 and face_img.shape[1] > 10:
                try:
                    live_feature = extract_features(face_img, facenet_session)
                    name = "Unknown"
                    highest_sim = -1
                    
                    for reg_name, reg_feat in registered_faces.items():
                        similarity = np.dot(live_feature, reg_feat) / (np.linalg.norm(live_feature) * np.linalg.norm(reg_feat))
                        if similarity > 0.65: 
                            if similarity > highest_sim:
                                highest_sim = similarity
                                name = f"{reg_name} ({similarity*100:.1f}%)"
                    
                    text_y = y1 - 10 if y1 - 10 > 20 else y1 + 20
                    cv2.putText(img, name, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception as e:
                    err_msg = str(e)[:20]
                    cv2.putText(img, f"Err: {err_msg}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return img

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    processed_img = process_frame_logic(img)
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# ==========================================
# HALAMAN 1: LIVE FACE RECOGNITION (WebRTC)
# ==========================================
if "Halaman Live" in menu:
    st.title("🔴 Live Face Recognition")
    
    if not registered_faces:
        st.warning("⚠️ Belum ada wajah terdaftar. Silakan ke halaman 'Daftar Wajah'.")
        
    webrtc_streamer(
        key="face-recognition",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# ==========================================
# HALAMAN 2: IP CAMERA (CCTV / Aplikasi HP)
# ==========================================
elif "Kamera IP" in menu:
    st.title("📹 Live dari IP Camera / Aplikasi HP")
    
    st.info("💡 **Petunjuk:** Gunakan aplikasi seperti **IP Webcam** di HP Android. Masukkan URL videonya ke kolom di bawah ini.")
    
    if not registered_faces:
        st.warning("⚠️ Belum ada wajah terdaftar. Wajah Anda akan berlabel 'Unknown'.")
        
    # Input URL untuk Kamera IP
    ip_url = st.text_input("URL IP Camera:", value="http://192.168.1.10:8080/video")
    
    run_ip_cam = st.checkbox("Mulai Kamera IP")
    
    FRAME_WINDOW = st.image([])
    
    if run_ip_cam:
        cap = cv2.VideoCapture(ip_url)
        if not cap.isOpened():
            st.error("Gagal terhubung! Pastikan URL benar dan HP dalam satu jaringan Wi-Fi dengan komputer ini.")
        else:
            while run_ip_cam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Koneksi terputus dari IP Camera.")
                    break
                
                # Resize sedikit agar proses ringan (opsional)
                frame = cv2.resize(frame, (640, 480))
                
                # Lempar ke fungsi processing yang sama persis dengan WebRTC
                processed_img = process_frame_logic(frame)
                
                # Streamlit membaca format RGB
                frame_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb)
            
            cap.release()

# ==========================================
# HALAMAN 3: DAFTAR WAJAH (REGISTRASI)
# ==========================================
elif "Daftar Wajah" in menu:
    st.title("📝 Registrasi Wajah")
    st.write("Silakan ambil foto untuk menyimpan data wajah Anda ke dalam sistem.")
    
    reg_name = st.text_input("Masukkan Nama Anda:")
    reg_image = st.camera_input("Ambil Foto Wajah Anda")
    
    if st.button("Simpan Data Wajah"):
        if reg_name and reg_image:
            file_bytes = np.asarray(bytearray(reg_image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            height, width, _ = img.shape
            yunet_detector.setInputSize((width, height))
            _, faces = yunet_detector.detect(img)
            
            if faces is not None:
                box = list(map(int, faces[0][:4]))
                x1, y1 = max(0, box[0]), max(0, box[1])
                x2, y2 = min(width, box[0]+box[2]), min(height, box[1]+box[3])
                
                img_with_box = img.copy()
                cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img_with_box, f"Wajah terdeteksi", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                st.image(img_with_box, channels="BGR", caption="Hasil Deteksi Wajah pada Pendaftaran")
                
                face_crop = img[y1:y2, x1:x2]
                
                if face_crop.shape[0] > 10 and face_crop.shape[1] > 10:
                    feat = extract_features(face_crop, facenet_session)
                    registered_faces[reg_name] = feat
                    st.success(f"✅ Wajah '{reg_name}' berhasil didaftarkan! Silakan buka halaman Live.")
                else:
                    st.error("Wajah terdeteksi tapi terlalu kecil/buram.")
            else:
                st.error("Wajah tidak terdeteksi di foto.")
        else:
            st.error("Kolom Nama dan Foto wajib diisi!")
            
    st.divider()
    st.write("### 🗃️ Data Wajah Tersimpan:")
    if registered_faces:
        for name in registered_faces.keys():
            st.info(f"👤 {name}")
    else:
        st.write("Belum ada data wajah.")
