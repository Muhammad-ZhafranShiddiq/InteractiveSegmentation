import streamlit as st
import numpy as np
import cv2
import math
import io
import base64
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers

st.set_page_config(layout="wide", page_title="Interactive Image Segmentation")

# Konstanta
DESIRED_WIDTH = 480
DESIRED_HEIGHT = 480
OVERLAY_COLOR = (0, 255, 255)  # Biru Cyan
MODEL_PATH = "magic_touch.tflite"

# Resize image agar proporsional
def resize(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    return img

# Konversi koordinat normalisasi ke piksel
def _normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    def is_valid_normalized_value(value):
        return (0 <= value <= 1)
    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

# Buat overlay image warna solid
def create_overlay(image, color=(0, 255, 255)):
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[:] = color
    return overlay

# Fungsi untuk membuat gambar dengan background transparan
def create_transparent_result(image_np, mask):
    """
    Membuat gambar dengan objek tetap berwarna dan background transparan.
    """
    height, width = image_np.shape[:2]
    rgba_image = np.zeros((height, width, 4), dtype=np.uint8)

    # BALIK MASK: background = mask > threshold
    background_mask = mask > 0.5  # ini adalah area BACKGROUND

    # Salin warna dari image asli
    rgba_image[:, :, :3] = image_np
    # Alpha = 0 di background, 255 di objek
    rgba_image[:, :, 3] = (~background_mask * 255).astype(np.uint8)

    return rgba_image



# Segmentasi dengan MediaPipe
def run_segmentation(image_np, x_click, y_click):
    RegionOfInterest = vision.InteractiveSegmenterRegionOfInterest
    NormalizedKeypoint = containers.keypoint.NormalizedKeypoint

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)

    with vision.InteractiveSegmenter.create_from_options(options) as segmenter:
        x_norm = float(x_click) / mp_image.width
        y_norm = float(y_click) / mp_image.height

        roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT, keypoint=NormalizedKeypoint(x_norm, y_norm))
        result = segmenter.segment(mp_image, roi)
        mask = result.category_mask.numpy_view()

        # Untuk tampilan dengan overlay
        alpha = np.stack((mask,) * 3, axis=-1) > 0.1
        alpha = alpha.astype(float) * 0.7
        overlay_image = create_overlay(image_np)
        blended = image_np * (1 - alpha) + overlay_image * alpha
        blended = blended.astype(np.uint8)

        # Tambahkan titik klik
        keypoint_px = _normalized_to_pixel_coordinates(x_norm, y_norm, mp_image.width, mp_image.height)
        if keypoint_px:
            cv2.circle(blended, keypoint_px, 9, (0, 0, 0), -1)
            cv2.circle(blended, keypoint_px, 6, (255, 255, 255), -1)

        # Buat versi transparan untuk download
        transparent_result = create_transparent_result(image_np, mask)

        return blended, transparent_result, mask

# Fungsi untuk mengonversi gambar ke base64
def get_image_download_link(img_array, filename, text):
    """
    Membuat link download untuk gambar
    """
    if img_array.shape[2] == 4:  # RGBA
        img_pil = Image.fromarray(img_array, 'RGBA')
        img_format = 'PNG'
    else:  # RGB
        img_pil = Image.fromarray(img_array, 'RGB')
        img_format = 'PNG'
    
    buffered = io.BytesIO()
    img_pil.save(buffered, format=img_format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}" style="text-decoration: none;"><button style="background-color: #ff4b4b; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 14px;">📥 {text}</button></a>'
    return href

# ------------------- Streamlit UI -------------------

st.title("🧠 Interactive Image Segmentation with MediaPipe")
st.markdown("*Klik pada objek yang ingin disegmentasi dan dapatkan hasil dengan background transparan*")

# Input mode selection
input_mode = st.radio("Pilih Input Gambar:", ("Unggah Gambar", "Gunakan Webcam"))

img = None

if input_mode == "Unggah Gambar":
    uploaded_file = st.file_uploader("Unggah Gambar", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        img = resize(image_np)
elif input_mode == "Gunakan Webcam":
    camera_input = st.camera_input("Ambil Gambar")
    if camera_input:
        image_pil = Image.open(camera_input).convert("RGB")
        image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        img = resize(image_np)

if img is not None:
    # Layout bersebelahan menggunakan columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Gambar Asli")
        st.markdown("*Klik pada objek yang ingin disegmentasi*")
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=1,
            stroke_color="#000000",
            background_color="#eee",
            background_image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
            update_streamlit=True,
            height=img.shape[0],
            width=img.shape[1],
            drawing_mode="point",
            point_display_radius=3,
            key="canvas",
        )

    with col2:
        st.subheader("🧪 Hasil Segmentasi")
        
        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            obj = canvas_result.json_data["objects"][-1]
            x_click = int(obj["left"])
            y_click = int(obj["top"])

            # Jalankan segmentasi
            segmented_img, transparent_img, mask = run_segmentation(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB), x_click, y_click
            )

            # Tampilkan hasil segmentasi
            st.image(segmented_img, caption="Segmentasi dengan overlay", use_column_width=True)
            
            # Area download dengan layout yang rapi
            st.markdown("---")
            st.markdown("### 📥 Download Opsi")
            
            # Buat tombol download dalam dua kolom
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                # Download hasil transparan (stiker)
                transparent_download = get_image_download_link(
                    transparent_img, 
                    "segmented_transparent.png", 
                    "Download Stiker"
                )
                st.markdown(transparent_download, unsafe_allow_html=True)
                st.caption("Format PNG dengan background transparan")
            
            with download_col2:
                # Download hasil dengan overlay
                overlay_download = get_image_download_link(
                    segmented_img, 
                    "segmented_overlay.png", 
                    "Download dengan Overlay"
                )
                st.markdown(overlay_download, unsafe_allow_html=True)
                st.caption("Format PNG dengan overlay berwarna")
            
            # Preview hasil transparan
            st.markdown("### 👁️ Preview Hasil Transparan")
            
            # Buat background kotak-kotak untuk menunjukkan transparansi
            checker_bg = np.zeros((transparent_img.shape[0], transparent_img.shape[1], 3), dtype=np.uint8)
            checker_size = 20
            for i in range(0, transparent_img.shape[0], checker_size):
                for j in range(0, transparent_img.shape[1], checker_size):
                    if (i // checker_size + j // checker_size) % 2 == 0:
                        checker_bg[i:i+checker_size, j:j+checker_size] = [200, 200, 200]
                    else:
                        checker_bg[i:i+checker_size, j:j+checker_size] = [255, 255, 255]
            
            # Composite transparent image dengan background kotak-kotak
            alpha_normalized = transparent_img[:, :, 3:4] / 255.0
            preview_img = (transparent_img[:, :, :3] * alpha_normalized + 
                          checker_bg * (1 - alpha_normalized)).astype(np.uint8)
            
            st.image(preview_img, caption="Preview dengan background transparan", use_column_width=True)
            
        else:
            st.info("👆 Klik pada gambar di sebelah kiri untuk mulai segmentasi")
            st.image("https://via.placeholder.com/400x300/f0f0f0/999999?text=Hasil+akan+muncul+di+sini", 
                    use_column_width=True)

# Informasi tambahan
with st.expander("ℹ️ Informasi Penggunaan"):
    st.markdown("""
    **Cara Menggunakan:**
    1. Pilih input gambar (unggah file atau gunakan webcam)
    2. Klik pada objek yang ingin disegmentasi di gambar sebelah kiri
    3. Hasil segmentasi akan muncul di sebelah kanan
    4. Download hasil dalam format:
       - **Stiker**: Background transparan (seperti fitur iPhone)
       - **Overlay**: Dengan highlight berwarna
    
    **Tips:**
    - Klik tepat di tengah objek untuk hasil terbaik
    - Format PNG mendukung transparansi
    - Hasil stiker dapat langsung digunakan untuk desain
    """)

# Footer
st.markdown("---")
st.markdown("*Dibuat dengan ❤️ menggunakan MediaPipe dan Streamlit*")