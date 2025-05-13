# Streamlit Wizard for BLIP Image Captioning – Updated with CAM & Cross-Attention
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from PIL import Image, ImageOps
from pytorch_grad_cam import EigenCAM, KPCA_CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import Saliency
import torchvision
from transformers import BlipProcessor, BlipForConditionalGeneration
import io, math

# --- Setup ---
st.set_page_config(page_title="X-Capindo - Visualisasi Proses Image Captioning", page_icon="imgs/logo.png", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #4B5563; text-align: center; margin-bottom: 2rem; }
    .step-header { font-size: 1.8rem; font-weight: bold; color: #1E3A8A; margin-bottom: 1rem; }
    .step-description { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .tech-details { background-color: #EFF6FF; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1.5rem; }
    .stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="main-header">Visualisasi Proses Image Captioning</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Pelajari tahapan bagaimana AI menghasilkan deskripsi dari gambar</div>', unsafe_allow_html=True)

# --- Sidebar Info ---
with st.sidebar.expander("ℹ️ Tentang X-Capindo", expanded=False):
    st.image("imgs/logo.png", width=90)
    st.markdown("""
**X-Capindo** adalah aplikasi captioning berbasis BLIP (Salesforce) yang dilengkapi dengan penjelasan visual melalui CAM (Class Activation Maps), attention transformer, dan feature map.

**Langkah-langkah**:
1. Unggah gambar yang ingin dideskripsikan.
2. Lihat area fokus menggunakan CAM.
3. Tinjau feature map dan cross-attention.
4. Amati proses decoding.
5. Dapatkan caption akhir.
    """)
    st.markdown("""
**Model**: BLIP (Base) – `Salesforce/blip-image-captioning-base`

**Teknik XAI**:
- EigenCAM / KPCA-CAM
- Attention Rollout
- Saliency Map
- Cross-Attention Decoder
    """)

# --- Model Load (once) ---
@st.cache_resource
def load_model(model_choice="BLIP-Base (local)"):
    if model_choice == "BLIP-Base (local)":
        model_path = r"models/blip-image-captioning-base/v1.0"
    elif model_choice == "BLIP-Large (local)":
        model_path = r"models/blip-image-captioning-large/v1.0"
    elif model_choice == "BLIP-Large (HF Hub)":
        model_path = snapshot_download(repo_id=r"HusniFd/blip-image-captioning-large")
    elif model_choice == "BLIP-Base (HF Hub)":
        model_path = snapshot_download(repo_id=r"HusniFd/blip-image-captioning-base")
    else:
        raise ValueError("Model tidak dikenali.")
    processor = BlipProcessor.from_pretrained(model_path)
    model = BlipForConditionalGeneration.from_pretrained(model_path).to("cpu")
    return processor, model

processor, model = load_model()

# --- Image Upload Helper ---
@st.cache_data
def load_uploaded_image(img):
    if isinstance(img, str):
        image = Image.open(img)
    else:
        image = Image.open(io.BytesIO(img.read())).convert("RGB")
    return ImageOps.exif_transpose(image)

# --- Step logic ---
steps = [
    {"title": "Unggah Gambar", "description": "Sistem menerima gambar dari pengguna", "tech_details": "Gambar diubah menjadi tensor 3-channel dengan normalisasi dan penskalaan untuk digunakan model."},
    {"title": "Peta Aktivasi (CAM)", "description": "Visualisasi fokus model pada gambar", "tech_details": "Menggunakan teknik explainability: EigenCAM, KPCA-CAM, Attention Rollout, dan Saliency Map untuk menampilkan area penting yang diproses model."},
    {"title": "Feature Map", "description": "Ekstraksi dan visualisasi saluran fitur dari encoder", "tech_details": "Menampilkan 16 channel awal dari token visual BLIP (tanpa CLS), disusun dalam grid 4x4 sebagai representasi patch embedding."},
    {"title": "Cross-Attention per Kata", "description": "Visualisasi atensi model terhadap setiap kata yang dihasilkan", "tech_details": "Membandingkan peta atensi cross-attention decoder dengan hasil Grad-CAM untuk setiap token dalam caption."},
    {"title": "Generasi Caption", "description": "Caption akhir berdasarkan analisis visual dan bahasa", "tech_details": "Model BLIP menghasilkan deskripsi kata demi kata dengan bantuan beam search dan pengaruh dari representasi visual."},
]

# --- Session state init ---
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "image" not in st.session_state:
    st.session_state.image = None

# --- Step Navigator ---
progress_bar = st.progress(st.session_state.current_step / (len(steps) - 1))
cols = st.columns(len(steps))
for i, col in enumerate(cols):
    with col:
        if st.button(f"{i+1}. {steps[i]['title']}", key=f"step_{i}"):
            st.session_state.current_step = i
            st.rerun()

current_step = steps[st.session_state.current_step]
st.markdown(f'<div class="step-header">Langkah {st.session_state.current_step + 1}: {current_step["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="step-description">{current_step["description"]}</div>', unsafe_allow_html=True)
with st.expander("Tampilkan Detail Teknis"):
    st.markdown(f'<div class="tech-details">{current_step["tech_details"]}</div>', unsafe_allow_html=True)

# --- Step 1: Image Upload ---
if st.session_state.current_step == 0:
    col_left, col_right = st.columns([1, 2])
    with col_left:
        if st.session_state.image is None:
            st.image("imgs/test5.jpg", caption="Contoh Gambar Input", use_container_width=True)
        else:
            st.image(st.session_state.image, caption="🖼️ Gambar yang Diunggah", use_container_width=True)
        st.caption("Format yang didukung: JPG, PNG, WEBP")
    with col_right:
        uploaded_file = st.file_uploader("Unggah gambar Anda", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file:
            st.session_state.image = load_uploaded_image(uploaded_file)
            st.rerun()

# --- Step 2: Visualisasi CAM ---
if st.session_state.current_step == 1:
    st.subheader("🔍 Pilih Teknik Visualisasi Fokus Gambar")
    selected_cam = st.selectbox("Metode Explainability", ["EigenCAM", "KPCA-CAM", "Attention Rollout", "Saliency Map"])

    image = st.session_state.image or load_uploaded_image("imgs/test5.jpg")
    input_tensor = processor(images=image, return_tensors="pt").pixel_values
    rgb_image = np.array(image).astype(np.float32) / 255.0

    class BlipWrapper(torch.nn.Module):
        def __init__(self, patch_module): super().__init__(); self.patch_module = patch_module
        def forward(self, x): return self.patch_module(x)

    wrapped = BlipWrapper(model.vision_model.embeddings.patch_embedding)
    target_layers = [model.vision_model.embeddings.patch_embedding]

    if selected_cam == "EigenCAM":
        cam = EigenCAM(model=wrapped, target_layers=target_layers)
        cam_map = cam(input_tensor=input_tensor)[0]
    elif selected_cam == "KPCA-CAM":
        cam = KPCA_CAM(model=wrapped, target_layers=target_layers)
        cam_map = cam(input_tensor=input_tensor)[0]
    elif selected_cam == "Attention Rollout":
        def rollout_fn(pixel_values):
            attn_maps = []
            def hook(module, input, output):
                if isinstance(output, tuple): attn_maps.append(output[1].detach().cpu())
            hooks = [layer.self_attn.register_forward_hook(hook) for layer in model.vision_model.encoder.layers]
            with torch.no_grad(): _ = model.vision_model(pixel_values, output_attentions=True)
            for h in hooks: h.remove()
            result = torch.eye(attn_maps[0].shape[-1])
            for attn in attn_maps:
                attn_avg = attn.mean(1) + torch.eye(attn.shape[-1])
                attn_avg /= attn_avg.sum(dim=-1, keepdim=True)
                result = torch.matmul(attn_avg[0], result)
            return result[0, 1:]

        rollout = rollout_fn(input_tensor)
        size = int(np.sqrt(rollout.shape[0]))
        rollout = rollout[:size**2].reshape(size, size).numpy()
        cam_map = cv2.resize((rollout - rollout.min()) / (rollout.max() - rollout.min() + 1e-8), (image.width, image.height))
    elif selected_cam == "Saliency Map":
        def forward_fn(x): return model.vision_model(x).last_hidden_state[:, 0, :].norm(dim=1)
        sal = Saliency(forward_fn)
        sal_attr = sal.attribute(input_tensor)[0].cpu().permute(1, 2, 0).numpy()
        cam_map = np.mean(np.abs(sal_attr), axis=-1)
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)

    cam_overlay = show_cam_on_image(rgb_image, cam_map, use_rgb=True)
    st.image(cam_overlay, caption=f"Visualisasi: {selected_cam}", use_container_width=True)

# --- Navigation Buttons ---
b_prev, _, b_next = st.columns([1, 6, 1])
if b_prev.button("◄ Sebelumnya") and st.session_state.current_step > 0:
    st.session_state.current_step -= 1
    st.rerun()
if b_next.button("Berikutnya ►") and st.session_state.current_step < len(steps) - 1:
    st.session_state.current_step += 1
    st.rerun()
