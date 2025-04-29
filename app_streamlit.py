import os
os.environ["STREAMLIT_WATCHED_MODULES"] = ""

import torch, cv2, time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageOps
import io
import torchvision.transforms as T
import torch.nn.functional as F
from captum.attr import Saliency
from pytorch_grad_cam import EigenCAM, KPCA_CAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ====== Konfigurasi ======
device = "cuda" if torch.cuda.is_available() else "cpu"
st.set_page_config(page_title="X-Capindo", page_icon="🧠", layout="wide")

# ====== Header & Logo ======
col1, col2 = st.columns([1, 8])
with col1:
    st.image("imgs/logo.png", width=90)
with col2:
    st.title("X-Capindo – Captioning dan Eksplorasi Fokus Gambar Berbahasa Indonesia")
    st.markdown("""
    Aplikasi ini menggunakan model **BLIP** untuk menghasilkan deskripsi otomatis dari gambar, dilengkapi dengan **penjelasan visual (explainable AI)** seperti **CAM**, **Attention Rollout**, dan **Saliency Map**.

    Setiap peta atensi disertai *Skor Fokus*, yang menunjukkan seberapa terfokus perhatian model terhadap objek penting.
    """)

# ====== Load Model Lokal ======
@st.cache_resource
def load_blip_model():
    model_path = r"models/v1.0/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_path)
    model = BlipForConditionalGeneration.from_pretrained(model_path).to(device)
    return processor, model

processor, model = load_blip_model()

# ====== Utilitas Gambar ======
@st.cache_data
def load_uploaded_image(img):
    if isinstance(img, str):
        image = Image.open(img)
    else:
        image = Image.open(io.BytesIO(img.read())).convert("RGB")
    return ImageOps.exif_transpose(image)

def transform_to_tensor(image):
    return processor(images=image, return_tensors="pt").pixel_values.to(device)

# ====== Caption dan Attention ======
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    attention_maps = []

    def hook(module, input, output):
        attention_maps.append(output)

    handle = model.vision_model.encoder.layers[-1].self_attn.register_forward_hook(hook)

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs)
    inference_time = time.time() - start

    handle.remove()
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    attention = attention_maps[0][0].cpu().detach().numpy().mean(axis=1) if attention_maps else None
    return caption, attention, inference_time

# ====== Skor Fokus ======
def compute_focus_score(attn_map):
    flat = attn_map.flatten()
    flat = flat / (flat.sum() + 1e-8)
    entropy = -np.sum(flat * np.log(flat + 1e-8))
    normalized_entropy = entropy / np.log(len(flat))
    focus_score = 1 - normalized_entropy
    return round(focus_score, 3)

# ====== CAM Visualisasi ======
def compute_visual_explanations(image_pil, input_tensor, caption):
    rgb_image = input_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)

    class BlipWrapper(torch.nn.Module):
        def __init__(self, patch_module): super().__init__(); self.patch_module = patch_module
        def forward(self, x): return self.patch_module(x)

    wrapped = BlipWrapper(model.vision_model.embeddings.patch_embedding)
    target_layers = [model.vision_model.embeddings.patch_embedding]

    visuals = {}
    scores = {}
    times = {}

    for name, CAM in {"EigenCAM": EigenCAM, "KPCA-CAM": KPCA_CAM}.items():
        try:
            start = time.time()
            cam = CAM(model=wrapped, target_layers=target_layers)
            cam_map = cam(input_tensor=input_tensor)[0]
            cam_img = show_cam_on_image(rgb_image, cam_map, use_rgb=True)
            visuals[name] = cam_img
            scores[name] = compute_focus_score(cam_map)
            times[name] = round(time.time() - start, 2)
        except Exception as e:
            st.warning(f"{name} gagal: {e}")

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

    try:
        start = time.time()
        rollout = rollout_fn(input_tensor)
        size = int(np.sqrt(rollout.shape[0]))
        rollout = rollout[:size**2].reshape(size, size).numpy()
        rollout = (rollout - rollout.min()) / (rollout.max() - rollout.min() + 1e-8)
        rollout_resized = cv2.resize(rollout, (rgb_image.shape[1], rgb_image.shape[0]))
        visuals["Attention Rollout"] = show_cam_on_image(rgb_image, rollout_resized, use_rgb=True)
        scores["Attention Rollout"] = compute_focus_score(rollout_resized)
        times["Attention Rollout"] = round(time.time() - start, 2)
    except Exception as e:
        st.warning(f"Rollout gagal: {e}")

    def forward_fn(x): return model.vision_model(x).last_hidden_state[:, 0, :].norm(dim=1)
    try:
        start = time.time()
        sal = Saliency(forward_fn)
        sal_attr = sal.attribute(input_tensor)[0].cpu().permute(1, 2, 0).numpy()
        sal_map = np.mean(np.abs(sal_attr), axis=-1)
        sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)
        visuals["Saliency Map"] = show_cam_on_image(rgb_image, sal_map, use_rgb=True)
        scores["Saliency Map"] = compute_focus_score(sal_map)
        times["Saliency Map"] = round(time.time() - start, 2)
    except Exception as e:
        st.warning(f"Saliency Map gagal: {e}")

    return visuals, scores, times

# ====== Main App ======
uploaded = st.file_uploader("Unggah gambar (JPG/PNG/JPEG)", type=["jpg", "jpeg", "png", "webp"])
img_path = "imgs/test2.jpeg" if uploaded is None else uploaded

image = load_uploaded_image(img_path)
st.image(image, caption="🖼️ Gambar yang Diunggah", width=400)  # Ukuran diperkecil

if st.button("🎯 Hasilkan Caption & Penjelasan"):
    with st.spinner("Menghasilkan caption dan menjelaskan fokus model..."):
        start_total = time.time()
        caption, attention, inf_time = generate_caption(image)
        input_tensor = transform_to_tensor(image)
        visuals, scores, vis_times = compute_visual_explanations(image, input_tensor, caption)
        total_time = round(time.time() - start_total, 2)

    st.markdown(f"### 📾 Caption Hasil Generasi:")
    st.success(f"\"{caption}\"")

    st.markdown("### 🔍 Penjelasan Visual (CAM)")
    st.markdown("Setiap metode menampilkan area penting dari gambar yang dipertimbangkan oleh model. Skor Fokus menunjukkan seberapa terpusat perhatian model terhadap bagian gambar yang spesifik.")

    cols = st.columns(4)
    for idx, (title, vis) in enumerate(visuals.items()):
        with cols[idx % 4]:
            st.image(vis, caption=f"{title}\nSkor Fokus: {scores.get(title, '-')} | Waktu: {vis_times.get(title, '-')}s", use_container_width=True)

    st.markdown("### 📊 Ringkasan Skor dan Waktu")
    df_summary = {
        "Metode": list(scores.keys()),
        "Skor Fokus": list(scores.values()),
        "Waktu (detik)": [vis_times[k] for k in scores.keys()]
    }
    st.dataframe(df_summary)

    st.info(f"**Waktu Inference Caption:** {round(inf_time, 2)} detik\n\n**Total Waktu Seluruh Proses:** {total_time} detik")

# ====== Sidebar Info ======
st.sidebar.markdown("## ℹ️ Tentang X-Capindo")
st.sidebar.info("""
**X-Capindo** adalah aplikasi captioning berbasis BLIP (Salesforce) yang dilengkapi dengan penjelasan visual melalui CAM (Class Activation Maps) dan attention transformer.

**Langkah-langkah**:
1. Unggah gambar yang ingin dideskripsikan.
2. Klik *Hasilkan Caption*.
3. Lihat caption dan peta atensi model.

**Skor Fokus** adalah nilai antara 0 dan 1 yang mengukur apakah perhatian model tersebar atau terfokus. Nilai tinggi berarti fokus tinggi.

**Waktu inference** dan **waktu visualisasi per metode** juga ditampilkan untuk memahami performa.

🔗 Model: `blip-image-captioning-large`
""")