# Streamlit Wizard for BLIP Image Captioning – Updated with CAM & Cross-Attention
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from PIL import Image, ImageOps
from pytorch_grad_cam import EigenCAM, KPCA_CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import Saliency, LayerGradCam, LayerAttribution
import torchvision
from transformers import BlipProcessor, BlipForConditionalGeneration
import io, math, time

# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
st.set_page_config(page_title="X-Capindo - Visualisasi Proses Image Captioning", page_icon="imgs/logo.png", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #4B5563; text-align: center; margin-bottom: 2rem; }
    .step-header { font-size: 1.8rem; font-weight: bold; color: #1E3A8A; margin-bottom: 1rem; }
    .step-description { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .tech-details { background-color: #EFF6FF; color: black; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1.5rem; }
    .stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="main-header">Visualisasi Proses Image Captioning</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Pelajari tahapan bagaimana AI menghasilkan deskripsi dari gambar</div>', unsafe_allow_html=True)

# --- Sidebar Info ---
with st.sidebar.expander("ℹ️ Tentang X-Capindo", expanded=True):
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
    model = BlipForConditionalGeneration.from_pretrained(model_path).to(device)
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

def factorize(n):
    for i in range(int(math.sqrt(n)), 0, -1):
        if n % i == 0:
            return (i, n // i)
    return (1, n)

def reconstruct_words_and_indices(token_strs, ignore_tokens={'[PAD]', '[SEP]', '.', ',', ';', '!', '?', '[CLS]', '[MASK]', '[UNK]'}):
    words = []
    indices = []
    current_word = ""
    current_idxs = []

    for idx, token in enumerate(token_strs):
        if token is None or token in ignore_tokens:
            continue
        if token.startswith("##"):
            current_word += token[2:]
            current_idxs.append(idx)
        else:
            if current_word:
                words.append(current_word)
                indices.append(current_idxs)
            current_word = token
            current_idxs = [idx]
    if current_word:
        words.append(current_word)
        indices.append(current_idxs)
    return words, indices

def compare_attention_vs_gradcam(model, processor, pixel_values, input_ids, rgb_image):
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state

    token_ids = input_ids[0].tolist()
    token_strs = processor.tokenizer.convert_ids_to_tokens(token_ids)
    words, word_token_indices = reconstruct_words_and_indices(token_strs)

    target_layer = model.vision_model.embeddings.patch_embedding
    gradcam_images = []

    for word, idx_list in zip(words, word_token_indices):
        token_idx = idx_list[-1]

        def forward_fn(pixel_values):
            out = model(pixel_values=pixel_values, input_ids=input_ids)
            return out.logits[:, token_idx, :]

        cam = LayerGradCam(forward_fn, target_layer)
        attr = cam.attribute(pixel_values, target=token_ids[token_idx])
        cam_map = LayerAttribution.interpolate(attr, pixel_values.shape[-2:])
        cam_map = cam_map[0].mean(0).detach().cpu().numpy()
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)
        cam_map_resized = cv2.resize(cam_map, (rgb_image.shape[1], rgb_image.shape[0]))
        cam_overlay = show_cam_on_image(rgb_image.copy(), cam_map_resized, use_rgb=True)

        gradcam_images.append((word, cam_overlay))

    if not gradcam_images:
        return None

    max_cols = 5
    rows = int(np.ceil(len(gradcam_images) / max_cols))
    fig, axes = plt.subplots(rows, max_cols, figsize=(3 * max_cols, 3 * rows))

    # Pastikan axes selalu 2D array
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    if max_cols == 1:
        axes = np.expand_dims(axes, 1)

    for idx, (word, gradcam_img) in enumerate(gradcam_images):
        r, c = divmod(idx, max_cols)
        axes[r, c].imshow(gradcam_img)
        axes[r, c].set_title(f"{word}", fontsize=9)
        axes[r, c].axis("off")

    # Kosongkan sisa kolom di akhir baris terakhir
    for empty_idx in range(len(gradcam_images), rows * max_cols):
        r, c = divmod(empty_idx, max_cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    return fig

# --- Step logic ---
steps = [
    {"title": "Unggah Gambar", "description": "Sistem menerima gambar dari pengguna", "tech_details": "Gambar diubah menjadi tensor 3-channel dengan normalisasi dan penskalaan untuk digunakan model."},
    {"title": "Peta Aktivasi (CAM)", "description": "Visualisasi fokus model pada gambar", "tech_details": "Menggunakan teknik explainability: EigenCAM, KPCA-CAM, Attention Rollout, dan Saliency Map untuk menampilkan area penting yang diproses model."},
    {"title": "Feature Map", "description": "Ekstraksi dan visualisasi saluran fitur dari encoder", "tech_details": "Menampilkan 16 channel awal dari token visual BLIP (tanpa CLS), disusun dalam grid 4x4 sebagai representasi patch embedding."},
    {"title": "Cross-Attention per Kata", "description": "Visualisasi atensi model terhadap setiap kata yang dihasilkan", "tech_details": "Membandingkan peta atensi cross-attention decoder dengan hasil Grad-CAM untuk setiap token dalam caption."},
]

# --- Session state init ---
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "image" not in st.session_state:
    st.session_state.image = load_uploaded_image("imgs/test5.jpg")

for i in range(len(steps)):
    step_key = f"step_{i}_done"
    if step_key not in st.session_state:
        st.session_state[step_key] = False

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

    # Generate Caption
    image = st.session_state.image
    inputs = processor(images=image, return_tensors="pt", truncation=True).to(device)

    progress_bar = st.progress(0)
    progress_text = st.empty()
    start_total = time.time()
    progress_text.text("📄 Menghasilkan caption… (0%)")

    with torch.no_grad():
        gen_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
        output_ids = gen_output.sequences
        caption = processor.decode(output_ids[0], skip_special_tokens=True)

    st.session_state.generated_ids = output_ids
    st.session_state.inputs = inputs

    total_time = round(time.time() - start_total, 2)
    progress_bar.progress(100)
    progress_text.text(f"✅ Caption dihasilkan dalam {total_time} detik")

    st.success(f"📄 Caption yang Dihasilkan: \"{caption}\"")
    st.session_state.step_1_done = True

# --- Step 2: Visualisasi CAM ---
if st.session_state.current_step == 1:
    st.subheader("🔍 Pilih Teknik Visualisasi Fokus Gambar")
    selected_cam = st.selectbox(
        "Metode Explainability", 
        ["EigenCAM", "KPCA-CAM", "Attention Rollout", "Saliency Map"]
    )

    image = st.session_state.image or load_uploaded_image("imgs/test5.jpg")
    input_tensor = processor(images=image, return_tensors="pt", truncation=True).pixel_values.to(device)
    rgb_image = np.array(image).astype(np.float32) / 255.0

    # === Siapkan untuk CAM global
    class BlipWrapper(torch.nn.Module):
        def __init__(self, patch_module):
            super().__init__()
            self.patch_module = patch_module
        def forward(self, x):
            return self.patch_module(x)

    wrapped = BlipWrapper(model.vision_model.embeddings.patch_embedding)
    target_layers = [model.vision_model.embeddings.patch_embedding]

    # === Eksekusi CAM Global
    start_time = time.time()

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
                if isinstance(output, tuple):
                    attn_maps.append(output[1].detach().cpu())
            hooks = [layer.self_attn.register_forward_hook(hook) 
                     for layer in model.vision_model.encoder.layers]
            with torch.no_grad():
                _ = model.vision_model(pixel_values, output_attentions=True)
            for h in hooks:
                h.remove()
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

    elapsed = round(time.time() - start_time, 2)
    cam_map_resized = cv2.resize(cam_map, (rgb_image.shape[1], rgb_image.shape[0]))
    cam_overlay = show_cam_on_image(rgb_image.copy(), cam_map_resized, use_rgb=True)

    # === Tampilkan Hasil CAM Global
    st.markdown("### 🌐 Visualisasi Global")
    st.image(cam_overlay, caption=f"{selected_cam} (waktu: {elapsed} detik)", width=320)  # Lebih ramping

    # === Grad-CAM Eksklusif untuk per Kata
    st.markdown("### 🧠 Grad-CAM per Kata")
    with st.spinner("🔍 Menghitung Grad-CAM tiap kata..."):
        inputs = st.session_state.inputs
        pixel_values = inputs["pixel_values"]
        input_ids = st.session_state.generated_ids

        fig = compare_attention_vs_gradcam(
            model, processor, pixel_values, input_ids, rgb_image
        )

    if fig:
        st.pyplot(fig)
        st.session_state.step_2_done = True
    else:
        st.warning("⚠️ Gagal menampilkan Grad-CAM per kata.")

# --- Step 3: Feature Map ---
# if st.session_state.current_step == 2 and st.session_state.image:
if st.session_state.current_step == 2:
    st.subheader("🔬 Visualisasi Feature Map dari Encoder BLIP")

    image = st.session_state.image
    input_tensor = processor(images=image, return_tensors="pt", truncation=True).pixel_values.to(device)

    with torch.no_grad():
        outputs = model.vision_model(pixel_values=input_tensor)
        feature_map = outputs.last_hidden_state[:, 1:, :]  # exclude CLS

    num_patches = feature_map.shape[1]
    h, w = factorize(num_patches)  # ganti dari int(sqrt(...))
    if h * w != num_patches:
        st.warning("⚠️ Jumlah patch tidak bisa diubah menjadi grid 2D. Coba dengan gambar resolusi lain.")
    else:
        feature_map = feature_map[0].permute(1, 0).reshape(-1, h, w).detach().cpu()

        col1, col2 = st.columns([1, 4])
        with col1:
            selected = st.slider("Channel", 0, feature_map.shape[0] - 1, 0, 1)
        with col2:
            st.write(f"Menampilkan Feature Map - Channel #{selected}")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(feature_map[selected], cmap="viridis")
            ax.axis("off")
            st.pyplot(fig)

# --- Step 4: Cross-Attention ---
if st.session_state.current_step == 3:
# if st.session_state.current_step == 3 and st.session_state.image:
    st.subheader("🎯 Visualisasi Cross-Attention dan Grad-CAM")

    image = st.session_state.image
    pixel_values = processor(images=image, return_tensors="pt", truncation=True).pixel_values

    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state

        input_ids = processor(text="", return_tensors="pt").input_ids
        decoder_outputs = model.text_decoder.bert(
            input_ids=input_ids,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=None,
            output_attentions=True,
            return_dict=True,
        )
        cross_attentions = decoder_outputs.attentions

    token_ids = input_ids[0].tolist()
    token_strs = processor.tokenizer.convert_ids_to_tokens(token_ids)

    def reconstruct_words(token_strs):
        words, indices = [], []
        current, idxs = "", []
        for i, t in enumerate(token_strs):
            if t.startswith("##"):
                current += t[2:]; idxs.append(i)
            else:
                if current: words.append(current); indices.append(idxs)
                current, idxs = t, [i]
        if current: words.append(current); indices.append(idxs)
        return words, indices

    words, indices = reconstruct_words(token_strs)
    rgb_image = pixel_values[0].permute(1, 2, 0).numpy()
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

    fig, axes = plt.subplots(len(words), 2, figsize=(10, 4 * len(words)))
    if len(words) == 1:
        axes = np.expand_dims(axes, 0)

    for i, (word, idx_list) in enumerate(zip(words, indices)):
        token_idx = idx_list[-1]
        attn_heads = cross_attentions[-1][0, :, token_idx, :][:, 1:]
        best_head = torch.argmax(attn_heads.mean(1)).item()
        attn_map = attn_heads[best_head]
        h, w = int(attn_map.shape[0]**0.5), int(attn_map.shape[0]**0.5)
        attn_img = attn_map.reshape(h, w).detach().cpu().numpy()
        range_val = attn_img.max() - attn_img.min()
        attn_img = (attn_img - attn_img.min()) / (range_val + 1e-8)
        attn_overlay = show_cam_on_image(rgb_image, cv2.resize(attn_img, (rgb_image.shape[1], rgb_image.shape[0])), use_rgb=True)

        def forward_fn(pix):
            out = model(pixel_values=pix, input_ids=input_ids)
            return out.logits[:, token_idx, :]

        cam = LayerGradCam(forward_fn, model.vision_model.embeddings.patch_embedding)
        attr = cam.attribute(pixel_values, target=token_ids[token_idx])
        cam_map = LayerAttribution.interpolate(attr, pixel_values.shape[-2:])
        cam_map = cam_map[0].mean(0).detach().cpu().numpy()
        cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min())
        cam_overlay = show_cam_on_image(rgb_image, cam_map, use_rgb=True)

        axes[i, 0].imshow(attn_overlay)
        axes[i, 0].set_title(f"Cross-Attn '{word}'")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cam_overlay)
        axes[i, 1].set_title(f"Grad-CAM '{word}'")
        axes[i, 1].axis("off")

    st.pyplot(fig)

# --- Navigation Buttons ---
b_prev, _, b_next = st.columns([1, 6, 1])
if b_prev.button("◄ Sebelumnya") and st.session_state.current_step > 0:
    st.session_state.current_step -= 1
    st.rerun()
if b_next.button("Berikutnya ►") and st.session_state.current_step < len(steps) - 1:
    st.session_state.current_step += 1
    st.rerun()

