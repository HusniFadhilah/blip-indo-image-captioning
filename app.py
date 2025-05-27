# Streamlit Wizard for BLIP Image Captioning – Updated with CAM & Cross-Attention
import streamlit as st
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2, torch, torchvision, graphviz
import io, math, time
import torch.nn as nn
from PIL import Image, ImageOps
from collections import Counter
from huggingface_hub import snapshot_download
from pytorch_grad_cam import EigenCAM, KPCA_CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import Saliency, LayerGradCam, LayerAttribution
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
st.set_page_config(page_title="X-Capindo - Visualisasi Proses Image Captioning", page_icon="imgs/logo.png", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #dddddd; text-align: center; margin-bottom: 2rem; }
    .step-header { font-size: 1.8rem; font-weight: bold; color: #1E3A8A; margin-bottom: 1rem; }
    .step-description { font-size: 1.1rem; color: #dddddd; margin-bottom: 1.5rem; }
    .tech-details { background-color: #EFF6FF; color: black; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1.5rem; }
    .stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="main-header">Visualisasi Proses Image Captioning</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Pelajari tahapan bagaimana AI menghasilkan deskripsi dari gambar</div>', unsafe_allow_html=True)

# --- Session init ---
if "model_loaded_once" not in st.session_state:
    st.session_state.model_loaded_once = False

# --- Sidebar Info ---
with st.sidebar.expander("ℹ️ Tentang X-Capindo", expanded=True):
    def set_model_flag():
        st.session_state.model_loaded_once = True
    st.image("imgs/logo.png", width=90)
    model_option = st.selectbox(
        "🔧 Pilih Model BLIP",
        ["BLIP-Large (local)", "BLIP-Base (local)", "BLIP-Base (HF Hub)", "BLIP-Large (HF Hub)"],
        key="model_selection",on_change=set_model_flag
    )
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
**Model**: BLIP (Base/Large) – `Salesforce/blip-image-captioning`

**Teknik XAI**:
- EigenCAM / KPCA-CAM
- Attention Rollout
- Saliency Map
- Cross-Attention Decoder
    """)

# --- Model Load (once) ---
@st.cache_resource
def load_model(model_choice="BLIP-Large (local)"):
    if model_choice == "BLIP-Base (local)":
        model_path = snapshot_download(repo_id=r"HusniFd/blip-image-captioning-base")
        # model_path = r"models/blip-image-captioning-base/v1.0"
    elif model_choice == "BLIP-Large (local)":
        model_path = snapshot_download(repo_id=r"HusniFd/blip-image-captioning-large")
        # model_path = r"models/blip-image-captioning-large/v1.0"
    elif model_choice == "BLIP-Large (HF Hub)":
        model_path = snapshot_download(repo_id=r"HusniFd/blip-image-captioning-large")
    elif model_choice == "BLIP-Base (HF Hub)":
        model_path = snapshot_download(repo_id=r"HusniFd/blip-image-captioning-base")
    else:
        raise ValueError("Model tidak dikenali.")
    processor = BlipProcessor.from_pretrained(model_path)
    model = BlipForConditionalGeneration.from_pretrained(model_path).to(device)
    return processor, model

processor, model = load_model(st.session_state.get("model_selection", "BLIP-Large (local)"))

# ✅ Global-style alert jika model dimuat ulang
if st.session_state.model_loaded_once:
    st.markdown(f'<div class="global-alert">✅ Model berhasil dimuat: {st.session_state.get("model_selection")}</div>', unsafe_allow_html=True)
    st.session_state.model_loaded_once = False

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

def compare_attention_vs_gradcam(model, processor, pixel_values, input_ids, rgb_image, mode="overlay"):
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
        heatmap = np.uint8(255 * cam_map_resized)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        if mode == "overlay":
            overlay = np.clip(rgb_image * 255 + 0.5 * heatmap, 0, 255).astype(np.uint8)
        elif mode == "heatmap":
            overlay = heatmap
        elif mode == "original":
            overlay = (rgb_image * 255).astype(np.uint8)
        else:
            overlay = heatmap

        gradcam_images.append((word, overlay))

    if not gradcam_images:
        return None

    max_cols = 5
    rows = int(np.ceil(len(gradcam_images) / max_cols))
    fig, axes = plt.subplots(rows, max_cols, figsize=(3.5 * max_cols, 3.5 * rows))

    if rows == 1:
        axes = np.expand_dims(axes, 0)
    if max_cols == 1:
        axes = np.expand_dims(axes, 1)

    for idx, (word, img) in enumerate(gradcam_images):
        r, c = divmod(idx, max_cols)
        axes[r, c].imshow(img)
        axes[r, c].set_title(f"{word}", fontsize=17)
        axes[r, c].axis("off")

    for empty_idx in range(len(gradcam_images), rows * max_cols):
        r, c = divmod(empty_idx, max_cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    return fig

def apply_cam_overlay(image_np, cam_map, alpha=0.4, colormap=cv2.COLORMAP_JET):
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), colormap)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)

    # Resize CAM to match image size
    cam_heatmap = cv2.resize(cam_heatmap, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Combine image and heatmap
    overlay = (alpha * cam_heatmap + (1 - alpha) * np.uint8(255 * image_np)).astype(np.uint8)
    return overlay

def draw_blip_architecture():
    graph = graphviz.Digraph(format='png')
    graph.attr(rankdir='LR', size='10')

    # Root
    graph.node("BLIP", "BlipForConditionalGeneration", shape='box', style='filled', fillcolor='lightblue')

    # Vision Branch
    graph.node("Vision", "BlipVisionModel", shape='box', style='filled', fillcolor='lightblue')
    graph.edge("BLIP", "Vision")

    graph.node("PatchEmb", "patch_embedding\nConv2d(3→1024)", shape='box', fillcolor='lightyellow', style='filled')
    graph.node("Encoder", "BlipEncoder", shape='box', fillcolor='lightyellow', style='filled')
    graph.node("Layers", "24 x BlipEncoderLayer", shape='box', fillcolor='lightyellow', style='filled')
    graph.node("SelfAttn", "self_attn\nqkv(1024→3072)", shape='box', fillcolor='lightgray', style='filled')
    graph.node("MLP", "mlp\n1024→4096→1024", shape='box', fillcolor='lightgray', style='filled')

    graph.edge("Vision", "PatchEmb")
    graph.edge("PatchEmb", "Encoder")
    graph.edge("Encoder", "Layers")
    graph.edge("Layers", "SelfAttn")
    graph.edge("Layers", "MLP")

    # Text Branch
    graph.node("Text", "BlipTextLMHeadModel", shape='box', style='filled', fillcolor='lightblue')
    graph.edge("BLIP", "Text")

    graph.node("TextModel", "BlipTextModel", shape='box', fillcolor='lightyellow', style='filled')
    graph.node("TextEmb", "Text Embeddings\n(word + pos)", shape='box', fillcolor='lightgray', style='filled')
    graph.node("TextEnc", "BlipTextEncoder", shape='box', fillcolor='lightyellow', style='filled')
    graph.node("TxtLayer", "12 x BlipTextLayer", shape='box', fillcolor='lightyellow', style='filled')
    graph.node("SelfTxt", "self_attention\n768→768", shape='box', fillcolor='lightgray', style='filled')
    graph.node("CrossAttn", "cross_attention\n1024→768", shape='box', fillcolor='lightgray', style='filled')

    graph.edge("Text", "TextModel")
    graph.edge("TextModel", "TextEmb")
    graph.edge("TextModel", "TextEnc")
    graph.edge("TextEnc", "TxtLayer")
    graph.edge("TxtLayer", "SelfTxt")
    graph.edge("TxtLayer", "CrossAttn")

    return graph

def summarize_model_colored(model: nn.Module, max_depth=2):
    lines = []

    def recurse(module, prefix="", depth=0):
        if depth > max_depth:
            return

        # Hitung modul berulang pada level ini
        child_types = [child.__class__.__name__ for _, child in module.named_children()]
        counter = Counter(child_types)

        # Lacak indeks per jenis untuk menghitung jumlahnya
        handled = set()
        for name, child in module.named_children():
            class_name = child.__class__.__name__
            if class_name in handled:
                continue  # Sudah diringkas

            same_type_children = [(n, c) for n, c in module.named_children() if c.__class__.__name__ == class_name]

            # Pilih warna
            if depth == 0:
                color = "dodgerblue"
            elif depth == 1:
                color = "mediumseagreen"
            elif depth == 2:
                color = "darkorange"
            else:
                color = "gray"

            if len(same_type_children) > 3:  # threshold untuk diringkas
                count = len(same_type_children)
                line = f"{prefix}├─ <span style='color:{color}'><b>{count}x {class_name}</b></span>"
                lines.append(line)
                handled.add(class_name)
                # Rekursi hanya pada satu anak untuk representasi struktur
                recurse(same_type_children[0][1], prefix + "│&nbsp;&nbsp;&nbsp;", depth + 1)
            else:
                # Tampilkan masing-masing
                shape_info = ""
                if hasattr(child, 'weight') and hasattr(child.weight, 'shape'):
                    shape_info = f" <span style='color:gray'>({tuple(child.weight.shape)})</span>"
                line = f"{prefix}├─ <span style='color:{color}'><b>{name}: {class_name}</b></span>{shape_info}"
                lines.append(line)
                recurse(child, prefix + "│&nbsp;&nbsp;&nbsp;", depth + 1)
                handled.add(class_name)

    recurse(model)
    return "<div style='font-family:monospace; font-size:13px;'>" + "<br>".join(lines) + "</div>"

# --- Step logic ---
steps = [
    {
        "title": "Unggah Gambar",
        "description": "Sistem menerima gambar dari pengguna",
        "tech_details": """
Gambar diubah menjadi tensor 3-channel dengan normalisasi dan penskalaan untuk digunakan model.
Caption dihasilkan oleh BLIP menggunakan encoder-decoder vision-language transformer.
        """
    },
    {
        "title": "Peta Aktivasi (CAM)",
        "description": "Visualisasi fokus model pada gambar",
        "tech_details": """
**Visualisasi Global Fokus Model (CAM):**
- Gambar dikonversi ke tensor dan dinormalisasi.
- Layer `patch_embedding` dari vision encoder dibungkus sebagai target Grad-CAM.
- Berdasarkan metode terpilih (`EigenCAM`, `KPCA-CAM`, `Attention Rollout`, `Saliency Map`), dihitung peta aktivasi.
- Peta CAM dinormalisasi dan di-overlay ke gambar untuk menampilkan area penting.
- Waktu proses CAM juga ditampilkan.

**🎯 Visualisasi Cross-Attention Decoder:**
- Cross-attention diambil dari layer terakhir decoder (`cross_att = decoder_outputs.attentions[-1]`).
- Untuk setiap token output, dipilih attention head terbaik.
- Attention map dirubah ke format 2D dan ditumpangkan pada gambar input.
- Menunjukkan bagian gambar yang paling diperhatikan saat menghasilkan kata tertentu.

**🧠 Grad-CAM per Kata Output:**
- Token output diubah menjadi kata dengan indeks token yang relevan.
- Untuk setiap kata, dihitung peta aktivasi dari `patch_embedding` menggunakan `LayerGradCam`.
- CAM dinormalisasi dan di-overlay ke gambar asli (`rgb_image`) per kata.
- Semua hasil ditampilkan sebagai grid visual (per kata) untuk melihat area fokus visual terhadap token tertentu.
        """
    },
    {
        "title": "Feature Map",
        "description": "Ekstraksi dan visualisasi saluran fitur dari encoder",
        "tech_details": """
**Analisis Feature Map:**
- Mengambil feature map dari output vision encoder (tanpa token CLS).
- 16 channel pertama divisualisasikan dalam grid 4x4.
- Slider tersedia untuk menampilkan 1 channel secara individu.
- Menunjukkan representasi spasial patch dari gambar.

**Confidence per Kata (Token Probability):**
- Menghitung confidence untuk tiap kata dari caption menggunakan `softmax` pada skor token.
- Sub-token digabungkan menjadi kata dan ditampilkan sebagai bar chart.
- Memberi gambaran seberapa yakin model terhadap setiap prediksi kata.
        """
    }
]


# --- Session state init ---
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "image" not in st.session_state:
    st.session_state.image = load_uploaded_image("imgs/test.jpg")
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False

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
with st.expander("Tampilkan Detail Teknis", expanded=True):
    st.markdown(f'<div class="tech-details">{current_step["tech_details"]}</div>', unsafe_allow_html=True)

# --- Step 1: Image Upload ---
if st.session_state.current_step == 0:
    col_left, col_right = st.columns([1, 2])
    with col_left:
        if st.session_state.image is None:
            st.image("imgs/test5.jpg", caption="Contoh Gambar Input", use_container_width=True)
        else:
            st.image(st.session_state.image, caption="🖼️ Gambar yang Diunggah", use_container_width=True)
            st.session_state.image_uploaded = False
            # Tombol ada di sini, hanya jika ada gambar
            if st.button("📄 Hasilkan Caption", key="generate_caption_btn", type="primary"):
                st.session_state.generate_caption = True
                st.rerun()  # trigger rerender di mana state sudah siap
        st.caption("Format yang didukung: JPG, PNG, WEBP")

    with col_right:
        uploaded_file = st.file_uploader("Unggah gambar Anda", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file:
            st.session_state.image = load_uploaded_image(uploaded_file)
            st.session_state.image_uploaded = True  # 👉 tambahkan flag ini
            st.session_state.generate_caption = False
            st.success("✅ Gambar berhasil diunggah. Tekan tombol di bawah untuk menghasilkan caption.")
            st.rerun()  # rerun hanya untuk update tampilan gambar, belum proses caption
            
    with st.expander("📊 Tampilkan Arsitektur BLIP"):
        st.graphviz_chart(draw_blip_architecture())

    with st.expander("📄 Arsitektur Model BLIP"):
        html_summary = summarize_model_colored(model, max_depth=3)
        st.markdown(html_summary, unsafe_allow_html=True)

    # Proses hanya jika tombol ditekan
    if st.session_state.get("generate_caption", False):
        image = st.session_state.image
        inputs = processor(images=image, return_tensors="pt", truncation=True).to(device)

        progress_bar = st.progress(0)
        progress_text = st.empty()
        start_total = time.time()
        progress_text.text("📄 Menghasilkan caption… (0%)")

        with torch.no_grad():
            gen_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
            st.session_state.gen_output = gen_output
            output_ids = gen_output.sequences
            caption = processor.decode(output_ids[0], skip_special_tokens=True)

        st.session_state.generated_ids = output_ids
        st.session_state.inputs = inputs

        total_time = round(time.time() - start_total, 2)
        progress_bar.progress(100)
        progress_text.text(f"✅ Caption dihasilkan dalam {total_time} detik")

        st.success(f"📄 Caption yang Dihasilkan: \"{caption}\"")

        st.session_state.generate_caption = False  # Reset agar tidak auto trigger ulang
    st.session_state.step_1_done = True  # Step 1 selesai

# --- Step 2: Visualisasi CAM ---
if st.session_state.current_step == 1:
    if "inputs" not in st.session_state or "generated_ids" not in st.session_state:
        st.warning("⚠️ Silakan unggah gambar dan tekan tombol '📄 Hasilkan Caption' terlebih dahulu.")
        st.stop()
    st.subheader("🔍 Pilih Teknik Visualisasi Fokus Gambar")
    selected_cam = st.selectbox(
        "Metode Explainability", 
        ["KPCA-CAM", "Attention Rollout", "EigenCAM", "Saliency Map"]
    )

    image = st.session_state.image or load_uploaded_image("imgs/test5.jpg")
    input_tensor = st.session_state.inputs['pixel_values']
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
    # cam_overlay = apply_cam_overlay(rgb_image, cam_map_resized, alpha=0.9)
    cam_overlay = show_cam_on_image(rgb_image.copy(), cam_map_resized, use_rgb=True)

    # === Tampilkan Hasil CAM Global
    st.markdown("### 🌐 Visualisasi Global")
    st.image(cam_overlay, caption=f"{selected_cam} (waktu: {elapsed} detik)", width=320)  # Lebih ramping

    st.markdown("### 🎯 Visualisasi Cross-Attention dari Decoder")

    with torch.no_grad():
        pixel_values = st.session_state.inputs["pixel_values"]
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state

        output_ids = st.session_state.generated_ids
        input_ids = output_ids.unsqueeze(0) if output_ids.dim() == 1 else output_ids

        decoder_outputs = model.text_decoder.bert(
            input_ids=input_ids,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=None,
            output_attentions=True,
            return_dict=True,
        )
        cross_att = decoder_outputs.attentions  # List[len_layer] of (B, heads, seq_len, num_patches+1)

        tokens = processor.tokenizer.convert_ids_to_tokens(input_ids[0])
        words, indices = reconstruct_words_and_indices(tokens)

        rgb_image = pixel_values[0].permute(1, 2, 0).detach().cpu().numpy()
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)

        # 🔻 Resize jika terlalu besar
        max_dim = 128
        h, w = rgb_image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            rgb_image = cv2.resize(rgb_image, (int(w * scale), int(h * scale)))

        cross_images = []

        for word, idx_list in zip(words, indices):
            token_idx = idx_list[-1]
            attn_heads = cross_att[-1][0, :, token_idx, :][:, 1:]  # remove CLS
            best_head = torch.argmax(attn_heads.mean(1)).item()
            attn_map = attn_heads[best_head]
            attn_len = attn_map.shape[-1]
            side = int(attn_len ** 0.5)

            if side * side != attn_len:
                new_len = (side + 1) ** 2
                pad_len = new_len - attn_len
                attn_map = F.pad(attn_map, (0, pad_len), value=0)
                side = int(attn_map.shape[-1] ** 0.5)

            attn_img = attn_map.reshape(side, side).detach().cpu().numpy()
            attn_img = (attn_img - attn_img.min()) / (attn_img.max() - attn_img.min() + 1e-8)

            attn_overlay = show_cam_on_image(
                rgb_image.copy(),
                cv2.resize(attn_img, (rgb_image.shape[1], rgb_image.shape[0])),
                use_rgb=True
            )

            cross_images.append((word, attn_overlay))

    # ===== Plot sebagai grid horizontal seperti Grad-CAM =====
    if cross_images:
        max_cols = 5
        rows = int(np.ceil(len(cross_images) / max_cols))
        fig, axes = plt.subplots(rows, max_cols, figsize=(3 * max_cols, 3 * rows))

        if rows == 1:
            axes = np.expand_dims(axes, 0)
        if max_cols == 1:
            axes = np.expand_dims(axes, 1)

        for idx, (word, overlay_img) in enumerate(cross_images):
            r, c = divmod(idx, max_cols)
            axes[r, c].imshow(overlay_img)
            axes[r, c].set_title(f"{word}", fontsize=17)
            axes[r, c].axis("off")

        # Kosongkan sisa slot
        for idx in range(len(cross_images), rows * max_cols):
            r, c = divmod(idx, max_cols)
            axes[r, c].axis("off")

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("ℹ️ Tidak ada token yang dapat divisualisasikan karena ukuran attention map tidak bisa direstrukturisasi menjadi persegi.")

    # === Grad-CAM Eksklusif untuk per Kata
    st.markdown("### 🧠 Grad-CAM per Kata")
    with st.spinner("🔍 Menghitung Grad-CAM tiap kata..."):
        inputs = st.session_state.inputs
        pixel_values = inputs["pixel_values"]
        input_ids = st.session_state.generated_ids

        visual_mode = st.radio("Tampilan Grad-CAM", ["overlay", "heatmap", "original"], horizontal=True)

        # Di bagian Grad-CAM per kata:
        fig = compare_attention_vs_gradcam(
            model, processor, pixel_values, input_ids, rgb_image, mode=visual_mode
        )

    if fig:
        st.pyplot(fig)
        st.session_state.step_2_done = True
    else:
        st.warning("⚠️ Gagal menampilkan Grad-CAM per kata.")

# --- Step 3: Feature Map ---
if st.session_state.current_step == 2 and st.session_state.image:
# if st.session_state.current_step == 2:
    if "inputs" not in st.session_state or "generated_ids" not in st.session_state:
        st.warning("⚠️ Silakan unggah gambar dan tekan tombol '📄 Hasilkan Caption' terlebih dahulu.")
        st.stop()
    st.subheader("🔍 Analisis Feature Map")

    # Ambil tensor gambar
    image = st.session_state.image
    input_tensor = st.session_state.inputs['pixel_values']

    # Dapatkan feature map dari encoder
    with torch.no_grad():
        outputs = model.vision_model(pixel_values=input_tensor)
        feature_map = outputs.last_hidden_state[:, 1:, :]  # exclude CLS

        num_patches = feature_map.shape[1]
        h, w = factorize(num_patches)

        if h * w != num_patches:
            st.warning("⚠️ Jumlah patch tidak bisa diubah menjadi grid 2D. Coba dengan gambar resolusi lain.")
        else:
            feature_map = feature_map[0].permute(1, 0).reshape(-1, h, w).detach().cpu()

            # Buat dua kolom dalam satu baris
            col_left, col_right = st.columns([2, 5])  # kiri lebih kecil

            # Kolom Kiri: Slider + 1 Feature Map
            with col_left:
                st.markdown("### 🧩 Feature Map Tunggal")
                selected = st.slider("Channel", 0, feature_map.shape[0] - 1, 0, 1)
                st.caption(f"Channel #{selected}")
                fig_feat, ax_feat = plt.subplots(figsize=(4, 4))
                ax_feat.imshow(feature_map[selected], cmap="viridis")
                ax_feat.axis("off")
                st.pyplot(fig_feat)

            # Kolom Kanan: Grid Feature Map 16 channel pertama
            with col_right:
                st.markdown("### 🧩 Grid 16 Feature Maps Pertama")

                fig_grid, axs = plt.subplots(4, 4, figsize=(8, 8))
                for i in range(16):
                    row, col = divmod(i, 4)
                    axs[row, col].imshow(feature_map[i], cmap="viridis")
                    axs[row, col].set_title(f"#{i}", fontsize=8)
                    axs[row, col].axis("off")

                plt.tight_layout()
                st.pyplot(fig_grid)

    st.markdown("### 📈 Probabilitas Token (Confidence per Kata)")
    # Ambil data caption & skor
    output_ids = st.session_state.generated_ids[0]
    gen_output = st.session_state.gen_output
    scores = gen_output.scores
    tokens = processor.tokenizer.convert_ids_to_tokens(output_ids)
    probs = [F.softmax(score[0], dim=-1) for score in scores]
    selected_probs = [p[output_ids[i+1]].item() for i, p in enumerate(probs)]  # skip BOS

    # Gabungkan sub-token → kata
    words, indices = reconstruct_words_and_indices(tokens[1:])  # skip BOS
    word_probs = []
    word_entropies = []
    word_margins = []

    for idx_list in indices:
        if all(i < len(selected_probs) for i in idx_list):
            # Confidence
            avg_prob = sum([selected_probs[i] for i in idx_list]) / len(idx_list)
            word_probs.append(avg_prob)

            # Entropy & margin
            entropy_list = []
            margin_list = []
            for i in idx_list:
                soft = probs[i]
                entropy = -torch.sum(soft * torch.log(soft + 1e-8)).item()
                topk = torch.topk(soft, 2).values
                margin = abs(topk[0] - topk[1]).item()
                entropy_list.append(entropy)
                margin_list.append(margin)
            word_entropies.append(sum(entropy_list) / len(entropy_list))
            word_margins.append(sum(margin_list) / len(margin_list))

    # DataFrame gabungan
    df_token_stats = pd.DataFrame({
        "Word": words,
        "Confidence": word_probs,
        "Entropy": word_entropies,
        "Top-2 Margin": word_margins
    })

    # === Fitur 1: Tampilkan peringatan jika ada kata yang low confidence
    low_conf = df_token_stats[df_token_stats["Confidence"] < 0.5]["Word"].tolist()
    if low_conf:
        st.warning("⚠️ Kata dengan confidence rendah: " + ", ".join(low_conf))
    else:
        st.success("✅ Semua kata memiliki confidence tinggi.")

    # === Fitur 2: Caption berwarna berdasarkan confidence
    def color_word(w, p):
        green = int(p * 255)
        red = 255 - green
        return f"<span style='color: rgb({red}, {green}, 0)'>{w}</span>"

    colored_caption = " ".join([color_word(w, p) for w, p in zip(words, word_probs)])
    st.markdown("### 🖼️ Caption Berwarna Berdasarkan Confidence")
    st.markdown(f"<p style='font-size:18px'>{colored_caption}</p>", unsafe_allow_html=True)

    # === Fitur 3: Threshold slider untuk filter
    thresh = st.slider("Tampilkan hanya kata dengan confidence di bawah:", 0.0, 1.0, 0.5)
    st.dataframe(df_token_stats[df_token_stats["Confidence"] < thresh])

    # === Fitur 4 & 5: Kolom Preview + Grafik
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown("### 📷 Gambar dan Caption")
        st.image(st.session_state.image, caption="Gambar yang Diupload", use_container_width=True)
        st.success("Caption: " + " ".join(words))

    with col_right:
        st.markdown("### 📈 Confidence per Kata")
        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.bar(words, word_probs, color=["green" if p >= thresh else "red" for p in word_probs])
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1.05)
        ax.set_title("Confidence per Kata Output")

        for bar, prob in zip(bars, word_probs):
            height = bar.get_height()
            ax.annotate(f"{prob:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=45)
        st.pyplot(fig)

# --- Step 4: Cross-Attention ---
# if st.session_state.current_step == 3:
# if st.session_state.current_step == 3 and st.session_state.image:
#     st.subheader("🎯 Visualisasi Cross-Attention dan Grad-CAM")

#     image = st.session_state.image
#     inputs = st.session_state.inputs
#     pixel_values = inputs["pixel_values"]
#     input_ids = st.session_state.generated_ids

#     with torch.no_grad():
#         vision_outputs = model.vision_model(pixel_values=pixel_values)
#         image_embeds = vision_outputs.last_hidden_state
#         decoder_outputs = model.text_decoder.bert(
#             input_ids=input_ids,
#             encoder_hidden_states=image_embeds,
#             encoder_attention_mask=None,
#             output_attentions=True,
#             return_dict=True,
#         )
#         cross_attentions = decoder_outputs.attentions

#     token_ids = input_ids[0].tolist()
#     token_strs = processor.tokenizer.convert_ids_to_tokens(token_ids)

#     words, indices = reconstruct_words_and_indices(token_strs)
#     rgb_image = pixel_values[0].permute(1, 2, 0).numpy()
#     rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

#     fig, axes = plt.subplots(len(words), 2, figsize=(10, 4 * len(words)))
#     if len(words) == 1:
#         axes = np.expand_dims(axes, 0)

#     for i, (word, idx_list) in enumerate(zip(words, indices)):
#         token_idx = idx_list[-1]

#         attn_heads = cross_attentions[-1][0, :, token_idx, :][:, 1:]  # hapus CLS
#         best_head = torch.argmax(attn_heads.mean(1)).item()
#         attn_map = attn_heads[best_head]
#         attn_len = attn_map.shape[0]
#         side = int(attn_len ** 0.5)

#         valid_attn = side * side == attn_len
#         if valid_attn:
#             attn_img = attn_map.reshape(side, side).detach().cpu().numpy()
#             attn_img = (attn_img - attn_img.min()) / (attn_img.max() - attn_img.min() + 1e-8)
#             attn_overlay = show_cam_on_image(
#                 rgb_image,
#                 cv2.resize(attn_img, (rgb_image.shape[1], rgb_image.shape[0])),
#                 use_rgb=True
#             )
#         else:
#             attn_overlay = np.ones_like(rgb_image)  # gambar putih
#             attn_overlay[:] = [1, 1, 1]

#         # Grad-CAM
#         def forward_fn(pix):
#             out = model(pixel_values=pix, input_ids=input_ids)
#             return out.logits[:, token_idx, :]

#         cam = LayerGradCam(forward_fn, model.vision_model.embeddings.patch_embedding)
#         attr = cam.attribute(pixel_values, target=token_ids[token_idx])
#         cam_map = LayerAttribution.interpolate(attr, pixel_values.shape[-2:])
#         cam_map = cam_map[0].mean(0).detach().cpu().numpy()
#         cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min())
#         cam_overlay = show_cam_on_image(rgb_image, cam_map, use_rgb=True)

#         axes[i, 0].imshow(attn_overlay)
#         axes[i, 0].set_title(f"Cross-Attn '{word}'" + ("" if valid_attn else " ❌"))
#         axes[i, 0].axis("off")

#         axes[i, 1].imshow(cam_overlay)
#         axes[i, 1].set_title(f"Grad-CAM '{word}'")
#         axes[i, 1].axis("off")

#     plt.tight_layout()
#     st.pyplot(fig)

# --- Navigation Buttons ---
b_prev, _, b_next = st.columns([1, 6, 1])
if b_prev.button("◄ Sebelumnya") and st.session_state.current_step > 0:
    st.session_state.current_step -= 1
    st.rerun()
if b_next.button("Berikutnya ►") and st.session_state.current_step < len(steps) - 1:
    st.session_state.current_step += 1
    st.rerun()

