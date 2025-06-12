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
from pytorch_grad_cam.metrics.road import ROADCombined
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import Saliency, LayerGradCam, LayerAttribution
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
st.set_page_config(page_title="X-Capindo - Visualisasi Proses Image Captioning", page_icon="imgs/logo.png", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #2980B9; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #dddddd; text-align: center; margin-bottom: 2rem; }
    .step-header { font-size: 1.8rem; font-weight: bold; color: #2980B9; margin-bottom: 1rem; }
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

# Enhanced CAM Optimization Techniques for Better Accuracy

# === 1. Advanced Target Layer Selection ===
def get_optimal_target_layers(model, method="auto"):
    """
    Pilih layer target yang optimal untuk setiap metode CAM
    """
    if method == "auto":
        # Otomatis pilih berdasarkan analisis gradien
        return [
            model.vision_model.encoder.layers[-1].self_attn,  # Last attention
            model.vision_model.encoder.layers[-2].self_attn,  # Second last
            model.vision_model.embeddings.patch_embedding     # Patch embedding
        ]
    elif method == "deep":
        # Layer dalam untuk semantic features
        return [model.vision_model.encoder.layers[-1].self_attn]
    elif method == "shallow":
        # Layer dangkal untuk low-level features
        return [model.vision_model.embeddings.patch_embedding]
    elif method == "multi":
        # Multiple layers untuk comprehensive view
        return [
            model.vision_model.encoder.layers[i].self_attn 
            for i in [-1, -4, -8, -12]
        ]

# === 2. Enhanced Target Functions ===
class AdaptiveTarget:
    """Target function yang adaptif berdasarkan metode"""
    def __init__(self, model, target_type="cls_norm"):
        self.model = model
        self.target_type = target_type
    
    def __call__(self, model_output):
        if self.target_type == "cls_norm":
            # Standard CLS token norm
            return model_output[:, 0, :].norm(dim=1)
        elif self.target_type == "spatial_avg":
            # Average spatial features
            return model_output[:, 1:, :].mean(dim=1).norm(dim=1)
        elif self.target_type == "attention_weighted":
            # Weighted by attention patterns
            with torch.no_grad():
                attn_weights = self.model.vision_model.encoder.layers[-1].self_attn.attention_weights
                if attn_weights is not None:
                    weights = attn_weights[:, :, 0, 1:].mean(dim=1)  # CLS to patches
                    weighted_features = (model_output[:, 1:, :] * weights.unsqueeze(-1)).sum(dim=1)
                    return weighted_features.norm(dim=1)
            return model_output[:, 0, :].norm(dim=1)

def enhanced_forward_fn(model, target_type="adaptive"):
    """Enhanced forward function dengan multiple target options"""
    def forward_with_caption_loss(x):
        # Forward pass dengan caption generation loss
        outputs = model.vision_model(x)
        features = outputs.last_hidden_state
        
        if target_type == "adaptive":
            # Adaptif berdasarkan task
            cls_features = features[:, 0, :]
            spatial_features = features[:, 1:, :].mean(dim=1)
            combined = 0.7 * cls_features + 0.3 * spatial_features
            return combined.norm(dim=1)
        elif target_type == "caption_guided":
            # Guided by actual caption generation
            with torch.no_grad():
                # Simulasi caption generation untuk guidance
                text_features = model.text_decoder.bert.embeddings.word_embeddings.weight.mean(dim=0)
                similarity = torch.cosine_similarity(features[:, 0, :], text_features.unsqueeze(0), dim=1)
                return similarity * features[:, 0, :].norm(dim=1)
        else:
            return features[:, 0, :].norm(dim=1)
    
    return forward_with_caption_loss

# === 3. Multi-Scale CAM Integration ===
def multi_scale_cam(model, input_tensor, cam_method, scales=[0.8, 1.0, 1.2]):
    """
    Generate CAM pada multiple scales untuk robustness
    """
    cam_maps = []
    original_size = input_tensor.shape[2:]
    
    for scale in scales:
        # Resize input
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        scaled_input = F.interpolate(input_tensor, size=new_size, mode='bilinear', align_corners=False)
        
        # Generate CAM
        if cam_method == "EigenCAM":
            wrapped = BlipPatchWrapper(model.vision_model.embeddings.patch_embedding)
            targets = [AdaptiveTarget(model, "cls_norm")]
            cam = EigenCAM(model=wrapped, target_layers=get_optimal_target_layers(model, "deep"))
            cam_map = cam(input_tensor=scaled_input, targets=targets)[0]
        
        # Resize back to original
        cam_map_resized = cv2.resize(cam_map, (original_size[1], original_size[0]))
        cam_maps.append(cam_map_resized)
    
    # Ensemble CAM maps
    final_cam = np.mean(cam_maps, axis=0)
    return (final_cam - final_cam.min()) / (final_cam.max() - final_cam.min() + 1e-8)

# === 4. Gradient Enhancement Techniques ===
def enhanced_saliency_map(model, input_tensor, enhancement="guided"):
    """
    Enhanced saliency dengan berbagai teknik
    """
    def guided_backprop_fn(x):
        # Guided backpropagation untuk cleaner gradients
        outputs = model.vision_model(x)
        return outputs.last_hidden_state[:, 0, :].norm(dim=1)
    
    def integrated_gradients_fn(x):
        # Integrated gradients untuk better attribution
        from captum.attr import IntegratedGradients
        ig = IntegratedGradients(guided_backprop_fn)
        return ig.attribute(x, baselines=x * 0, n_steps=50)
    
    def smooth_grad_fn(x):
        # SmoothGrad untuk noise reduction
        from captum.attr import NoiseTunnel
        saliency = Saliency(guided_backprop_fn)
        nt = NoiseTunnel(saliency)
        return nt.attribute(x, nt_type='smoothgrad', nt_samples=25, stdevs=0.15)
    
    input_tensor.requires_grad_()
    
    if enhancement == "guided":
        attr = guided_backprop_fn(input_tensor)
        attr.backward(torch.ones_like(attr))
        sal_map = input_tensor.grad[0].detach().cpu().permute(1, 2, 0).numpy()
    elif enhancement == "integrated":
        sal_attr = integrated_gradients_fn(input_tensor)
        sal_map = sal_attr[0].detach().cpu().permute(1, 2, 0).numpy()
    elif enhancement == "smooth":
        sal_attr = smooth_grad_fn(input_tensor)
        sal_map = sal_attr[0].detach().cpu().permute(1, 2, 0).numpy()
    else:
        saliency = Saliency(guided_backprop_fn)
        sal_attr = saliency.attribute(input_tensor)
        sal_map = sal_attr[0].detach().cpu().permute(1, 2, 0).numpy()
    
    sal_map = np.mean(np.abs(sal_map), axis=-1)
    return (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)

# === 5. Attention Rollout Improvements ===
def enhanced_attention_rollout(model, pixel_values, rollout_type="weighted"):
    """
    Enhanced attention rollout dengan berbagai improvement
    """
    attn_maps = []
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple) and output[1] is not None:
            attn_maps.append(output[1].detach().cpu())
    
    # Register hooks pada semua layer
    hooks = []
    for i, layer in enumerate(model.vision_model.encoder.layers):
        hook = layer.self_attn.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    with torch.no_grad():
        _ = model.vision_model(pixel_values, output_attentions=True)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    if rollout_type == "weighted":
        # Weighted rollout berdasarkan layer importance
        layer_weights = np.exp(np.linspace(0, 1, len(attn_maps)))  # Exponential weighting
        layer_weights = layer_weights / layer_weights.sum()
        
        result = torch.eye(attn_maps[0].shape[-1])
        for i, attn in enumerate(attn_maps):
            attn_heads_avg = attn.mean(1)
            attn_heads_avg = attn_heads_avg + torch.eye(attn_heads_avg.size(-1))
            attn_heads_avg = attn_heads_avg / attn_heads_avg.sum(dim=-1, keepdim=True)
            
            # Apply layer weight
            weighted_attn = layer_weights[i] * attn_heads_avg[0] + (1 - layer_weights[i]) * torch.eye(attn_heads_avg.size(-1))
            result = torch.matmul(weighted_attn, result)
            
    elif rollout_type == "selective":
        # Hanya gunakan head dengan attention yang high variance
        result = torch.eye(attn_maps[0].shape[-1])
        for attn in attn_maps:
            # Pilih head dengan variance tertinggi
            head_variances = []
            for head in range(attn.shape[1]):
                head_attn = attn[0, head]
                variance = torch.var(head_attn).item()
                head_variances.append(variance)
            
            best_head = np.argmax(head_variances)
            attn_best = attn[0, best_head:best_head+1]
            attn_avg = attn_best.mean(0)
            attn_avg = attn_avg + torch.eye(attn_avg.size(-1))
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            result = torch.matmul(attn_avg, result)
    else:
        # Standard rollout
        result = torch.eye(attn_maps[0].shape[-1])
        for attn in attn_maps:
            attn_heads_avg = attn.mean(1)
            attn_heads_avg = attn_heads_avg + torch.eye(attn_heads_avg.size(-1))
            attn_heads_avg = attn_heads_avg / attn_heads_avg.sum(dim=-1, keepdim=True)
            result = torch.matmul(attn_heads_avg[0], result)
    
    return result[0, 1:]  # Remove CLS token

# === 6. Ensemble CAM Methods ===
def ensemble_cam_methods(model, input_tensor, rgb_image, weights=None):
    """
    Ensemble multiple CAM methods untuk better accuracy
    """
    if weights is None:
        weights = {"EigenCAM": 0.3, "KPCA-CAM": 0.3, "Attention": 0.2, "Saliency": 0.2}
    
    cam_results = {}
    
    # EigenCAM
    wrapped = BlipPatchWrapper(model.vision_model.embeddings.patch_embedding)
    targets = [AdaptiveTarget(model, "adaptive")]
    cam = EigenCAM(model=wrapped, target_layers=get_optimal_target_layers(model, "multi"))
    cam_results["EigenCAM"] = cam(input_tensor=input_tensor, targets=targets)[0]
    
    # KPCA-CAM  
    cam = KPCA_CAM(model=wrapped, target_layers=get_optimal_target_layers(model, "deep"))
    cam_results["KPCA-CAM"] = cam(input_tensor=input_tensor, targets=targets)[0]
    
    # Enhanced Attention Rollout
    rollout = enhanced_attention_rollout(model, input_tensor, "weighted")
    size = int(np.sqrt(rollout.shape[0]))
    rollout_map = rollout[:size**2].reshape(size, size).numpy()
    cam_results["Attention"] = (rollout_map - rollout_map.min()) / (rollout_map.max() - rollout_map.min() + 1e-8)
    
    # Enhanced Saliency
    cam_results["Saliency"] = enhanced_saliency_map(model, input_tensor, "smooth")
    
    # Weighted ensemble
    final_cam = np.zeros_like(cam_results["EigenCAM"])
    for method, weight in weights.items():
        if method in cam_results:
            # Normalize each CAM to [0,1]
            cam_norm = cam_results[method]
            cam_norm = (cam_norm - cam_norm.min()) / (cam_norm.max() - cam_norm.min() + 1e-8)
            # Resize to common size
            cam_resized = cv2.resize(cam_norm, (final_cam.shape[1], final_cam.shape[0]))
            final_cam += weight * cam_resized
    
    return final_cam, cam_results

# === 7. Post-processing Enhancements ===
def post_process_cam(cam_map, method="gaussian_smooth"):
    """
    Post-processing untuk improve CAM quality
    """
    if method == "gaussian_smooth":
        # Gaussian smoothing untuk reduce noise
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(cam_map, sigma=1.0)
    
    elif method == "bilateral_filter":
        # Bilateral filter untuk preserve edges
        cam_uint8 = (cam_map * 255).astype(np.uint8)
        filtered = cv2.bilateralFilter(cam_uint8, 9, 75, 75)
        return filtered.astype(np.float32) / 255.0
    
    elif method == "morphological":
        # Morphological operations untuk clean up
        cam_uint8 = (cam_map * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(cam_uint8, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed.astype(np.float32) / 255.0
    
    elif method == "threshold_smooth":
        # Threshold + smooth untuk focus on important areas
        threshold = np.percentile(cam_map, 70)
        thresholded = np.where(cam_map > threshold, cam_map, cam_map * 0.3)
        return gaussian_filter(thresholded, sigma=0.8)
    
    return cam_map

# === 8. Adaptive ROAD Evaluation ===
def adaptive_road_evaluation(model, input_tensor, cam_map, strategy="progressive"):
    """
    Adaptive ROAD evaluation dengan multiple strategies
    """
    if strategy == "progressive":
        # Progressive masking dengan multiple ratios
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        scores = []
        
        for ratio in ratios:
            score = evaluate_cam_drop(model, input_tensor.clone(), cam_map, ratio)
            scores.append(score)
        
        # Weighted average dengan preference untuk middle ratios
        weights = [0.1, 0.2, 0.4, 0.2, 0.1]
        final_score = sum(s * w for s, w in zip(scores, weights))
        return final_score, scores
    
    elif strategy == "adaptive_threshold":
        # Adaptive threshold berdasarkan CAM distribution
        sorted_cam = np.sort(cam_map.flatten())
        percentiles = [75, 80, 85, 90, 95]
        scores = []
        
        for p in percentiles:
            threshold = np.percentile(sorted_cam, p)
            mask_ratio = np.sum(cam_map > threshold) / cam_map.size
            if mask_ratio > 0.05:  # Minimum 5% masking
                score = evaluate_cam_drop(model, input_tensor.clone(), cam_map, mask_ratio)
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0, scores
    
    else:
        # Standard evaluation
        return evaluate_cam_drop(model, input_tensor.clone(), cam_map, 0.3), [0.3]

# === 9. Integration Interface ===
def optimize_cam_prediction(model, input_tensor, rgb_image, method="EigenCAM", optimization_level="high"):
    """
    Main function untuk generate optimized CAM predictions
    """
    start_time = time.time()
    
    if optimization_level == "basic":
        # Basic optimization
        if method == "EigenCAM":
            wrapped = BlipPatchWrapper(model.vision_model.embeddings.patch_embedding)
            targets = [AdaptiveTarget(model, "cls_norm")]
            cam = EigenCAM(model=wrapped, target_layers=[model.vision_model.embeddings.patch_embedding])
            cam_map = cam(input_tensor=input_tensor, targets=targets)[0]
        
        post_processed = post_process_cam(cam_map, "gaussian_smooth")
        road_score, _ = adaptive_road_evaluation(model, input_tensor, post_processed, "progressive")
        
    elif optimization_level == "high":
        # High-level optimization
        if method == "Ensemble":
            cam_map, individual_results = ensemble_cam_methods(model, input_tensor, rgb_image)
        else:
            # Multi-scale single method
            cam_map = multi_scale_cam(model, input_tensor, method, scales=[0.9, 1.0, 1.1])
        
        post_processed = post_process_cam(cam_map, "bilateral_filter")
        road_score, score_details = adaptive_road_evaluation(model, input_tensor, post_processed, "adaptive_threshold")
        
    elif optimization_level == "research":
        # Research-level optimization
        ensemble_cam, individual_cams = ensemble_cam_methods(model, input_tensor, rgb_image)
        
        # Apply different post-processing to each
        processed_cams = {}
        for name, cam_data in individual_cams.items():
            processed_cams[name] = post_process_cam(cam_data, "threshold_smooth")
        
        # Re-ensemble processed CAMs
        weights = {"EigenCAM": 0.4, "KPCA-CAM": 0.3, "Attention": 0.2, "Saliency": 0.1}
        final_cam = np.zeros_like(ensemble_cam)
        for name, weight in weights.items():
            if name in processed_cams:
                final_cam += weight * processed_cams[name]
        
        road_score, score_details = adaptive_road_evaluation(model, input_tensor, final_cam, "progressive")
        cam_map = final_cam
    
    elapsed_time = time.time() - start_time
    
    return {
        "cam_map": cam_map,
        "road_score": road_score,
        "elapsed_time": elapsed_time,
        "optimization_level": optimization_level,
        "method": method
    }

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
with st.expander("Tampilkan Detail Teknis", expanded=False):
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
        st.session_state.caption_output = caption

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
    
    # Layout dengan 2 kolom utama
    col_visual, col_explanation = st.columns([3, 2])
    
    with col_visual:
        st.subheader("🔍 Pilih Teknik Visualisasi Fokus Gambar")
        
        # Row 1: Method selection and optimization level
        method_col, opt_col = st.columns([2, 1])
        with method_col:
            selected_cam = st.selectbox(
                "Metode Explainability", 
                ["KPCA-CAM", "EigenCAM", "Attention Rollout", "Saliency Map", "Ensemble Methods"]
            )
        with opt_col:
            optimization_level = st.selectbox(
                "🚀 Level Optimasi",
                ["basic", "enhanced"],
                index=1,
                help="basic: Standard implementation | enhanced: With post-processing & multi-target"
            )
        
        # Row 2: Advanced controls
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            mask_ratio = st.slider(
                "🎯 Masking Ratio untuk ROAD", 
                min_value=0.1, max_value=0.8, value=0.3, step=0.05,
                help="Persentase area terpenting yang akan di-mask untuk evaluasi ROAD"
            )
        with adv_col2:
            if optimization_level == "enhanced":
                post_process_method = st.selectbox(
                    "🔧 Post-Processing",
                    ["none", "gaussian_smooth", "bilateral_filter"],
                    index=1,
                    help="Metode untuk meningkatkan kualitas CAM"
                )
            else:
                post_process_method = "none"

        image = st.session_state.image or load_uploaded_image("imgs/test5.jpg")
        input_tensor = st.session_state.inputs['pixel_values']
        rgb_image = np.array(image).astype(np.float32) / 255.0

        # === Enhanced Classes ===
        class BlipPatchWrapper(torch.nn.Module):
            """Wrapper untuk patch embedding agar dapat di-hook oleh CAM"""
            def __init__(self, patch_module):
                super().__init__()
                self.patch_module = patch_module
            def forward(self, x):
                return self.patch_module(x)

        class EnhancedTarget:
            """Enhanced target untuk CAM yang lebih robust"""
            def __init__(self, target_type="norm"):
                self.target_type = target_type
            
            def __call__(self, model_output):
                if self.target_type == "norm":
                    return model_output.norm(dim=(1, 2))
                elif self.target_type == "mean_norm":
                    return model_output.mean(dim=(1, 2)).norm(dim=1)
                else:
                    return model_output.norm(dim=(1, 2))

        def enhanced_forward_fn(x):
            """Enhanced forward function dengan error handling"""
            try:
                outputs = model.vision_model(x)
                hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs["last_hidden_state"]
                return hidden_states[:, 0, :].norm(dim=1)  # CLS token norm
            except Exception as e:
                st.error(f"Error in forward function: {str(e)}")
                return torch.zeros(x.shape[0])

        def safe_cam_computation(method_name, input_tensor, wrapped_model, targets):
            """Safe CAM computation dengan error handling"""
            try:
                target_layers = [model.vision_model.embeddings.patch_embedding]
                
                if method_name == "EigenCAM":
                    cam = EigenCAM(model=wrapped_model, target_layers=target_layers)
                    attributions = cam(input_tensor=input_tensor, targets=targets)
                    return attributions[0], True
                    
                elif method_name == "KPCA-CAM":
                    cam = KPCA_CAM(model=wrapped_model, target_layers=target_layers)
                    attributions = cam(input_tensor=input_tensor, targets=targets)
                    return attributions[0], True
                    
                else:
                    return None, False
                    
            except Exception as e:
                st.warning(f"Error in {method_name}: {str(e)}")
                return None, False

        def safe_attention_rollout(model, pixel_values):
            """Safe attention rollout dengan error handling"""
            try:
                attn_maps = []
                def hook_fn(module, input, output):
                    if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                        attn_maps.append(output[1].detach().cpu())
                
                hooks = []
                for layer in model.vision_model.encoder.layers:
                    hook = layer.self_attn.register_forward_hook(hook_fn)
                    hooks.append(hook)
                
                with torch.no_grad():
                    _ = model.vision_model(pixel_values, output_attentions=True)
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
                
                if not attn_maps:
                    return None
                
                result = torch.eye(attn_maps[0].shape[-1])
                for attn in attn_maps:
                    attn_heads_avg = attn.mean(1)
                    attn_heads_avg = attn_heads_avg + torch.eye(attn_heads_avg.size(-1))
                    attn_heads_avg = attn_heads_avg / attn_heads_avg.sum(dim=-1, keepdim=True)
                    result = torch.matmul(attn_heads_avg[0], result)
                
                return result[0, 1:]  # Remove CLS token
                
            except Exception as e:
                st.warning(f"Error in attention rollout: {str(e)}")
                return None

        def safe_saliency_map(model, input_tensor):
            """Safe saliency map computation"""
            try:
                saliency = Saliency(enhanced_forward_fn)
                input_tensor.requires_grad_()
                sal_attr = saliency.attribute(input_tensor)
                sal_map = sal_attr[0].detach().cpu().permute(1, 2, 0).numpy()
                sal_map = np.mean(np.abs(sal_map), axis=-1)
                sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)
                return sal_map
            except Exception as e:
                st.warning(f"Error in saliency map: {str(e)}")
                return None

        def safe_post_process(cam_map, method="none"):
            """Safe post-processing dengan error handling"""
            if method == "none" or cam_map is None:
                return cam_map
                
            try:
                if method == "gaussian_smooth":
                    from scipy.ndimage import gaussian_filter
                    return gaussian_filter(cam_map, sigma=1.0)
                elif method == "bilateral_filter":
                    cam_uint8 = (cam_map * 255).astype(np.uint8)
                    filtered = cv2.bilateralFilter(cam_uint8, 9, 75, 75)
                    return filtered.astype(np.float32) / 255.0
                else:
                    return cam_map
            except Exception as e:
                st.warning(f"Post-processing error: {str(e)}")
                return cam_map

        def evaluate_cam_drop(model, input_tensor, cam_map, mask_ratio=0.3):
            """Safe ROAD evaluation dengan error handling"""
            try:
                if cam_map is None:
                    return 0.0
                    
                with torch.no_grad():
                    # Skor original
                    orig_score = enhanced_forward_fn(input_tensor).item()
                    
                    # Resize CAM map jika perlu
                    if cam_map.shape != input_tensor.shape[2:]:
                        cam_map_resized = cv2.resize(cam_map, (input_tensor.shape[3], input_tensor.shape[2]))
                    else:
                        cam_map_resized = cam_map
                    
                    # Normalisasi CAM map
                    cam_tensor = torch.tensor(cam_map_resized).float()
                    cam_tensor = (cam_tensor - cam_tensor.min()) / (cam_tensor.max() - cam_tensor.min() + 1e-6)
                    
                    # Pilih top-k piksel untuk di-mask
                    k = int(mask_ratio * cam_tensor.numel())
                    flat = cam_tensor.flatten()
                    idx = torch.topk(flat, k).indices
                    
                    # Buat mask
                    mask = torch.ones_like(flat)
                    mask[idx] = 0
                    mask = mask.reshape(cam_tensor.shape).unsqueeze(0).unsqueeze(0).to(input_tensor.device)
                    
                    # Hitung skor setelah masking
                    drop_score = enhanced_forward_fn(input_tensor * mask).item()
                    
                    # Return relative drop
                    road_score = (orig_score - drop_score) / (orig_score + 1e-6)
                    return round(road_score, 4)
                    
            except Exception as e:
                st.warning(f"ROAD evaluation error: {str(e)}")
                return 0.0

        # Progress indicator
        progress_container = st.empty()
        with progress_container:
            st.info(f"🔄 Menghitung CAM dengan optimasi level {optimization_level}...")

        # === Main CAM Computation ===
        start_time = time.time()
        cam_map = None
        road_score_lib = None
        road_score_custom = 0.0

        # Setup common components
        wrapped_model = BlipPatchWrapper(model.vision_model.embeddings.patch_embedding)
        targets = [EnhancedTarget("norm")]

        try:
            if selected_cam == "EigenCAM":
                cam_map, success = safe_cam_computation("EigenCAM", input_tensor, wrapped_model, targets)
                if success and optimization_level == "enhanced":
                    # Try library ROAD
                    try:
                        cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
                        cam_obj = EigenCAM(model=wrapped_model, target_layers=[model.vision_model.embeddings.patch_embedding])
                        attributions = cam_obj(input_tensor=input_tensor, targets=targets)
                        road_score_lib = float(np.array(cam_metric(input_tensor, attributions, targets, wrapped_model)).flatten()[0])
                    except:
                        road_score_lib = None

            elif selected_cam == "KPCA-CAM":
                cam_map, success = safe_cam_computation("KPCA-CAM", input_tensor, wrapped_model, targets)
                if success and optimization_level == "enhanced":
                    # Try library ROAD
                    try:
                        cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
                        cam_obj = KPCA_CAM(model=wrapped_model, target_layers=[model.vision_model.embeddings.patch_embedding])
                        attributions = cam_obj(input_tensor=input_tensor, targets=targets)
                        road_score_lib = float(np.array(cam_metric(input_tensor, attributions, targets, wrapped_model)).flatten()[0])
                    except:
                        road_score_lib = None

            elif selected_cam == "Attention Rollout":
                rollout = safe_attention_rollout(model, input_tensor)
                if rollout is not None:
                    size = int(np.sqrt(rollout.shape[0]))
                    if size * size == rollout.shape[0]:
                        rollout_map = rollout[:size**2].reshape(size, size).numpy()
                        cam_map = (rollout_map - rollout_map.min()) / (rollout_map.max() - rollout_map.min() + 1e-8)

            elif selected_cam == "Saliency Map":
                cam_map = safe_saliency_map(model, input_tensor)

            elif selected_cam == "Ensemble Methods":
                # Simple ensemble of available methods
                ensemble_maps = []
                
                # Try EigenCAM
                eigen_map, success = safe_cam_computation("EigenCAM", input_tensor, wrapped_model, targets)
                if success and eigen_map is not None:
                    ensemble_maps.append(eigen_map)
                
                # Try Saliency
                sal_map = safe_saliency_map(model, input_tensor)
                if sal_map is not None:
                    # Resize to match EigenCAM if needed
                    if len(ensemble_maps) > 0:
                        sal_map = cv2.resize(sal_map, (ensemble_maps[0].shape[1], ensemble_maps[0].shape[0]))
                    ensemble_maps.append(sal_map)
                
                # Average ensemble
                if ensemble_maps:
                    cam_map = np.mean(ensemble_maps, axis=0)
                    cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)

            # Fallback jika cam_map masih None
            if cam_map is None:
                st.warning("⚠️ Menggunakan fallback method...")
                cam_map = safe_saliency_map(model, input_tensor)
                if cam_map is None:
                    # Ultimate fallback - random map
                    cam_map = np.random.rand(224, 224)
                    st.error("❌ Semua metode gagal, menggunakan random map untuk demo")

            # Post-processing
            if optimization_level == "enhanced":
                cam_map = safe_post_process(cam_map, post_process_method)

            # ROAD evaluation
            road_score_custom = evaluate_cam_drop(model, input_tensor.clone(), cam_map, mask_ratio)

        except Exception as e:
            st.error(f"Critical error in CAM computation: {str(e)}")
            cam_map = np.random.rand(224, 224)  # Emergency fallback
            road_score_custom = 0.0

        elapsed_time = round(time.time() - start_time, 2)

        # === Visualization ===
        if cam_map is not None:
            try:
                cam_map_resized = cv2.resize(cam_map, (rgb_image.shape[1], rgb_image.shape[0]))
                cam_overlay = show_cam_on_image(rgb_image.copy(), cam_map_resized, use_rgb=True)
            except:
                # Fallback visualization
                cam_overlay = (rgb_image * 255).astype(np.uint8)
                st.warning("⚠️ Visualization error, showing original image")
        else:
            cam_overlay = (rgb_image * 255).astype(np.uint8)

        progress_container.empty()
        
        # Pilih skor primary
        primary_road_score = road_score_lib if road_score_lib is not None else road_score_custom
        score_source = "Library" if road_score_lib is not None else "Custom"
        
        st.markdown("### 🌐 Visualisasi Global dengan ROAD Evaluation")
        st.image(cam_overlay, caption=f"{selected_cam} | Waktu: {elapsed_time}s | Level: {optimization_level}", width=400)
        st.success(f"Caption: {st.session_state.caption_output}")
        
        # Tampilkan metrics
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("⏱️ Waktu Komputasi", f"{elapsed_time}s")
        with metric_cols[1]:
            st.metric("📊 ROAD Score", f"{primary_road_score:.4f}")
            # Tambahkan keterangan skala
            if score_source == "Custom":
                st.caption("📏 **Skala ROAD:**\n> 0.1: Baik | 0.0-0.1: Sedang | < 0.0: Buruk")
            else:
                st.caption("📏 **Skala Library ROAD:**\nBervariasi per metode, lebih tinggi = lebih baik")
        with metric_cols[2]:
            st.metric("📊 Sumber", score_source)

    # === Kolom Penjelasan ROAD (sama seperti sebelumnya) ===
    with col_explanation:
        st.markdown("### 📚 Evaluasi ROAD (Remove And Debias)")
        
        with st.expander("🔍 Konsep ROAD", expanded=True):
            st.markdown("""
            **ROAD** adalah metode evaluasi kuantitatif untuk mengukur kualitas interpretasi XAI:
            
            **🎯 Tujuan:**
            - Mengukur seberapa baik metode XAI mengidentifikasi area penting
            - Evaluasi tanpa perlu retraining model
            - Lebih stabil dibanding metode ROAR
            
            **⚙️ Cara Kerja:**
            1. **Remove**: Masking area yang dianggap penting oleh XAI
            2. **Debias**: Mengukur penurunan performa dengan debiased masking
            3. **Score**: Semakin besar penurunan = semakin baik interpretasi
            """)

        with st.expander("🛠️ Implementasi Teknis", expanded=False):
            st.markdown(f"""
            **Komponen Utama:**
            
            ```python
            # 1. BlipPatchWrapper
            # Membungkus patch embedding untuk CAM hooks
            
            # 2. EnhancedTarget  
            # Target scoring yang lebih robust
            
            # 3. enhanced_forward_fn
            # Forward function dengan error handling
            
            # 4. Safe ROAD Evaluation
            # Masking {int(mask_ratio*100)}% area paling penting
            # Ukur penurunan: (orig - masked) / orig
            ```
            
            **Optimasi Level {optimization_level}:**
            {"- Post-processing dengan " + post_process_method if optimization_level == "enhanced" else "- Standard implementation"}
            {"- Enhanced error handling" if optimization_level == "enhanced" else ""}
            {"- Multi-method ensemble support" if optimization_level == "enhanced" else ""}
            
            **Formula ROAD:**
            ```
            ROAD_Score = (Original_Score - Masked_Score) / Original_Score
            ```
            
            **Mengapa Masking {int(mask_ratio*100)}%?**
            - **Teoritis**: ROAD paper merekomendasikan 20-50%
            - **Praktis**: Balance antara sensitivity vs stability  
            - **Empiris**: Sweet spot untuk deteksi area penting
            """)

        with st.expander("📊 Hasil Evaluasi", expanded=True):
            # Interpretasi hasil berdasarkan skor
            if primary_road_score > 0.1:
                interpretation = "🟢 **Baik** - Area fokus sangat relevan"
            elif primary_road_score > 0.0:
                interpretation = "🟡 **Sedang** - Area fokus cukup relevan"  
            else:
                interpretation = "🔴 **Buruk** - Area fokus tidak relevan"
            
            st.markdown(f"""
            **Metode:** {selected_cam}
            
            **ROAD Score:** {primary_road_score:.4f}
            
            **Sumber Score:** {score_source}
            
            **Level Optimasi:** {optimization_level}
            
            **Interpretasi:** {interpretation}
            
            **Detail:**
            - Masking ratio: {int(mask_ratio*100)}% area terpenting
            - Waktu komputasi: {elapsed_time} detik
            - Target: CLS token confidence
            - Post-processing: {post_process_method if optimization_level == "enhanced" else "none"}
            
            **📏 Panduan Interpretasi ROAD Score:**
            ```
            Custom Implementation:
            > 0.1    : Interpretasi Excellent 🟢
            0.05-0.1 : Interpretasi Good 🟡  
            0.0-0.05 : Interpretasi Fair 🟠
            < 0.0    : Interpretasi Poor 🔴
            
            Library Implementation:
            Skala bervariasi per metode, 
            nilai lebih tinggi = lebih baik
            ```
            """)
            
            # Progress bar visual untuk skor
            score_normalized = max(0, min(1, (primary_road_score + 0.2) / 0.4))
            st.progress(score_normalized)

        # Perbandingan metode
        with st.expander("⚖️ Tips Optimasi CAM", expanded=False):
            st.markdown(f"""
            **Level Optimasi Saat Ini:** {optimization_level}
            
            **Basic Level:**
            - Standard CAM implementation
            - Faster computation
            - Suitable untuk quick analysis
            
            **Enhanced Level:**
            - Post-processing untuk cleaner maps
            - Error handling yang lebih robust
            - Support untuk ensemble methods
            - Library ROAD untuk metode yang support
            
            **Tips Pemilihan Metode:**
            - **KPCA-CAM**: Robust, good untuk complex images
            - **EigenCAM**: Fast, good untuk simple images  
            - **Attention Rollout**: Interpretable, shows model attention
            - **Saliency Map**: Basic gradient-based, always available
            - **Ensemble**: Kombinasi multiple methods untuk robustness
            
            **Post-Processing:**
            - **none**: No processing, fastest
            - **gaussian_smooth**: Reduces noise, smoother maps
            - **bilateral_filter**: Preserves edges, cleaner boundaries
            """)

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

