import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
from PIL import Image, ImageOps
import io
import torchvision.transforms as T
import torch.nn.functional as F

# **Cek Device**
device = "cuda" if torch.cuda.is_available() else "cpu"

# **Konfigurasi Halaman Streamlit**
st.set_page_config(
    initial_sidebar_state="expanded",
    page_title="Explainable Image Caption Bot"
)

# **Load Model BLIP**
@st.cache_resource
def load_blip_model():
    # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    # model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)
    return processor, model

processor, model = load_blip_model()

# **Transformasi Gambar untuk Model**
def transform_image(img):
    transform = T.Compose([
        T.Resize((384, 384)),  # Resize sesuai model BLIP
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    return transform(img)

def generate_caption(image, processor, model):
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Pastikan kita menangkap perhatian dari Transformer
    attention_maps = []

    def get_attention_hook(module, input, output):
        print("✅ Hook executed! Attention captured.")  # Debugging
        attention_maps.append(output)  # Output adalah tuple

    # Pasang hook ke layer yang sesuai
    handle = model.vision_model.encoder.layers[-1].self_attn.register_forward_hook(get_attention_hook)

    # Generate caption
    with torch.no_grad():
        caption_ids = model.generate(**inputs)

    # Hapus hook setelah digunakan
    handle.remove()

    caption = processor.decode(caption_ids[0], skip_special_tokens=True)

    # **Periksa apakah attention_maps berhasil ditangkap**
    if not attention_maps:
        print("❌ Attention Maps tidak terisi! Hook mungkin tidak bekerja.")
        return caption, None

    # **Ambil tensor dari tuple**
    attention_tensor = attention_maps[0][0]  # Ambil tensor pertama dari tuple
    attention = attention_tensor.cpu().detach().numpy().mean(axis=1)

    return caption, attention


# **Fungsi untuk Memuat Gambar**
@st.cache_data
def load_uploaded_image(img):
    if isinstance(img, str):
        image = Image.open(img)
    else:
        img_bytes = img.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    image = ImageOps.exif_transpose(image)  # Perbaiki orientasi gambar
    return image

def plot_attention(image, caption, attention):
    """
    Menampilkan heatmap attention untuk setiap kata dalam caption.
    """

    if attention is None or len(attention.shape) != 2:
        st.error("Attention map tidak valid! Tidak bisa menampilkan heatmap.")
        return

    num_words = len(caption.split())
    num_attention_steps = min(num_words, attention.shape[0])  # Sesuaikan panjang attention

    fig, axes = plt.subplots(1, num_attention_steps, figsize=(num_attention_steps * 3, 5))

    if num_attention_steps == 1:
        axes = [axes]  # Pastikan list jika hanya ada satu kata

    for i in range(num_attention_steps):
        attn_map = attention[i]

        # **Reshape attention ke bentuk yang sesuai**
        if attn_map.shape[0] == 768:
            grid_size = 24  # Vision Transformer biasanya menggunakan 24x32 patches
            attn_map = attn_map[:grid_size * grid_size].reshape(grid_size, grid_size)
        else:
            st.warning(f"Attention map tidak bisa diubah menjadi grid! (Token count: {attn_map.shape[0]})")
            continue

        # **Interpolasi agar ukuran sesuai dengan gambar**
        attn_resized = F.interpolate(
            torch.tensor(attn_map).unsqueeze(0).unsqueeze(0), 
            size=(image.size[1], image.size[0]),  # Sesuaikan ke ukuran gambar
            mode="bilinear",
            align_corners=False
        ).squeeze().numpy()

        # **Plot setiap heatmap per kata**
        axes[i].imshow(image)
        axes[i].imshow(attn_resized, cmap='jet', alpha=0.5)
        axes[i].set_title(caption.split()[i])
        axes[i].axis("off")

    plt.tight_layout()
    st.pyplot(fig)

# **Streamlit UI**
st.title("Explainable Image Captioning Bot 🤖🖼️")
st.text("Powered by BLIP (Salesforce) - A Transformer-based Image Captioning Model")

st.success("Upload an image and generate a caption!")

# **File Upload**
uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["png", "jpg", "jpeg", "webp"])
img_path = "imgs/test2.jpeg" if uploaded_file is None else uploaded_file

# **Muat dan Tampilkan Gambar**
image = load_uploaded_image(img_path)
st.image(image, use_column_width=True, caption="Uploaded Image")

# **Generate Caption Button**
# Jika tombol ditekan, jalankan captioning dan attention visualization
if st.button("Generate Caption"):
    caption, attention = generate_caption(image, processor, model)

    if attention is None:
        st.error("Attention map tidak tersedia! Coba ganti layer yang di-hook.")
    else:
        st.markdown(f"### **Generated Caption:**\n📢 *{caption}*")
        plot_attention(image, caption, attention)  # ✅ Panggil dengan 3 argumen

    st.balloons()


# **Sidebar Info**
st.sidebar.markdown("""
### About This App 📝
This app generates captions for images using **Hugging Face's BLIP model** trained by **Salesforce**.  
It also provides **explainable AI insights** into how images are understood by deep learning models.

### How to Use:
1. **Upload an image** 📷 (JPG/PNG/JPEG).
2. **Click "Generate Caption"** 🏷️.
3. **View AI-generated caption** for your image along with **attention heatmap**!

### Want More Features?
Check the model on [Hugging Face](https://huggingface.co/Salesforce/blip-image-captioning-base).
""")
