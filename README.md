# X-Capindo: Visual Explanation for Image Captioning with BLIP

## Overview

**X-Capindo** is an interactive Streamlit-based application that enables users to explore and understand how the BLIP (Bootstrapping Language-Image Pretraining) model generates captions from images. The tool enhances interpretability by incorporating several explainable AI (XAI) techniques, such as CAM (Class Activation Maps), cross-attention visualizations, and feature map analysis.

With a user-friendly step-by-step interface, X-Capindo empowers researchers, educators, and developers to:

* Upload and process images
* Generate descriptive captions using BLIP models
* Visualize attention and focus areas of the model during caption generation
* Analyze token confidence and model uncertainty

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

Ensure the following:

* A CUDA-capable GPU is available (recommended)
* BLIP models are accessible via Hugging Face (`HusniFd/blip-image-captioning-base` or `...-large`)

## Application Workflow

1. **Upload an image** via the wizard UI
2. **Generate a caption** using a selected BLIP model
3. **Visualize Class Activation Maps (CAM)** to understand focus regions
4. **Inspect cross-attention** for word-region correspondence
5. **Analyze encoder feature maps** and token-level confidence

## UI Steps

### Step 1: Image Upload

* Upload or use a default image
* Generate caption using BLIP Base or Large (local/remote)
* Visualize model architecture interactively (Graphviz + tree-style summary)

### Step 2: Visual CAM

* Select one of the CAM/XAI methods:

  * `EigenCAM`, `KPCA-CAM`
  * `Attention Rollout`
  * `Saliency Map`
* View global heatmap overlays on the image
* View per-token heatmaps using Grad-CAM
* Decode cross-attention layers for each output token

### Step 3: Feature Map

* Grid view of the first 16 encoder channels
* Inspect individual channels using a slider
* Compute token-level metrics:

  * Confidence
  * Entropy
  * Top-2 prediction margin
* View color-coded captions and bar charts of confidence

## Core Functions

### `load_model()`

Loads the BLIP model and processor from Hugging Face or local files.

```python
def load_model(model_choice="BLIP-Large (local)") -> Tuple[BlipProcessor, BlipForConditionalGeneration]
```

### `load_uploaded_image(img)`

Loads and prepares uploaded image for processing.

```python
def load_uploaded_image(img: Union[str, UploadedFile]) -> Image.Image
```

### `reconstruct_words_and_indices(token_strs)`

Reconstructs tokens into words and tracks their indices.

```python
def reconstruct_words_and_indices(token_strs: List[str]) -> Tuple[List[str], List[List[int]]]
```

### `compare_attention_vs_gradcam(...)`

Generates Grad-CAM overlays per generated word.

```python
def compare_attention_vs_gradcam(model, processor, pixel_values, input_ids, rgb_image, mode="overlay") -> Figure
```

### `apply_cam_overlay(...)`

Overlays CAM heatmap onto the input image.

```python
def apply_cam_overlay(image_np, cam_map, alpha=0.4) -> np.ndarray
```

### `draw_blip_architecture()`

Builds a Graphviz diagram of BLIP’s vision and language components.

```python
def draw_blip_architecture() -> graphviz.Digraph
```

### `summarize_model_colored()`

Generates collapsible HTML structure of the model with color-coded modules.

```python
def summarize_model_colored(model: nn.Module, max_depth=2) -> str
```

## Advanced Techniques

### EigenCAM, KPCA-CAM

PCA-based CAM visualizations showing which patches dominate the representation.

### Cross-Attention Visualization

Maps decoder attention heads over image patches to explain per-token focus.

### Saliency Map

Gradient-based method visualizing sensitivity of output to each image pixel.

## Model Architecture (BLIP)

* Vision encoder: ViT-style transformer (24 layers)
* Text decoder: 12-layer transformer
* Decoder uses cross-attention to focus on encoder outputs
* CAM targets `patch_embedding` layer for interpretability

To visualize architecture:

```python
st.graphviz_chart(draw_blip_architecture())
```

## Sample Output

Include a screenshot or visual output like this:

```markdown
![Sample Output](imgs/sample_output.jpg)
```

## License

Licensed under the MIT License. See `LICENSE` for details.

## Author

Developed by \[Husni Fadhilah] as part of the X-Capindo Explainable Image Captioning project.

---

> For academic use or contributions, please cite or fork from the original GitHub repository.
