import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


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

# -------- Fixtures --------

@pytest.fixture(scope="module")
def sample_tokens():
    return ['[CLS]', 'seorang', 'pria', 'muda', 'yang', 'mengenakan', 'kacamata', 'duduk', 'di', 'bangku', '[SEP]']

@pytest.fixture(scope="module")
def blip_components():
    try:
        processor = BlipProcessor.from_pretrained("models/blip-image-captioning-base/v1.0")
        model = BlipForConditionalGeneration.from_pretrained("models/blip-image-captioning-base/v1.0")
    except:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

@pytest.fixture(scope="module")
def white_image():
    return Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)

# -------- Unit Tests --------
def test_reconstruct_words_and_indices(sample_tokens):
    words, indices = reconstruct_words_and_indices(sample_tokens)

    expected_words = ['seorang', 'pria', 'muda', 'yang', 'mengenakan', 'kacamata', 'duduk', 'di', 'bangku']
    expected_indices = [[1], [2], [3], [4], [5], [6], [7], [8], [9]]

    assert words == expected_words
    assert indices == expected_indices

def test_reconstruct_with_subtokens():
    tokens = ['[CLS]', 'meng', '##enakan', 'kacamata', '[SEP]']
    words, indices = reconstruct_words_and_indices(tokens)

    assert words == ['mengenakan', 'kacamata']
    assert indices == [[1, 2], [3]]

def test_model_caption_generation(blip_components, white_image):
    processor, model = blip_components
    inputs = processor(images=white_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    assert isinstance(caption, str)
    assert len(caption.strip()) > 0
