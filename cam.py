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