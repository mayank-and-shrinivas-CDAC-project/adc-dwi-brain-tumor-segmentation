"""Notebook-equivalent MONAI inference and lesion post-processing."""

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from scipy import ndimage as ndi


ROI_SIZE = (96, 96, 64)


def clean_brain_mask(brain_mask):
    """Create the notebook's conservative connected 3D brain mask."""
    brain = np.asarray(brain_mask, dtype=bool)
    if brain.ndim != 3:
        raise ValueError(f"Expected 3D brain mask, got shape={brain.shape}")
    brain = ndi.binary_fill_holes(brain)
    labels, count = ndi.label(brain, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if not count:
        return np.zeros_like(brain, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    brain = labels == int(np.argmax(sizes))
    brain = ndi.binary_closing(brain, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    return ndi.binary_fill_holes(brain).astype(bool)


def filter_small_clusters(binary_mask, min_size=20):
    """Keep the notebook's primary lesion even when it is a small cluster."""
    if not binary_mask.any():
        return binary_mask
    labels, count = ndi.label(binary_mask)
    if not count:
        return binary_mask
    sizes = np.bincount(labels.ravel())
    largest = sizes[1:].max() if len(sizes) > 1 else 0
    effective_min_size = min(min_size, largest) if largest else min_size
    result = binary_mask.copy()
    result[(sizes < effective_min_size)[labels]] = False
    return result


def refine_tumor_segmentation(segmentation, probabilities, brain_mask, min_cluster_size=20, prob_threshold=0.5):
    """Apply the notebook's final probability-driven edema/core refinement."""
    refined = np.zeros_like(segmentation, dtype=np.uint8)
    brain_mask = clean_brain_mask(brain_mask)
    if not np.any(brain_mask):
        return refined

    edema_probability = probabilities[1]
    core_probability = probabilities[2]
    max_core = float(core_probability.max())
    max_edema = float(edema_probability.max())
    core_threshold = min(prob_threshold, max(max_core * 0.6, 0.25)) if max_core else prob_threshold
    edema_threshold = min(prob_threshold, max(max_edema * 0.6, 0.25)) if max_edema else prob_threshold

    edema = (edema_probability > edema_threshold) & (edema_probability > core_probability) & brain_mask
    core = (core_probability > core_threshold) & brain_mask

    # Suppress implausibly diffuse core predictions, as in the notebook.
    if core.sum() > 0.40 * brain_mask.sum():
        brain_probabilities = core_probability[brain_mask]
        core = (core_probability >= np.percentile(brain_probabilities, 95)) & brain_mask

    refined[filter_small_clusters(edema, min_cluster_size)] = 1
    refined[filter_small_clusters(core, min_cluster_size)] = 2

    # Do not silently turn a positive model prediction into a negative one.
    if not np.any(refined) and np.any(segmentation > 0):
        return segmentation.astype(np.uint8)
    return refined


def predict(model, input_tensor, device, brain_mask=None):
    """Run notebook-style Gaussian sliding-window inference and refinement."""
    model.eval()
    if not torch.is_tensor(input_tensor):
        input_tensor = torch.from_numpy(input_tensor)
    input_tensor = input_tensor.float()
    if input_tensor.ndim == 4:
        input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = sliding_window_inference(
            inputs=input_tensor,
            roi_size=ROI_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=0.625,
            mode="gaussian",
        )
        probabilities = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()
        raw_prediction = probabilities.argmax(axis=0)

    if brain_mask is None:
        brain_mask = input_tensor.squeeze(0)[0].cpu().numpy() > 0
    return refine_tumor_segmentation(
        raw_prediction,
        probabilities,
        brain_mask,
    )
