#!/usr/bin/env python3
"""
Nightshade Antidote
===================

A digital image forensics tool that can analyze PNG or JPG images
for signs of manipulation or forgery, including:
- Metadata analysis
- Copy-move forgery detection
- Frequency domain checks (spectral analysis)
- Basic format checks
- Poison detection call (placeholder for advanced detection logic)

Usage:
  python nightshade_antidote.py input.png

The script will generate a console report and show relevant plots.

Requirements:
  - OpenCV
  - numpy
  - matplotlib
  - scipy
  - Pillow
  - scikit-learn
  - exiftool (plus the `exiftool` Python wrapper)
"""

import os
import sys
import cv2
import glob
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from PIL import Image
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import exiftool
import torch

# ---- Settings / Parameters ----
PATCH_SIZE = 8
CM_THRESHOLD = 0.1
DCT_SIZE = 512  # The size to which images are resized for DCT-based checks

# --------------------------------------------------------------------------------
# Example: Simple "model" saving/loading (for the poison detection placeholder)
# --------------------------------------------------------------------------------

def save_dummy_model(file_path="nightshade_model.pkl"):
    """
    Example function to save a "model" for reuse. In reality, you'd
    store your threshold, top singular vectors, or other learned parameters.
    """
    dummy_data = {
        "threshold": 0.9,
        "notes": "Example placeholder model for Nightshade detection"
    }
    with open(file_path, "wb") as f:
        pickle.dump(dummy_data, f)
    print(f"[INFO] Model saved to {file_path}")

def load_dummy_model(file_path="nightshade_model.pkl"):
    """
    Load the "model" data for the poison detection. This is just a stub example.
    """
    if not os.path.exists(file_path):
        print(f"[WARN] {file_path} does not exist. Returning None.")
        return None
    with open(file_path, "rb") as f:
        model_data = pickle.load(f)
    print(f"[INFO] Loaded model from {file_path}: {model_data}")
    return model_data


# --------------------------------------------------------------------------------
# Utility / Basic Forensic Routines
# --------------------------------------------------------------------------------

def detect_copy_move(img, patch_size=PATCH_SIZE, threshold=CM_THRESHOLD):
    """
    Detect copy-move forgery using KNN on DCT patches.
    If patches are too similar, they might be duplicates (copy-paste).
    """
    flags = np.zeros_like(img)
    # Convert to float32 if needed
    if img.dtype != np.float32:
        img = img.astype(np.float32)

    dct_img = cv2.dct(img)
    patches = []
    indices = []
    # Slide over the image in patch_size steps
    for i in range(0, img.shape[0], patch_size):
        for j in range(0, img.shape[1], patch_size):
            patch = dct_img[i:i+patch_size, j:j+patch_size]
            patches.append(patch.flatten())
            indices.append((i, j))

    patches = np.array(patches)
    if len(patches) < 2:
        return flags

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(patches)
    distances, neighbors = nbrs.kneighbors(patches)

    for i in range(len(patches)):
        # If the distance to the second nearest neighbor is below threshold
        if distances[i][1] < threshold:
            x1, y1 = indices[i]
            x2, y2 = indices[neighbors[i][1]]
            # Raise flags
            flags[x1:x1+patch_size, y1:y1+patch_size] = 1
            flags[x2:x2+patch_size, y2:y2+patch_size] = 1

    return flags

def analyze_metadata(img_path):
    """
    Analyze metadata using ExifTool. Returns a string summary.
    """
    report_str = ""
    try:
        with exiftool.ExifTool() as et:
            metadata = et.get_metadata(img_path)
            for tag in metadata:
                report_str += f"{tag}: {metadata[tag]}\n"
    except Exception as e:
        report_str += f"[WARN] ExifTool error: {str(e)}\n"

    return report_str

def spectral_analysis(img):
    """
    Perform FFT on a grayscale image, returning (magnitude_spectrum, phase_spectrum).
    We'll do a quick shift + log transform for the magnitude.
    """
    f_img = fft.fft2(img.astype(np.float64))
    f_img_shifted = fft.fftshift(f_img)

    magnitude_spectrum = np.log(np.abs(f_img_shifted) + 1e-8)
    phase_spectrum = np.angle(f_img_shifted)

    return magnitude_spectrum, phase_spectrum

def pixel_ordering_check(img):
    """
    Compare DCT of 'img' to a reference image 'reference.jpg' and
    compute correlation coefficient. Used for pixel ordering or tampering checks.
    """
    if not os.path.exists("reference.jpg"):
        print("No reference.jpg found. Skipping pixel ordering check.")
        return 0.0

    ref_img = cv2.imread("reference.jpg", cv2.IMREAD_GRAYSCALE)
    ref_img = cv2.resize(ref_img, (img.shape[1], img.shape[0]))

    img_dct = cv2.dct(img.astype(np.float32))
    ref_dct = cv2.dct(ref_img.astype(np.float32))

    img_dct_normed = (img_dct - img_dct.mean()) / (img_dct.std() + 1e-8)
    ref_dct_normed = (ref_dct - ref_dct.mean()) / (ref_dct.std() + 1e-8)

    corr_coeff = np.corrcoef(img_dct_normed.flatten(), ref_dct_normed.flatten())[0, 1]
    return corr_coeff

def compression_artifacts_check(img_path, img):
    """
    Check for JPEG quantization artifacts.
    If input is PNG, ratio won't reflect typical JPEG compression.
    """
    bit_depth = img.dtype.itemsize * 8
    file_size = os.path.getsize(img_path)
    uncompressed_size = img.shape[0] * img.shape[1] * (bit_depth / 8.0)
    if uncompressed_size == 0:
        ratio = 0
    else:
        ratio = file_size / uncompressed_size

    print(f"Bit depth: {bit_depth}")
    print(f"Approx. Compression ratio: {ratio:.2f}")

    dct_img = cv2.dct(img.astype(np.float32))
    quantization_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68,109,103, 77],
        [24, 35, 55, 64, 81,104,113, 92],
        [49, 64, 78, 87,103,121,120,101],
        [72, 92, 95, 98,112,100,103, 99]
    ], dtype=np.float32)

    H, W = dct_img.shape
    flags = np.zeros((H, W), dtype=np.uint8)
    step = 8
    for i in range(0, H, step):
        for j in range(0, W, step):
            block = dct_img[i:i+step, j:j+step]
            if block.shape != (8, 8):
                continue
            remainder = np.abs(block) % quantization_table
            threshold = quantization_table * 0.1
            flags_block = (remainder < threshold).astype(np.uint8)
            flags[i:i+step, j:j+step] = flags_block

    return flags

def file_format_check(img_path):
    """
    Check file format by reading the first few bytes.
    Distinguish PNG vs JPG if possible.
    """
    with open(img_path, "rb") as f:
        file_bytes = f.read()
    if len(file_bytes) < 8:
        print("File too small or corrupted. Cannot identify format.")
        return

    # PNG check
    if file_bytes[:4] == b"\x89PNG":
        print("File format appears to be PNG.")
    elif file_bytes[:2] == b"\xff\xd8":
        print("File format appears to be JPEG.")
    else:
        print("Unknown or non-PNG/JPEG file format signature.")

# --------------------------------------------------------------------------------
# Poison detection logic (placeholder)
# --------------------------------------------------------------------------------

MODEL_DATA = None

def init_poison_detector(model_path="nightshade_model.pkl"):
    """
    Example to load or create a "model" for poisoning detection.
    """
    global MODEL_DATA
    if os.path.exists(model_path):
        MODEL_DATA = load_dummy_model(model_path)
    else:
        print("[INFO] No existing poison model found. Creating dummy and saving.")
        save_dummy_model(model_path)
        MODEL_DATA = load_dummy_model(model_path)

def detect_poisoning(img_tensor):
    """
    A simple SVD-based 'poison detection' demonstration.
    Returns True if the image is 'suspiciously' different from
    typical distribution, based on singular values.

    Steps:
      1. Flatten the image data (C,H,W) -> (N,)
      2. Center it (subtract mean)
      3. Compute SVD on the reshaped data or on a small patch
      4. Use a ratio of top singular value to sum of singular values as a heuristic
         (If it's too high, we suspect it might be 'poisoned').

    In a real pipeline, you'd:
      - Extract stable diffusion VAE or CNN features
      - Combine them into a large matrix of known-clean samples
      - Fit an SVD or other model and store top vectors, thresholds, etc.
      - Evaluate a new sample's outlier score.

    For now, we do everything ad-hoc each time for demonstration.
    """

    # If the image is grayscale: shape = (H,W). If it's 3-channel: shape = (C,H,W).
    # Let's unify to a 1D vector for SVD. We'll just do a 1D SVD on the entire flattened image.
    # Typically, you’d do something more advanced with blocks or feature extraction.
    arr = img_tensor.cpu().numpy()

    if arr.ndim == 3:
        # (C,H,W) -> flatten
        vec = arr.reshape(-1).astype(np.float64)
    elif arr.ndim == 2:
        # (H,W) -> flatten
        vec = arr.ravel().astype(np.float64)
    else:
        print("[WARN] Unexpected shape in detect_poisoning:", arr.shape)
        return False  # fallback

    # Center (subtract mean)
    vec_mean = np.mean(vec)
    centered = vec - vec_mean

    # We do an SVD on (some dimension,1) or a small patch. 
    # For demonstration, treat centered as row matrix: shape (1,N).
    # Then s[0] is the single singular value if shape is (1,N).
    # Actually let's chunk it for demonstration so it's at least (M,N).
    chunk_size = 512  # arbitrary chunk for a pseudo-matrix shape
    if len(centered) < chunk_size:
        # not enough data; just do a fallback
        s = np.linalg.svd(centered.reshape(1, -1), full_matrices=False, compute_uv=False)
    else:
        # reshape e.g. (M,N) with M*N = len(centered)
        # pick M = chunk_size
        M = chunk_size
        N = len(centered) // M
        truncated = centered[: M*N]  # take first M*N
        matrix = truncated.reshape(M, N)
        s = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)  # shape: min(M,N) singular values

    # Heuristic: ratio of top singular value to sum of them
    top_singular = s[0]
    total_singular = np.sum(s)
    ratio = top_singular / (total_singular + 1e-8)

    # Decide a threshold
    # For demonstration, let's say ratio > 0.20 => suspicious 
    # (You would tune / learn this threshold from known data.)
    threshold = 0.20
    is_poisoned = (ratio > threshold)

    # Optional console prints for debugging
    print(f"[PoisonCheck] top_singular={top_singular:.3f}, sum={total_singular:.3f}, ratio={ratio:.3f}, threshold={threshold}")

    return is_poisoned

# --------------------------------------------------------------------------------
# Final report assembly
# --------------------------------------------------------------------------------

def output_report(img_path, img):
    """
    Generate a forensic report summarizing checks:
    - metadata
    - copy-move
    - spectral analysis
    - pixel ordering
    - compression artifacts
    - file format check
    - poison detection
    """
    print("\n===== NIGHTSHADE ANTIDOTE REPORT =====\n")

    print("[Step 1] File format check...")
    file_format_check(img_path)

    print("\n[Step 2] Metadata Analysis...")
    metadata_str = analyze_metadata(img_path)
    print(metadata_str)

    print("[Step 3] Copy-Move Forgery Detection...")
    flags = detect_copy_move(img)
    num_flags = np.sum(flags)
    if num_flags > 0:
        print(f"Detected {num_flags} regions likely copied and pasted.")
        plt.imshow(flags, cmap='gray')
        plt.title("Copy-Move Regions")
        plt.show()
    else:
        print("No copy-move forgery detected.")

    print("\n[Step 4] Spectral Analysis...")
    mag_spec, phase_spec = spectral_analysis(img)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.imshow(mag_spec, cmap='gray')
    ax1.set_title("Magnitude Spectrum")
    ax2.imshow(phase_spec, cmap='gray')
    ax2.set_title("Phase Spectrum")
    plt.show()

    # Quick anomaly detection for magnitude
    mag_mean = np.mean(mag_spec)
    mag_std = np.std(mag_spec)
    mag_threshold = mag_mean + 3 * mag_std
    mag_anomalies = np.where(mag_spec > mag_threshold)
    if len(mag_anomalies[0]) > 0:
        print(f"Detected {len(mag_anomalies[0])} anomalies in the magnitude spectrum.")
    else:
        print("No anomalies in magnitude spectrum.")

    print("\n[Step 5] Pixel Ordering Check (Reference-based)...")
    corr_coeff = pixel_ordering_check(img)
    if corr_coeff != 0.0:
        print(f"Correlation to reference: {corr_coeff:.4f}")
    else:
        print("No reference image or correlation is 0.0")

    print("\n[Step 6] Compression Artifacts Check...")
    artifact_flags = compression_artifacts_check(img_path, img)
    artifact_count = np.sum(artifact_flags)
    if artifact_count > 0:
        print(f"Detected {artifact_count} potential compression artifact blocks.")
    else:
        print("No compression artifacts flagged.")

    print("\n[Step 7] Poison Detection Check...")
    # Convert grayscale to 3-channel for detect_poisoning logic if needed
    if len(img.shape) == 2:
        cimg = np.stack([img]*3, axis=0)
    else:
        cimg = np.transpose(img, (2, 0, 1))
    cimg_t = torch.from_numpy(cimg).float()

    is_poisoned = detect_poisoning(cimg_t)
    if is_poisoned:
        print("[ALERT] This image appears POISONED by Nightshade (placeholder logic)!")
    else:
        print("[OK] No poisoning suspected by placeholder logic.")

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python nightshade_antidote.py <input_image>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"File {input_path} does not exist.")
        sys.exit(1)

    # Initialize or load the poison detection model (dummy)
    init_poison_detector("nightshade_model.pkl")

    # Read the image (PNG or JPG) using OpenCV
    print(f"[INFO] Reading {input_path}")
    img_cv = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_cv is None:
        print(f"[ERROR] Could not read image {input_path} with OpenCV.")
        sys.exit(1)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Resize for uniform DCT processing
    img_gray = cv2.resize(img_gray, (DCT_SIZE, DCT_SIZE))

    # Output a forensic report (with plots)
    print("[INFO] Generating forensic report...")
    output_report(input_path, img_gray)

    # Example: We have a simple "model", so we might save updated info if we had any updates
    # Right now no training is done, but let's show how we might do it:
    # save_dummy_model("nightshade_model.pkl")

    print("[INFO] Finished. Exiting.")

if __name__ == "__main__":
    main()
