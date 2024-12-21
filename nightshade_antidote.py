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

# ---- Settings / Parameters ----
PATCH_SIZE = 8
CM_THRESHOLD = 0.1
DCT_SIZE = 512  # The size to which images are resized for DCT-based checks


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
    with exiftool.ExifTool() as et:
        metadata = et.get_metadata(img_path)
        for tag in metadata:
            report_str += f"{tag}: {metadata[tag]}\n"
    return report_str


def spectral_analysis(img):
    """
    Perform FFT on a grayscale image, returning (magnitude_spectrum, phase_spectrum).
    Also does a quick show in subplots.
    """
    # Convert to float64 for FFT
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
    However, if input is PNG, compression ratio logic won't reflect JPEG.
    """
    bit_depth = img.dtype.itemsize * 8
    # We can't guess actual uncompressed size for PNG, but let's just do a naive ratio:
    file_size = os.path.getsize(img_path)
    # approximate uncompressed size
    uncompressed_size = img.shape[0] * img.shape[1] * bit_depth/8
    if uncompressed_size == 0:
        ratio = 0
    else:
        ratio = file_size / uncompressed_size

    print(f"Bit depth: {bit_depth}")
    print(f"Approx Compression ratio: {ratio:.2f}")

    dct_img = cv2.dct(img.astype(np.float32))
    # Toy example of a standard luminance quant table
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
            # If block is not the same shape as the quant table, skip
            if block.shape != (8, 8):
                continue
            # remainder mod
            remainder = np.abs(block) % quantization_table
            threshold = quantization_table * 0.1
            # If remainder < threshold => we suspect quantization artifact
            flags_block = (remainder < threshold).astype(np.uint8)
            flags[i:i+step, j:j+step] = flags_block
    return flags


def file_format_check(img_path):
    """
    Check file format by reading the first few bytes. 
    If PNG, signature is 0x89 0x50 0x4E 0x47
    If JPEG, signature is 0xFF 0xD8, and end is 0xFF 0xD9
    """
    with open(img_path, "rb") as f:
        file_bytes = f.read()
    if len(file_bytes) < 8:
        print("File too small or corrupted. Can't identify format.")
        return

    # Check PNG
    if file_bytes[:4] == b"\x89PNG":
        print("File format appears to be PNG.")
    elif file_bytes[:2] == b"\xff\xd8":
        print("File format appears to be JPEG.")
    else:
        print("Unknown or non-PNG/JPEG file format signature.")


# --------------------------------------------------------------------------------
# Poison detection placeholder
# --------------------------------------------------------------------------------

def detect_poisoning(img_tensor):
    """
    This is a placeholder for advanced spectral/backdoor detection logic.
    Return True if poisoning is suspected, False otherwise.
    Replace with your actual logic from your 'NightshadeDetector' or other code.
    """
    # For now, random detection for demonstration
    return random.choice([False, True])


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
    # 1. File format
    file_format_check(img_path)

    # 2. Metadata
    print("\n--- Metadata Analysis ---")
    metadata_str = analyze_metadata(img_path)
    print(metadata_str)

    # 3. Copy-move
    print("--- Copy-Move Forgery Detection ---")
    flags = detect_copy_move(img)
    num_flags = np.sum(flags)
    if num_flags > 0:
        print(f"Detected {num_flags} regions likely copied and pasted.")
        plt.imshow(flags, cmap='gray')
        plt.title("Copy-Move Regions")
        plt.show()
    else:
        print("No copy-move forgery detected.")

    # 4. Spectral analysis
    print("\n--- Spectral Analysis ---")
    mag_spec, phase_spec = spectral_analysis(img)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.imshow(mag_spec, cmap='gray')
    ax1.set_title("Magnitude Spectrum")
    ax2.imshow(phase_spec, cmap='gray')
    ax2.set_title("Phase Spectrum")
    plt.show()

    # Check anomalies in magnitude
    mag_mean = np.mean(mag_spec)
    mag_std = np.std(mag_spec)
    mag_threshold = mag_mean + 3 * mag_std
    mag_anomalies = np.where(mag_spec > mag_threshold)
    if len(mag_anomalies[0]) > 0:
        print(f"Detected {len(mag_anomalies[0])} anomalies in the magnitude spectrum.")
    else:
        print("No anomalies detected in the magnitude spectrum.")

    # 5. Pixel ordering check (if reference.jpg is available)
    print("\n--- Pixel Ordering Check vs. reference.jpg ---")
    corr_coeff = pixel_ordering_check(img)
    if corr_coeff != 0.0:
        print(f"Correlation to reference: {corr_coeff:.4f}")
    else:
        print("No reference image found or correlation is 0.0")

    # 6. Compression artifacts
    print("\n--- Compression Artifacts Check ---")
    artifact_flags = compression_artifacts_check(img_path, img)
    artifact_count = np.sum(artifact_flags)
    if artifact_count > 0:
        print(f"Detected {artifact_count} potential compression artifact blocks.")
    else:
        print("No compression artifacts flagged.")
    
    # 7. Poison detection (placeholder)
    print("\n--- Poison Detection (Demo) ---")
    # Convert to (3,512,512) float32 for the logic
    # (Here, it's grayscale, so we just stack 3 channels or something)
    if len(img.shape) == 2:
        # stack to fake 3 channels
        cimg = np.stack([img]*3, axis=0)
    else:
        # reorder to (C,H,W)
        cimg = np.transpose(img, (2,0,1))
    cimg_t = torch.from_numpy(cimg).float() / 255. if img.dtype == np.uint8 else torch.from_numpy(cimg).float()
    suspicious = detect_poisoning(cimg_t)
    if suspicious:
        print("Poison detection indicates this image might be Nightshade-poisoned!")
    else:
        print("No poisoning suspected by placeholder logic.")


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

    # Read the image (png or jpg) using OpenCV
    img_cv = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img_cv is None:
        print(f"Could not read image {input_path} with OpenCV. Check file format.")
        sys.exit(1)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Resize to DCT_SIZE if desired
    img_gray = cv2.resize(img_gray, (DCT_SIZE, DCT_SIZE))
    # Convert to float [0..255], or normalized
    # but many checks assume [0..1], so let's keep [0..255] for the existing code except for some ops
    # We'll do [0..255] as is:
    
    # Generate the forensic report
    output_report(input_path, img_gray)

if __name__ == "__main__":
    main()
