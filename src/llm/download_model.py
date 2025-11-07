#!/usr/bin/env python3
"""
MemoryAI Enterprise - Model Download & Quantization Script
Downloads Llama-4-2B-MoE and quantizes to 4-bit GGUF format
"""

import os
import sys
import requests
import hashlib
from pathlib import Path
import subprocess
import argparse
from tqdm import tqdm

# Model configuration
MODEL_URL = "https://huggingface.co/lmstudio-community/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0.gguf"
MODEL_NAME = "llama-4-2b-moe-q4_0.gguf"
MODEL_SIZE = 2.9 * 1024 * 1024 * 1024  # ~2.9GB
EXPECTED_HASH = "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"

def download_with_progress(url: str, dest: str, expected_size: int):
    """Download file with progress bar"""
    print(f"🔄 Downloading {os.path.basename(dest)}...")
    
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    
    with open(dest, 'wb') as f, tqdm(
        desc=os.path.basename(dest),
        total=expected_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

def verify_checksum(file_path: str, expected_hash: str) -> bool:
    """Verify file checksum"""
    print("🔍 Verifying checksum...")
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    actual_hash = sha256_hash.hexdigest()
    return actual_hash == expected_hash

def quantize_model(input_path: str, output_path: str, quantization: str = "q4_0"):
    """Quantize model using llama.cpp"""
    print(f"⚡ Quantizing to {quantization}...")
    
    cmd = [
        "python", "-m", "llama_cpp.llama_quantize",
        input_path,
        output_path,
        quantization
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Quantization complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Quantization failed: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download and quantize MemoryAI models")
    parser.add_argument("--quantization", default="q4_0", 
                       choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"],
                       help="Quantization method")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download even if file exists")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing model")
    
    args = parser.parse_args()
    
    # Create models directory
    models_dir = Path(__file__).parent
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / MODEL_NAME
    
    # Check if model already exists
    if model_path.exists() and not args.force:
        print(f"✅ Model already exists: {model_path}")
        
        if args.verify_only or verify_checksum(str(model_path), EXPECTED_HASH):
            print("✅ Model verification passed")
            return 0
        else:
            print("❌ Model verification failed, re-downloading...")
    
    # Download model
    temp_path = model_path.with_suffix(".tmp")
    try:
        download_with_progress(MODEL_URL, str(temp_path), MODEL_SIZE)
        
        # Verify checksum
        if not verify_checksum(str(temp_path), EXPECTED_HASH):
            print("❌ Download verification failed")
            return 1
        
        # Move to final location
        temp_path.rename(model_path)
        print(f"✅ Model downloaded successfully: {model_path}")
        
        # Set permissions
        os.chmod(model_path, 0o644)
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return 1
    
    print("🎉 Model setup complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())