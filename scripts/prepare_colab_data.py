#!/usr/bin/env python3
"""Prepare data for Google Colab training."""

import pandas as pd
import numpy as np
from scipy.sparse import save_npz
import pickle
import json
from pathlib import Path

def prepare_colab_data():
    """Prepare data files for Google Colab."""
    print("🚀 Preparing data for Google Colab...")
    
    # Data paths
    data_dir = Path("data/processed")
    output_dir = Path("colab_data")
    output_dir.mkdir(exist_ok=True)
    
    # Files to copy
    files_to_copy = [
        "user_item_matrix.npz",
        "user_mapping.pkl", 
        "item_mapping.pkl",
        "matrix_info.json"
    ]
    
    print("📁 Copying matrix files...")
    for file in files_to_copy:
        src = data_dir / file
        dst = output_dir / file
        if src.exists():
            import shutil
            shutil.copy2(src, dst)
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} not found")
    
    # Create a simple data info file
    info = {
        "description": "Recommndr Phase 3 Training Data",
        "matrix_shape": [10000, 1000],
        "interactions": 99521,
        "sparsity": 0.99,
        "instructions": [
            "1. Upload all files to Google Colab",
            "2. Run the training notebook",
            "3. Download trained models back to local"
        ]
    }
    
    with open(output_dir / "colab_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\n📦 Data prepared in: {output_dir}")
    print("📋 Next steps:")
    print("   1. Upload 'colab_data' folder to Google Drive")
    print("   2. Open the Colab notebook")
    print("   3. Mount Google Drive and start training!")

if __name__ == "__main__":
    prepare_colab_data()
