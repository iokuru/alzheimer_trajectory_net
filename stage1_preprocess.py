import os
import numpy as np
import pandas as pd
import torchio as tio
import SimpleITK as sitk
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define a standalone worker function for multi-processing compatibility
def process_single_mri(row_data, raw_data_dir, processed_dir):
    mri_id = str(row_data['MRI ID'])
    processed_filename = f"{mri_id}.npy"
    processed_filepath = os.path.join(processed_dir, processed_filename)
    
    # Skip if already processed
    if os.path.exists(processed_filepath):
        return mri_id, "Skipped (Already exists)"
        
    raw_dir_path = Path(raw_data_dir)
    subject_folder = raw_dir_path / mri_id
    found_files = list(subject_folder.rglob("*.hdr"))
    
    if not found_files:
        return mri_id, "Skipped (No .hdr file found)"
        
    raw_filepath = str(found_files[0])
    temp_filepath = os.path.join(processed_dir, f"temp_{mri_id}_{os.getpid()}.nii.gz")
    
    try:
        # Preprocessing Pipeline (Local instance per process to avoid multi-threading conflicts)
        preprocess_pipeline = tio.Compose([
            tio.ToCanonical(),
            tio.Resize((96, 96, 96)),
            tio.ZNormalization()
        ])
        
        # SimpleITK reads the .hdr file and automatically pairs it with the matching .img volume file
        input_image = sitk.ReadImage(raw_filepath, sitk.sitkFloat32)
        
        # Apply N4 Bias Field Correction
        mask_image = sitk.OtsuThreshold(input_image, 0, 1, 200)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_image = corrector.Execute(input_image, mask_image)
        
        # Temporary file write for TorchIO processing integration
        sitk.WriteImage(corrected_image, temp_filepath)
        
        # Handoff to TorchIO spatial pipeline
        subject = tio.Subject(mri=tio.ScalarImage(temp_filepath))
        processed_subject = preprocess_pipeline(subject)
        
        # Squeeze and convert processed data back out to a 3D NumPy array
        tensor_3d = processed_subject.mri.data.numpy().squeeze()
        np.save(processed_filepath, tensor_3d)
        
        return mri_id, "Success"
        
    except Exception as e:
        return mri_id, f"Failed: {str(e)}"
        
    finally:
        # Guarantee cleanup of temporary process files
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

def process_mri_volumes():
    print("=== Stage 1: Initializing Medical-Grade Parallel MRI Preprocessing Pipeline ===")
    
    RAW_DATA_DIR = os.path.join("data", "extracted")
    PROCESSED_DIR = os.path.join("data", "processed")
    MANIFEST_DIR = "manifests"
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Load manifests
    manifest_files = ['train_manifest.csv', 'val_manifest.csv', 'test_manifest.csv']
    dfs = [pd.read_csv(os.path.join(MANIFEST_DIR, m)) for m in manifest_files if os.path.exists(os.path.join(MANIFEST_DIR, m))]
    
    if not dfs:
        print(f"Error: No manifest files found in '{MANIFEST_DIR}'.")
        return
        
    full_manifest = pd.concat(dfs).drop_duplicates(subset=['MRI ID'])
    rows = [row for _, row in full_manifest.iterrows()]
    
    # Determine the number of worker CPU cores available (leaving 1-2 free to prevent OS lockup)
    num_workers = 3
    print(f" Launching parallel execution pool utilizing {num_workers} CPU cores.")
    print(f" Total tasks in manifest queue: {len(rows)}")

    # Execute concurrent preprocessing jobs across the CPU worker pool
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_mri, row, RAW_DATA_DIR, PROCESSED_DIR): row for row in rows}
        
        # Real-time progress metric bar update
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing MRIs concurrently"):
            mri_id, status = future.result()
            if "Failed" in status:
                print(f"\n[ERROR] {mri_id}: {status}")

    # Patch manifests to establish reference paths for Stage 2
    print("\n Finalizing database manifest update hooks...")
    for m in manifest_files:
        path = os.path.join(MANIFEST_DIR, m)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Tensor_File'] = df['MRI ID'].astype(str) + '.npy'
            df.to_csv(path, index=False)

    print("\n=== Stage 1 Complete! ===")

if __name__ == "__main__":
    process_mri_volumes()