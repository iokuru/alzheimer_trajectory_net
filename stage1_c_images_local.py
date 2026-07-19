import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchio as tio

class OasisLongitudinalDataset(Dataset):
    def __init__(self, manifest_path, processed_dir, features, is_training=False):
        """
        Dataset for longitudinal 3D MRI data.
        Compatible with C-order (fortran_order=False) .npy files.
        """
        self.processed_dir = processed_dir
        self.features = features
        self.is_training = is_training
        
        # 1. Load manifest
        raw_manifest = pd.read_csv(manifest_path)
        
        # 2. Filter files existing on disk
        valid_rows = []
        for _, row in raw_manifest.iterrows():
            tensor_filename = row.get('Tensor_File', f"{row['MRI ID']}.npy")
            tensor_path = os.path.join(self.processed_dir, tensor_filename)
            if os.path.exists(tensor_path):
                valid_rows.append(row)
                
        if len(valid_rows) == 0:
            raise FileNotFoundError(f"No valid .npy files found in {processed_dir}.")
            
        self.manifest = pd.DataFrame(valid_rows)
        self.subjects = self.manifest['Subject ID'].unique()
        
        # Define 3D Spatial Augmentations
        if self.is_training:
            self.transform = tio.Compose([
                tio.RandomAffine(scales=(0.9, 1.1), degrees=15, translation=(5, 5, 5)),
                tio.RandomFlip(axes=('LR',)), 
                tio.RandomGamma(log_gamma=(-0.3, 0.3)), 
                tio.ZNormalization()
            ])
        else:
            self.transform = tio.ZNormalization()

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        subject_data = self.manifest[self.manifest['Subject ID'] == subject_id].sort_values('Visit')
        
        images = []
        tabular_data = []
        
        for _, row in subject_data.iterrows():
            tensor_filename = row.get('Tensor_File', f"{row['MRI ID']}.npy")
            tensor_path = os.path.join(self.processed_dir, tensor_filename)
            
            # C-order .npy files are read natively by numpy without transposition overhead[cite: 9]
            image_np = np.load(tensor_path)
            
            image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)
            subject = tio.Subject(mri=tio.ScalarImage(tensor=image_tensor))
            
            subject = self.transform(subject)
            images.append(subject.mri.data)
            
            tab_vals = row[self.features].values.astype(np.float32)
            tabular_data.append(torch.tensor(tab_vals))
            
        return {
            'subject_id': subject_id,
            'images': torch.stack(images),
            'tabular': torch.stack(tabular_data),
            'label': torch.tensor(subject_data.iloc[0]['Trajectory_Label'], dtype=torch.long),
            'sequence_length': len(images)
        }

def collate_patient_sequences(batch):
    """
    Pads longitudinal visit sequences and creates masks.
    """
    subject_ids = [item['subject_id'] for item in batch]
    images = [item['images'] for item in batch]
    tabular = [item['tabular'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    padded_images = pad_sequence(images, batch_first=True, padding_value=0.0)
    padded_tabular = pad_sequence(tabular, batch_first=True, padding_value=0.0)
    
    batch_size = len(batch)
    max_len = padded_images.shape[1]
    masks = torch.zeros((batch_size, max_len), dtype=torch.float32)
    
    for i, item in enumerate(batch):
        masks[i, :item['sequence_length']] = 1.0
        
    return {
        'subject_ids': subject_ids,
        'images': padded_images,
        'tabular': padded_tabular,
        'labels': labels,
        'masks': masks
    }

def apply_3d_cutmix(images, labels, alpha=1.0):
    """
    Volumetric 3D CutMix for batch sequences.
    """
    if np.random.rand() > 0.5:
        return images, labels, labels, 1.0
        
    batch_size = images.size(0)
    indices = torch.randperm(batch_size).to(images.device)
    
    # Check for (B, T, C, D, H, W)
    if len(images.shape) == 6:
        _, _, _, d, h, w = images.shape
        lam = np.random.beta(alpha, alpha)
        cut_rat = np.sqrt(1. - lam)
        
        # Calculate random bounding box coordinates
        cz, cy, cx = np.random.randint(d), np.random.randint(h), np.random.randint(w)
        cut_d, cut_h, cut_w = int(d * cut_rat), int(h * cut_rat), int(w * cut_rat)
        
        bbz1, bbz2 = np.clip(cz - cut_d // 2, 0, d), np.clip(cz + cut_d // 2, 0, d)
        bby1, bby2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
        bbx1, bbx2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
        
        images[:, :, :, bbz1:bbz2, bby1:bby2, bbx1:bbx2] = images[indices, :, :, bbz1:bbz2, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbz2 - bbz1) * (bby2 - bby1) * (bbx2 - bbx1) / (d * h * w))
        
    return images, labels, labels[indices], lam