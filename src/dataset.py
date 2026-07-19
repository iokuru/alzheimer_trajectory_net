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
        Args:
            manifest_path: Path to train_manifest.csv or val_manifest.csv.
            processed_dir: Directory containing the preprocessed .npy tensors.
            features: List of tabular continuous features to track.
            is_training: Flag to apply heavy 3D Torchio augmentations on-the-fly.
        """
        self.manifest = pd.read_csv(manifest_path)
        self.processed_dir = processed_dir
        self.features = features
        self.is_training = is_training
        
        # Explicitly targets 'Subject ID' to match manifest columns exactly
        self.subjects = self.manifest['Subject ID'].unique()
        
        # Define Medical-Grade Spatial Augmentations using TorchIO
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
        # Sorts by 'Visit' to preserve longitudinal chronological tracking
        subject_data = self.manifest[self.manifest['Subject ID'] == subject_id].sort_values('Visit')
        
        images = []
        tabular_data = []
        
        for _, row in subject_data.iterrows():
            # FORCED RE-ALIGNMENT: Dynamically builds path utilizing the true '_3d.npy' disk format
            mri_id = str(row['MRI ID'])
            tensor_filename = f"{mri_id}_3d.npy"
            tensor_path = os.path.join(self.processed_dir, tensor_filename)
            
            image_np = np.load(tensor_path)
            
            # Convert to TorchIO Subject for robust 3D augmentations
            image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)
            subject = tio.Subject(mri=tio.ScalarImage(tensor=image_tensor))
            
            # Apply dynamic augmentations
            subject = self.transform(subject)
            images.append(subject.mri.data)
            
            # Extract Tabular Tracking Metrics
            tab_vals = row[self.features].values.astype(np.float32)
            tabular_data.append(torch.tensor(tab_vals))
            
        images_stack = torch.stack(images)       # (T, 1, 96, 96, 96)
        tabular_stack = torch.stack(tabular_data) # (T, Num_Features)
        label = torch.tensor(subject_data.iloc[0]['Trajectory_Label'], dtype=torch.long)
        
        return {
            'subject_id': subject_id,
            'images': images_stack,
            'tabular': tabular_stack,
            'label': label,
            'sequence_length': len(images)
        }

def collate_patient_sequences(batch):
    """
    Pads longitudinal visit sequences to the maximum sequence length in the batch.
    Returns tracking masks to ensure the GRU ignores padded phantom visits.
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
        seq_len = item['sequence_length']
        masks[i, :seq_len] = 1.0
        
    return {
        'subject_ids': subject_ids,
        'images': padded_images,
        'tabular': padded_tabular,
        'labels': labels,
        'masks': masks
    }

def apply_3d_cutmix(images, labels, alpha=1.0):
    """
    Volumetric CutMix specifically designed for 3D MRI Tensors.
    Swaps sub-volumes between batch samples to force global feature learning.
    """
    if np.random.rand() > 0.5:
        return images, labels, labels, 1.0
        
    batch_size = images.size(0)
    indices = torch.randperm(batch_size).to(images.device)
    
    _, _, _, d, h, w = images.shape
    
    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1. - lam)
    
    cut_d = int(d * cut_rat)
    cut_h = int(h * cut_rat)
    cut_w = int(w * cut_rat)
    
    cz = np.random.randint(d)
    cy = np.random.randint(h)
    cx = np.random.randint(w)
    
    bbz1 = np.clip(cz - cut_d // 2, 0, d)
    bbz2 = np.clip(cz + cut_d // 2, 0, d)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    
    images[:, :, :, bbz1:bbz2, bby1:bby2, bbx1:bbx2] = images[indices, :, :, bbz1:bbz2, bby1:bby2, bbx1:bbx2]
    
    lam = 1 - ((bbz2 - bbz1) * (bby2 - bby1) * (bbx2 - bbx1) / (d * h * w))
    
    return images, labels, labels[indices], lam