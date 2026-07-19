import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, recall_score
from stage1_c_images_local import OasisLongitudinalDataset, collate_patient_sequences, apply_3d_cutmix
from src.models.temporal_engine import AlzheimerTrajectoryNet

# Focal Loss Implementation to stabilize minority class learning
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

def train_one_epoch(model, dataloader, optimizer, criterion_cls, criterion_reg, device, mmse_idx, lambda_reg):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
            
    for batch in dataloader:
        images = batch['images'].to(device)
        tabular = batch['tabular'].to(device)
        masks = batch['masks'].to(device)
        labels = batch['labels'].to(device)
        
        last_visit_indices = (masks.sum(dim=1) - 1).long().to(device)
        batch_indices = torch.arange(tabular.size(0), device=device)
        target_mmse = tabular[batch_indices, last_visit_indices, mmse_idx]
        
        optimizer.zero_grad()
        images, labels_a, labels_b, lam = apply_3d_cutmix(images, labels, alpha=1.0)
        
        class_logits, mmse_preds = model(images, tabular, masks)
        
        loss_cls = lam * criterion_cls(class_logits, labels_a) + (1 - lam) * criterion_cls(class_logits, labels_b)
        loss_reg = criterion_reg(mmse_preds, target_mmse)
        total_loss = loss_cls + lambda_reg * loss_reg
            
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += total_loss.item() * images.size(0)
        _, preds = torch.max(class_logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    return running_loss / len(dataloader.dataset), f1_score(all_labels, all_preds, average='macro')

def validate_one_epoch(model, dataloader, criterion_cls, criterion_reg, device, mmse_idx, lambda_reg):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            images, tabular, masks, labels = batch['images'].to(device), batch['tabular'].to(device), batch['masks'].to(device), batch['labels'].to(device)
            
            last_visit_indices = (masks.sum(dim=1) - 1).long().to(device)
            batch_indices = torch.arange(tabular.size(0), device=device)
            target_mmse = tabular[batch_indices, last_visit_indices, mmse_idx]
            
            class_logits, mmse_preds = model(images, tabular, masks)
            loss_cls = criterion_cls(class_logits, labels)
            loss_reg = criterion_reg(mmse_preds, target_mmse)
            
            running_loss += (loss_cls + lambda_reg * loss_reg).item() * images.size(0)
            _, preds = torch.max(class_logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return running_loss / len(dataloader.dataset), f1_score(all_labels, all_preds, average='macro'), recall_score(all_labels, all_preds, average=None)

def main():
    print("=== Stage 2: Initializing Robust Optimization Engine ===")
    
    BATCH_SIZE, EPOCHS, LEARNING_RATE = 4, 40, 1e-4
    FEATURES = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV']
    MMSE_INDEX = FEATURES.index('MMSE')
    
    TRAIN_MANIFEST = os.path.join("manifests", "train_manifest.csv")
    VAL_MANIFEST = os.path.join("manifests", "val_manifest.csv")
    PROCESSED_DIR = "data/processed"
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_df = pd.read_csv(TRAIN_MANIFEST)
    train_dataset = OasisLongitudinalDataset(TRAIN_MANIFEST, PROCESSED_DIR, FEATURES, is_training=True)
    val_dataset = OasisLongitudinalDataset(VAL_MANIFEST, PROCESSED_DIR, FEATURES, is_training=False)
    
    class_sample_count = train_df['Trajectory_Label'].value_counts().sort_index().values
    weight = 1. / torch.tensor(class_sample_count, dtype=torch.float)
    samples_weight = [weight[train_df[train_df['Subject ID'] == subj]['Trajectory_Label'].iloc[0]] for subj in train_dataset.subjects]
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_patient_sequences)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_patient_sequences)
    
    model = AlzheimerTrajectoryNet(tabular_dim=len(FEATURES), embed_dim=128, num_classes=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Using Focal Loss for imbalance and reduced lambda for classification stability
    criterion_cls = FocalLoss(alpha=1, gamma=2)
    criterion_reg = nn.MSELoss()
    
    best_val_loss, patience_counter, patience = float('inf'), 0, 8
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion_cls, criterion_reg, device, MMSE_INDEX, lambda_reg=0.05)
        val_loss, val_f1, val_recalls = validate_one_epoch(model, val_loader, criterion_cls, criterion_reg, device, MMSE_INDEX, lambda_reg=0.05)
        
        scheduler.step()
        print(f"Epoch {epoch:02d} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Recalls: {np.round(val_recalls, 2)}")
        
        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    main()