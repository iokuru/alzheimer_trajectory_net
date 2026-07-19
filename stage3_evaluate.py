import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from src.dataset import OasisLongitudinalDataset, collate_patient_sequences
from src.models.temporal_engine import AlzheimerTrajectoryNet

class SpatialGradCAM3D:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        # Fix Deficiency 3: Accessing the layer module configuration path via .blocks array mapping
        target_layer = self.model.vision_branch.blocks[2].layers[-1]
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, batch, target_class):
        self.model.zero_grad()
        
        images = batch['images'].to(self.device)
        tabular = batch['tabular'].to(self.device)
        masks = batch['masks'].to(self.device)
        
        last_valid_visit_idx = int(masks.sum(dim=1).item() - 1)
        class_logits, _ = self.model(images, tabular, masks)
        
        score = class_logits[0, target_class]
        score.backward()
        
        target_step_idx = last_valid_visit_idx 
        step_gradients = self.gradients[target_step_idx:target_step_idx+1]
        step_activations = self.activations[target_step_idx:target_step_idx+1]
        
        pooled_gradients = torch.mean(step_gradients, dim=[2, 3, 4]).squeeze()
        
        for i in range(step_activations.shape[1]):
            step_activations[:, i, :, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(step_activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
            
        return heatmap

def save_confusion_matrix(y_true, y_pred, output_path):
    labels = ['Nondemented', 'Converted', 'Demented']
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Multi-Modal Trajectory Prediction Confusion Matrix')
    plt.ylabel('Ground Truth')
    plt.xlabel('Model Predictions')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f" Metrics Plot exported successfully to: {output_path}")

def main():
    print("=== Stage 3: Initializing Multi-Modal Model Evaluation Engine (Secure) ===")
    
    # Fix Deficiency 1 & 2: Exactly matches the 6 tracked features list definition layout
    FEATURES = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV']
    
    PROCESSED_DIR = os.path.join("data", "processed")
    CHECKPOINT_FILE = os.path.join("checkpoints", "best_trajectory_model.pt")
    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dataset = OasisLongitudinalDataset(os.path.join("manifests", "test_manifest.csv"), PROCESSED_DIR, FEATURES)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_patient_sequences)
    
    model = AlzheimerTrajectoryNet(tabular_dim=len(FEATURES), embed_dim=128, num_classes=3).to(device)
    
    if not os.path.exists(CHECKPOINT_FILE):
        raise FileNotFoundError(f"Missing model parameters file at: {CHECKPOINT_FILE}. Run Stage 2 optimizer first.")
        
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f" Successfully reloaded optimized model weights from Epoch {checkpoint['epoch']} loss minimum.")
    
    grad_cam_engine = SpatialGradCAM3D(model)
    
    all_preds = []
    all_trues = []
    heatmap_generated = False
    
    print("\n Sweeping inference configurations over the test split manifest...")
    for batch in test_loader:
        images = batch['images'].to(device)
        tabular = batch['tabular'].to(device)
        masks = batch['masks'].to(device)
        labels = batch['labels'].to(device)
        
        model.eval()
        with torch.no_grad():
            class_logits, _ = model(images, tabular, masks)
            _, preds = torch.max(class_logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(labels.cpu().numpy())
            
        if not heatmap_generated and labels.item() == 1:
            print(f"   --> Found high-risk 'Converted' patient ({batch['subject_ids'][0]}). Generating 3D Explainability Heatmap...")
            
            model.train() 
            cam_3d = grad_cam_engine.generate_heatmap(batch, target_class=1)
            model.eval()
            
            mid_slice = cam_3d.shape[0] // 2
            plt.figure(figsize=(5, 5))
            plt.imshow(cam_3d[mid_slice, :, :], cmap='jet')
            plt.title(f"3D Grad-CAM Target Focus: {batch['subject_ids'][0]} (Mid-Brain Voxel Slice)")
            plt.axis('off')
            plt.savefig(os.path.join(OUTPUT_DIR, f"{batch['subject_ids'][0]}_gradcam_slice.png"))
            plt.close()
            
            heatmap_generated = True

    print("\n=== Publicly Explainable Classification Deliverables ===")
    print(classification_report(all_trues, all_preds, target_names=['Nondemented', 'Converted', 'Demented']))
    
    save_confusion_matrix(all_trues, all_preds, os.path.join(OUTPUT_DIR, "final_test_confusion_matrix.png"))
    print("\n=== All Evaluation Operations Finalized Successfully! Review '/outputs' folder for plots. ===")

if __name__ == "__main__":
    main()