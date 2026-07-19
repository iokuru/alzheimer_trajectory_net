import os
import torch
from torch.utils.data import DataLoader
from src.dataset import OasisLongitudinalDataset, collate_patient_sequences
from src.models.temporal_engine import AlzheimerTrajectoryNet

if __name__ == "__main__":
    FEATURES = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    MANIFEST = os.path.join("manifests", "train_manifest.csv")
    PROCESSED_DIR = os.path.join("data", "processed")
    
    # Initialize data pipeline
    dataset = OasisLongitudinalDataset(MANIFEST, PROCESSED_DIR, FEATURES)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_patient_sequences)
    
    # Grab one batch
    batch = next(iter(dataloader))
    
    # Instantiate the Master Trajectory Wrapper Model
    print("=== Initializing Master AlzheimerTrajectoryNet Module ===")
    model = AlzheimerTrajectoryNet(tabular_dim=len(FEATURES), embed_dim=128, num_classes=3)
    model.eval() # Shift to evaluation state to freeze dropout paths
    
    print("\n=== Executing Structural Network Handshake ===")
    with torch.no_grad():
        logits, mmse_pred = model(batch['images'], batch['tabular'], batch['masks'])
        
    print("\n SUCCESS: Model execution loop cleared cleanly!")
    print("-------------------------------------------------")
    print("Output Classification Logits Shape :", logits.shape)     # Expect: [2, 3]
    print("Output Regression MMSE Pred Shape  :", mmse_pred.shape)  # Expect: [2]