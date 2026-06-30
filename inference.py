import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import Dataset, DataLoader

from utils.visualization import CLASS_NAMES, COLORS_HEX
from datasets.dataset import load_indian_pines, create_pixel_patches
from models.model import FactoDualX

class InferenceDataset(Dataset):
    def __init__(self, patches: np.ndarray):
        self.patches = patches
    def __len__(self):
        return len(self.patches)
    def __getitem__(self, idx: int):
        return torch.from_numpy(self.patches[idx]), torch.tensor(0)

def get_args():
    parser = argparse.ArgumentParser(description="Run Inference using FactoDual-X")
    
    # Dataset paths
    parser.add_argument("--data_path", type=str, default="datasets/Indian_pines_corrected.mat",
                        help="Path to the Indian Pines corrected dataset (.mat)")
    parser.add_argument("--gt_path", type=str, default="datasets/Indian_pines_gt.mat",
                        help="Path to the Indian Pines ground truth dataset (.mat) (for masking background)")
    parser.add_argument("--patch_size", type=int, default=11, help="Spatial patch size")
    parser.add_argument("--spectral_int", type=int, default=10, help="Band group size / spectral interval")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--depth", type=int, default=4, help="Number of transformer blocks")
    parser.add_argument("--mlp_ratio", type=int, default=4, help="MLP expansion ratio")
    
    # Inference parameters
    parser.add_argument("--checkpoint", type=str, default="checkpoints/factodualx_best.pth",
                        help="Path to the model checkpoint (.pth)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--save_path", type=str, default="images/prediction_map_only.png",
                        help="Output path to save the prediction map image")
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset file not found at {args.data_path}")
        
    print("Loading image for inference...")
    data, gt = load_indian_pines(args.data_path, args.gt_path)
    H, W, C = data.shape
    patches = create_pixel_patches(data, args.patch_size)
    
    # Setup data loader
    inference_dataset = InferenceDataset(patches)
    loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model initialization
    num_classes = len(CLASS_NAMES)
    dk = args.d_model // args.num_heads
    
    model = FactoDualX(
        P=args.patch_size,
        C=C,
        num_classes=num_classes,
        B=args.spectral_int,
        D=args.d_model,
        depth=args.depth,
        heads=args.num_heads,
        dk=dk,
        mlp_ratio=args.mlp_ratio,
        dropout=0.0
    ).to(device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint}")
    print(f"Loading weights from: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    # Run prediction
    model.eval()
    preds = []
    print("Running pixel-wise model inference...")
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x.to(device))
            preds.extend(logits.argmax(dim=1).cpu().numpy())
    preds = np.array(preds)
    
    # Reshape predictions back to 2D image coordinates (1-indexed classes)
    pred_map = (preds + 1).reshape(H, W)
    
    # Mask out background if ground truth is available
    if gt is not None:
        pred_map = np.where(gt == 0, 0, pred_map)
        
    # Save the classification map
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    cmap = mcolors.ListedColormap(COLORS_HEX)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(pred_map, cmap=cmap, vmin=0, vmax=16)
    plt.axis('off')
    plt.title("FactoDual-X Prediction Map", fontsize=14)
    
    # Add legend
    handles = [
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=COLORS_HEX[i], markersize=10)
        for i in range(1, 17)
    ]
    plt.legend(handles, CLASS_NAMES, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, title="Classes")
    plt.tight_layout()
    plt.savefig(args.save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction map visualization to: {args.save_path}")

if __name__ == "__main__":
    main()
