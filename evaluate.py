import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.seed import set_seed
from utils.metrics import calculate_metrics
from utils.visualization import plot_classification_map, CLASS_NAMES
from datasets.dataset import (
    load_indian_pines,
    create_pixel_patches,
    split_train_test,
    IndianPinesDataset
)
from models.model import FactoDualX

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a Trained FactoDual-X Model")
    
    # Dataset parameters
    parser.add_argument("--data_path", type=str, default="datasets/Indian_pines_corrected.mat",
                        help="Path to the Indian Pines corrected dataset (.mat)")
    parser.add_argument("--gt_path", type=str, default="datasets/Indian_pines_gt.mat",
                        help="Path to the Indian Pines ground truth dataset (.mat)")
    parser.add_argument("--patch_size", type=int, default=11, help="Spatial patch size")
    parser.add_argument("--spectral_int", type=int, default=10, help="Band group size / spectral interval")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--depth", type=int, default=4, help="Number of transformer blocks")
    parser.add_argument("--mlp_ratio", type=int, default=4, help="MLP expansion ratio")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    
    # Evaluation parameters
    parser.add_argument("--train_ratio", type=float, default=0.10, help="Ratio of training samples per class")
    parser.add_argument("--min_per_class", type=int, default=15, help="Minimum training samples per class")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/factodualx_best.pth",
                        help="Path to the model checkpoint (.pth)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_path", type=str, default="images/prediction_map_eval.png",
                        help="Output path to save prediction map visualization")
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    if not os.path.exists(args.data_path) or not os.path.exists(args.gt_path):
        raise FileNotFoundError(
            f"Dataset files not found at {args.data_path} or {args.gt_path}. "
            "Please ensure they are downloaded and placed correctly."
        )
        
    print("Loading Indian Pines dataset...")
    data, gt = load_indian_pines(args.data_path, args.gt_path)
    patches = create_pixel_patches(data, args.patch_size)
    train_idx, test_idx = split_train_test(gt, args.train_ratio, args.min_per_class, seed=args.seed)
    
    test_dataset = IndianPinesDataset(patches, gt.ravel(), test_idx, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Model initialization
    num_classes = len(CLASS_NAMES)
    dk = args.d_model // args.num_heads
    
    model = FactoDualX(
        P=args.patch_size,
        C=data.shape[2],
        num_classes=num_classes,
        B=args.spectral_int,
        D=args.d_model,
        depth=args.depth,
        heads=args.num_heads,
        dk=dk,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout
    ).to(device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint}")
        
    print(f"Loading checkpoint from: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    # Evaluation
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            logits = model(x.to(device))
            preds.extend(logits.argmax(dim=1).cpu().numpy())
    preds = np.array(preds)
    
    # Compute metrics
    true_labels = gt.ravel()[test_idx] - 1
    metrics = calculate_metrics(true_labels, preds, target_names=CLASS_NAMES)
    
    print("\n=== Evaluation Results ===")
    print(f"Overall Accuracy (OA) : {metrics['oa'] * 100:.2f}%")
    print(f"Average Accuracy (AA) : {metrics['aa'] * 100:.2f}%")
    print(f"Cohen's Kappa (k)     : {metrics['kappa']:.4f}")
    
    print("\n--- Classification Report ---")
    print(metrics["report"])
    
    # Save/Plot classification map
    plot_classification_map(gt, preds, train_idx, test_idx, save_path=args.save_path)
    print(f"Saved prediction map to: {args.save_path}")

if __name__ == "__main__":
    main()
