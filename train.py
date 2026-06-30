import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.seed import set_seed
from utils.scheduler import WarmupCosineScheduler
from utils.metrics import calculate_metrics
from utils.visualization import plot_classification_map, CLASS_NAMES, COLORS_HEX
from datasets.dataset import (
    load_indian_pines,
    create_pixel_patches,
    split_train_test,
    compute_class_weights,
    IndianPinesDataset
)
from models.model import FactoDualX

def get_args():
    parser = argparse.ArgumentParser(description="Train FactoDual-X on Indian Pines Dataset")
    
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
    
    # Training parameters
    parser.add_argument("--train_ratio", type=float, default=0.10, help="Ratio of training samples per class")
    parser.add_argument("--min_per_class", type=int, default=15, help="Minimum training samples per class")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs for scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_name", type=str, default="factodualx_best.pth", help="Checkpoint filename")
    
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
            "Please download the dataset files and place them in the datasets/ directory. "
            "Refer to datasets/README.md for download links."
        )
    
    print("Loading Indian Pines dataset...")
    data, gt = load_indian_pines(args.data_path, args.gt_path)
    patches = create_pixel_patches(data, args.patch_size)
    train_idx, test_idx = split_train_test(gt, args.train_ratio, args.min_per_class, seed=args.seed)
    
    # Data loaders
    train_dataset = IndianPinesDataset(patches, gt.ravel(), train_idx, augment=True)
    test_dataset = IndianPinesDataset(patches, gt.ravel(), test_idx, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
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
    
    # Loss, optimizer, and scheduler
    class_weights = compute_class_weights(gt.ravel(), train_idx, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)
    
    print("Training model...")
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{args.epochs:03d} | Loss: {epoch_loss / len(train_loader):.4f}")
            
    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, args.save_name)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint → {checkpoint_path}")
    
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
    
    print("\n--- Classification Report ---")
    print(metrics["report"])
    print(f"Overall Accuracy : {metrics['oa'] * 100:.2f}%")
    print(f"Average Accuracy : {metrics['aa'] * 100:.2f}%")
    print(f"Cohen's Kappa    : {metrics['kappa']:.4f}")
    
    # Save classification report
    os.makedirs("results", exist_ok=True)
    report_path = "results/classification_report.txt"
    with open(report_path, "w") as f:
        f.write("=== FactoDual-X Classification Report ===\n\n")
        f.write(metrics["report"])
        f.write(f"\nOverall Accuracy : {metrics['oa'] * 100:.2f}%\n")
        f.write(f"Average Accuracy : {metrics['aa'] * 100:.2f}%\n")
        f.write(f"Cohen's Kappa    : {metrics['kappa']:.4f}\n")
    print(f"Saved classification report → {report_path}")
    
    # Plot classification map
    plot_classification_map(gt, preds, train_idx, test_idx, save_path="images/prediction_map.png")

if __name__ == "__main__":
    main()
