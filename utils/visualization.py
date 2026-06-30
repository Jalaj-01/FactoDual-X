import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

CLASS_NAMES = [
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
    "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
    "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean",
    "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"
]

# Colors for each class index 0-16 (0 = background/black)
COLORS_HEX = [
    '#000000',   #  0  Background
    '#1F7870',   #  1  Alfalfa             (dark teal)
    '#E91C20',   #  2  Corn-notill         (red)
    '#3A7DB9',   #  3  Corn-mintill        (blue)
    '#4DAE49',   #  4  Corn                (green)
    '#974FA3',   #  5  Grass-pasture       (purple)
    '#F87D0C',   #  6  Grass-trees         (orange)
    '#F56D43',   #  7  Grass-pasture-mowed (salmon)
    '#FCBB76',   #  8  Hay-windrowed       (light orange)
    '#2E78B5',   #  9  Oats                (blue)
    '#955627',   # 10  Soybean-notill      (brown)
    '#DDB2C4',   # 11  Soybean-mintill     (pink)
    '#5A5A5A',   # 12  Soybean-clean       (gray)
    '#515179',   # 13  Wheat               (dark blue-gray)
    '#55AAB0',   # 14  Woods               (teal)
    '#DA8D21',   # 15  Buildings-Grass-Trees-Drives (gold)
    '#A37B74',   # 16  Stone-Steel-Towers  (mauve)
]

def plot_classification_map(gt: np.ndarray, predictions: np.ndarray, 
                            train_idx: np.ndarray, test_idx: np.ndarray, 
                            save_path: str = "images/prediction_map.png"):
    """
    Plots the ground truth classification map and the model prediction map side-by-side.
    
    Args:
        gt (np.ndarray): Ground truth labels array of shape (H, W).
        predictions (np.ndarray): Flattened model predictions for test index set (0-indexed).
        train_idx (np.ndarray): Flattened pixel indices used for training.
        test_idx (np.ndarray): Flattened pixel indices used for testing.
        save_path (str): Output path to save the generated image file.
    """
    H, W = gt.shape

    # Build full prediction map (fill training pixels from ground truth)
    full_pred = np.zeros(H * W, dtype=np.int64)
    full_pred[test_idx] = predictions + 1
    full_pred[train_idx] = gt.ravel()[train_idx]
    full_pred = full_pred.reshape(H, W)

    # Mask background pixels to 0 (black) in prediction map
    masked_pred = np.where(gt == 0, 0, full_pred)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cmap = mcolors.ListedColormap(COLORS_HEX)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(gt, cmap=cmap, vmin=0, vmax=16)
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    axes[1].imshow(masked_pred, cmap=cmap, vmin=0, vmax=16)
    axes[1].set_title("FactoDual-X Prediction")
    axes[1].axis('off')

    handles = [
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=COLORS_HEX[i], markersize=10)
        for i in range(1, 17)
    ]
    fig.legend(handles, CLASS_NAMES, loc='center right', fontsize=8, title="Classes")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved → {save_path}")
