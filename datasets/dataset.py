import random
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

def load_indian_pines(data_path: str, gt_path: str):
    """
    Loads and normalizes the Indian Pines hyperspectral dataset.
    
    Args:
        data_path (str): Path to Indian_pines_corrected.mat
        gt_path (str): Path to Indian_pines_gt.mat
    Returns:
        tuple: (normalized data array of shape (H, W, C), ground truth array of shape (H, W))
    """
    data = sio.loadmat(data_path)['indian_pines_corrected'].astype(np.float32)
    gt = sio.loadmat(gt_path)['indian_pines_gt'].astype(np.int64)
    H, W, C = data.shape
    data_2d = data.reshape(-1, C)
    scaler = StandardScaler()
    data = scaler.fit_transform(data_2d).reshape(H, W, C).astype(np.float32)
    return data, gt

def create_pixel_patches(data: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Generates overlapping patches centered at each pixel location.
    Pads the data with reflect boundary condition to handle edges.
    
    Args:
        data (np.ndarray): Hyperspectral image of shape (H, W, C)
        patch_size (int): Size of the square patch (must be odd)
    Returns:
        np.ndarray: Patch tensor of shape (H * W, patch_size, patch_size, C)
    """
    H, W, C = data.shape
    half = patch_size // 2
    padded = np.pad(data, ((half, half), (half, half), (0, 0)), mode='reflect')
    patches = np.lib.stride_tricks.sliding_window_view(
        padded, (patch_size, patch_size, C)
    ).reshape(H * W, patch_size, patch_size, C)
    return patches.astype(np.float32)

def split_train_test(gt: np.ndarray, train_ratio: float = 0.10, 
                     min_per_class: int = 15, seed: int = 42):
    """
    Splits pixel indices into training and testing sets, ensuring minimum samples per class.
    
    Args:
        gt (np.ndarray): Ground truth labels of shape (H, W) (0 is background/ignored)
        train_ratio (float): Ratio of training samples per class.
        min_per_class (int): Minimum number of training samples per class.
        seed (int): Random seed for split reproducibility.
    Returns:
        tuple: (train_idx, test_idx) as 1D numpy arrays of flat index indices.
    """
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for cls in range(1, gt.max() + 1):
        idx = np.where(gt.ravel() == cls)[0]
        n_total = len(idx)
        if n_total == 0:
            continue
        n_train = max(min_per_class, int(n_total * train_ratio))
        # Ensure we don't select more than half of the total available samples for training
        n_train = min(n_train, n_total // 2)
        chosen = rng.choice(idx, n_train, replace=False)
        train_idx.extend(chosen.tolist())
        test_idx.extend(np.setdiff1d(idx, chosen).tolist())
    return np.array(train_idx), np.array(test_idx)

def compute_class_weights(labels: np.ndarray, train_idx: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Computes class weights for balancing cross-entropy loss based on training distribution.
    
    Args:
        labels (np.ndarray): Flattened labels array (1-indexed, or with background).
        train_idx (np.ndarray): Indices of training samples.
        num_classes (int): Number of target classes.
    Returns:
        torch.Tensor: Normalized class weights.
    """
    train_labels = labels[train_idx]
    counts = np.bincount(train_labels - 1, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / np.sqrt(counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)

class IndianPinesDataset(Dataset):
    """
    PyTorch Dataset wrapper for Indian Pines patches.
    Includes data augmentation (flips, rotations, Gaussian noise) for training.
    """
    def __init__(self, patches: np.ndarray, labels: np.ndarray, 
                 indices: np.ndarray, augment: bool = False):
        """
        Args:
            patches (np.ndarray): Full patch array of shape (N, P, P, C)
            labels (np.ndarray): 1-indexed ground truth array of shape (N,)
            indices (np.ndarray): Active indices for this dataset split.
            augment (bool): Whether to apply data augmentation.
        """
        self.patches = patches[indices]
        # Convert labels to 0-indexed for CrossEntropyLoss
        self.labels = labels[indices] - 1
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.patches[idx].copy()
        if self.augment:
            if random.random() > 0.5:
                x = x[::-1].copy()
            if random.random() > 0.5:
                x = x[:, ::-1].copy()
            x = np.rot90(x, random.randint(0, 3), axes=(0, 1)).copy()
            x += np.random.normal(0, 0.005, x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(self.labels[idx], dtype=torch.long)
