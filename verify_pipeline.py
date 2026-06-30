import os
import subprocess
import numpy as np
import scipy.io as sio

def create_mock_dataset():
    print("Generating mock dataset for pipeline verification...")
    os.makedirs("datasets", exist_ok=True)
    
    # 15x15 image, 20 bands
    H, W, C = 15, 15, 20
    data = np.random.rand(H, W, C).astype(np.float32) * 100
    
    # Generate ground truth. We have 16 classes.
    # We will distribute classes 1 to 16, and some background 0.
    # 15 * 15 = 225 pixels. 16 classes * 10 = 160 pixels. Rest 65 pixels background (0).
    gt = np.zeros(H * W, dtype=np.int64)
    for c in range(1, 17):
        indices = np.arange((c-1)*10, c*10)
        gt[indices] = c
    np.random.shuffle(gt)
    gt = gt.reshape(H, W)
    
    data_path = "datasets/Indian_pines_corrected_mock.mat"
    gt_path = "datasets/Indian_pines_gt_mock.mat"
    
    sio.savemat(data_path, {'indian_pines_corrected': data})
    sio.savemat(gt_path, {'indian_pines_gt': gt})
    
    print(f"Mock dataset saved to: {data_path} and {gt_path}")
    return data_path, gt_path

def run_command(command_list):
    cmd_str = " ".join(command_list)
    print(f"\nRunning command: {cmd_str}")
    result = subprocess.run(command_list, capture_output=True, text=True)
    if result.returncode != 0:
        print("Command FAILED!")
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
        raise RuntimeError(f"Command failed: {cmd_str}")
    else:
        print("Command completed successfully.")
        print(result.stdout)

def main():
    data_path, gt_path = create_mock_dataset()
    
    # 1. Test train.py
    train_cmd = [
        "python", "train.py",
        "--data_path", data_path,
        "--gt_path", gt_path,
        "--epochs", "3",
        "--batch_size", "8",
        "--min_per_class", "2",
        "--train_ratio", "0.1",
        "--spectral_int", "5",  # B=5 (since C=20, N_spe = 20/5 = 4)
        "--d_model", "16",       # small d_model
        "--num_heads", "2",      # small heads
        "--checkpoint_dir", "checkpoints_mock",
        "--save_name", "factodualx_mock.pth"
    ]
    run_command(train_cmd)
    
    # 2. Test evaluate.py
    eval_cmd = [
        "python", "evaluate.py",
        "--data_path", data_path,
        "--gt_path", gt_path,
        "--min_per_class", "2",
        "--train_ratio", "0.1",
        "--spectral_int", "5",
        "--d_model", "16",
        "--num_heads", "2",
        "--checkpoint", "checkpoints_mock/factodualx_mock.pth",
        "--save_path", "images/prediction_map_eval_mock.png"
    ]
    run_command(eval_cmd)
    
    # 3. Test inference.py
    infer_cmd = [
        "python", "inference.py",
        "--data_path", data_path,
        "--gt_path", gt_path,
        "--spectral_int", "5",
        "--d_model", "16",
        "--num_heads", "2",
        "--checkpoint", "checkpoints_mock/factodualx_mock.pth",
        "--save_path", "images/prediction_map_only_mock.png"
    ]
    run_command(infer_cmd)
    
    # Clean up mock files and directories
    print("\nCleaning up mock test files...")
    for f in [data_path, gt_path, "checkpoints_mock/factodualx_mock.pth", 
              "images/prediction_map_eval_mock.png", "images/prediction_map_only_mock.png"]:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists("checkpoints_mock"):
        os.rmdir("checkpoints_mock")
        
    print("\nAll pipeline scripts verified successfully!")

if __name__ == "__main__":
    main()
