# config.py

import torch
import os

# ---- Base Dataset Path ----
SEMANTIC_KITTI_BASE_DIR = "C:/Users/Dream Team/Desktop/PointCloud completion"

# ---- Preprocessed Data Paths ----
PREPROCESSED_DATA_DIR = os.path.join(SEMANTIC_KITTI_BASE_DIR, "preprocessed_completion_data")

# ---- Sequence and Scan Selection ----
TRAIN_SEQUENCES = [f"{i:02d}" for i in range(11) if i != 8]
VAL_SEQUENCES = ["08"]
MAX_SCANS_TO_PREPROCESS_PER_SEQUENCE = None

# ---- Occlusion Generation Parameters ----
NUM_OCCLUSIONS_PER_SCAN = 1
OCCLUSION_MIN_SIZE_METERS = 8
OCCLUSION_MAX_SIZE_METERS = 25
OCCLUSION_CENTER_XY_RANGE = 15
OCCLUSION_CENTER_Z_RANGE = 2

# ---- Model & Data Parameters ----
NUM_CLASSES = 20
POINT_DIM = 3
PLANE_DIST_DIM = 1
EMBED_DIM = 256
TIME_EMBED_DIM = 128
HIDDEN_DIM_DENOISING = 256

# ---- Diffusion Parameters ----
NUM_TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02

# ---- Full-Scale Training Parameters ----
NUM_GLOBAL_EPOCHS = 100
DATALOADER_BATCH_SIZE = 12
NUM_WORKERS_DATALOADER = 0
LEARNING_RATE = 1e-5

# --- Hybrid Loss Weight ---
# Weight for the Chamfer Distance loss component.
# Total Loss = L_noise + LAMBDA_CD_LOSS * L_cd
LAMBDA_CD_LOSS = 1.0 # <--- NEW PARAMETER
CD_LOSS_NUM_POINTS = 8192 # <--- NEW PARAMETER

CHECKPOINT_DIR = "training_checkpoints"
SAVE_EVERY_N_EPOCHS = 5
VALIDATE_EVERY_N_EPOCHS = 1

# ---- Output Directories ----
EVALUATION_PCD_OUTPUT_DIR = "evaluation_pcd_outputs"
TRAINING_PLOTS_DIR = "training_plots"
MAX_SAMPLES_TO_EVALUATE = 100 # we have a max sample of 4071

# ---- Visualization Control ----
VISUALIZE_PER_SCAN_DURING_PIPELINE = False
VISUALIZE_LAST_SCAN_POST_PIPELINE = True

# ---- Device Configuration ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Path Construction Logic Helper (unchanged) ----
def get_sequence_paths(base_dir, sequence_id):
    # ... (function as before)
    if not base_dir or not os.path.isdir(base_dir): return None
    paths = {
        "velodyne": os.path.join(base_dir, "data_odometry_velodyne", "dataset", "sequences", sequence_id, "velodyne"),
        "labels": os.path.join(base_dir, "data_odometry_labels", "dataset", "sequences", sequence_id, "labels"),
        "calib": os.path.join(base_dir, "data_odometry_calib", "dataset", "sequences", sequence_id, "calib.txt"), 
        "poses": os.path.join(base_dir, "data_odometry_calib", "dataset", "sequences", sequence_id, "poses.txt")
    }
    return paths

if __name__ == '__main__':
    # ... (__main__ block as before, can add a print for LAMBDA_CD_LOSS)
    print("--- Configuration Settings for Full-Scale Training ---")
    print(f"Hybrid Loss CD Weight (Lambda): {LAMBDA_CD_LOSS}")
    # ...