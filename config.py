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
# ... (other occlusion params as before)

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
DATALOADER_BATCH_SIZE = 8 # Start with a safe batch size
NUM_WORKERS_DATALOADER = 0 
LEARNING_RATE = 5e-5       
CD_LOSS_NUM_POINTS = 8192

# --- Hybrid Loss Weights ---
# Total Loss = L_noise + LAMBDA_CD_LOSS * L_cd + LAMBDA_PARAMETRIC_LOSS * L_parametric
LAMBDA_CD_LOSS = 1.0 
LAMBDA_PARAMETRIC_LOSS = 0.5 # <--- NEW PARAMETER: Weight for the new parametric loss. 0.5 is a good starting point.

# --- Parametric Head Parameters ---
# List of semantic class IDs that are considered planar.
# You need to map this to your specific dataset's labels.
# Example mapping for SemanticKITTI (common):
# 40-49: road, sidewalk, parking, other-ground
# 50-51: building, fence
# 70: terrain
# We will use example class IDs here. You must verify these match your label mapping.
PLANAR_CLASS_IDS = [40, 44, 48, 49, 50, 51, 70] 

CHECKPOINT_DIR = "training_checkpoints"
SAVE_EVERY_N_EPOCHS = 5
VALIDATE_EVERY_N_EPOCHS = 1

# ---- Output Directories ----
EVALUATION_PCD_OUTPUT_DIR = "evaluation_pcd_outputs"
TRAINING_PLOTS_DIR = "training_plots"
MAX_SAMPLES_TO_EVALUATE = 100  # <--- ADD THIS LINE

# ---- Visualization Control ----
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
    print("--- Configuration Settings for Full-Scale Training with Parametric Head ---")
    print(f"Parametric Loss Weight (Lambda): {LAMBDA_PARAMETRIC_LOSS}")
    print(f"Planar Class IDs: {PLANAR_CLASS_IDS}")
    # ...
