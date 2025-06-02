# config.py

import torch
import os

# ---- Base Dataset Path ----
# This should be the path to the main directory containing the "data_odometry_velodyne",
# "data_odometry_labels", and "data_odometry_calib" subdirectories, which in turn contain "dataset/sequences".
# OR, if your structure is directly "kitti_dataset/sequences/00/...", then this should be "kitti_dataset".
# Example: "C:/datasets/semantic_kitti" if your data is in "C:/datasets/semantic_kitti/data_odometry_velodyne/dataset/sequences/00/velodyne"
SEMANTIC_KITTI_BASE_DIR = "C:/Users/Dream Team/Desktop/PointCloud completion" # Modify this to your actual base path

# ---- Sequence and Scan Selection ----
# Specify the sequence ID(s) you want to process.
# This can be a single string like "00" or a list of strings like ["00", "01", "08"]
SEQUENCES_TO_PROCESS = ["00", "01", "02"] # Example: process sequence "00". Change to ["00", "01"] to process two sequences.

# How many scans to process from each selected sequence.
# Set to None to process all available scans in the selected sequence(s).
MAX_SCANS_PER_SEQUENCE = 10 # Example: process the first 3 scans of each sequence in SEQUENCES_TO_PROCESS


# ---- Model & Data Parameters ----
NUM_CLASSES = 20  # Semantic KITTI has 19 valid classes (0-18 after remapping for 0-indexed arrays),
                  # plus one for "unlabeled" if you map it that way.
                  # Ensure this matches your label processing in data_utils.py
POINT_DIM = 3     # Dimension of points (x, y, z)
PLANE_DIST_DIM = 1 # Dimension of plane distance feature
EMBED_DIM = 256    # Embedding dimension for context encoder
TIME_EMBED_DIM = 128 # Dimension for time embedding in Denoising Network
HIDDEN_DIM_DENOISING = 256 # Hidden dimension for Denoising Network MLP

# ---- Diffusion Parameters ----
NUM_TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02

# ---- Training Parameters ----
NUM_EPOCHS_PER_SCAN_CONCEPTUAL = 50 # Conceptual epochs per scan in main_pipeline.py
BATCH_SIZE_OCC_TRAINING = 1024
LEARNING_RATE = 5e-5
# 1e-4

# ---- Output Directories ----
COMPLETED_SCANS_OUTPUT_DIR = "completed_scans_output"
TRAINING_PLOTS_DIR = "training_plots" # Existing, but good to group

# ---- Visualization Control ----
VISUALIZE_PER_SCAN_DURING_PIPELINE = False # Set to True to see visualization after each scan processed
VISUALIZE_LAST_SCAN_POST_PIPELINE = True  # Set to True to see visualization for the very last scan after all processing

# ---- Device Configuration ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- Path Construction Logic Helper ----
# This function is now primarily for clarity or direct use if needed,
# but the main_pipeline.py will iterate through SEQUENCES_TO_PROCESS
# and construct paths for each sequence.

def get_sequence_paths(base_dir, sequence_id):
    """
    Constructs paths for velodyne, labels, and calib for a given sequence_id.
    Returns a dictionary with 'velodyne', 'labels', 'calib_poses'.
    Note: 'calib.txt' for projection matrices is usually in the sequence folder directly,
          and 'poses.txt' for scan poses.
    """
    if not base_dir or not os.path.isdir(base_dir):
        print(f"Warning: SEMANTIC_KITTI_BASE_DIR '{base_dir}' not found or not a directory.")
        return None

    seq_path_base = os.path.join(base_dir, "dataset", "sequences", sequence_id) # Common structure within data_odometry_*
    
    paths = {
        "velodyne": os.path.join(base_dir, "data_odometry_velodyne", "dataset", "sequences", sequence_id, "velodyne"),
        "labels": os.path.join(base_dir, "data_odometry_labels", "dataset", "sequences", sequence_id, "labels"),
        "calib": os.path.join(base_dir, "data_odometry_calib", "dataset", "sequences", sequence_id, "calib.txt"), # Lidar to Camera etc.
        "poses": os.path.join(base_dir, "data_odometry_calib", "dataset", "sequences", sequence_id, "poses.txt")  # Scan poses
    }

    # Validate paths
    for key, path in paths.items():
        if key in ["velodyne", "labels"]: # These must be directories
            if not os.path.isdir(path):
                print(f"Warning: Constructed {key} directory '{path}' does not exist for sequence {sequence_id}.")
                # return None # Or handle more gracefully
        else: # These are files
            if not os.path.isfile(path):
                print(f"Warning: Constructed {key} file '{path}' does not exist for sequence {sequence_id}.")
                # return None

    return paths


if __name__ == '__main__':
    print("--- Configuration Settings ---")
    print(f"Semantic KITTI Base Directory: {SEMANTIC_KITTI_BASE_DIR}")
    print(f"Sequences to Process: {SEQUENCES_TO_PROCESS}")
    print(f"Max Scans per Sequence: {MAX_SCANS_PER_SEQUENCE}")
    print(f"Device: {DEVICE}")
    print(f"Number of Classes: {NUM_CLASSES}")

    print("\nExample constructed paths for the first sequence in SEQUENCES_TO_PROCESS:")
    if SEQUENCES_TO_PROCESS:
        example_seq_id = SEQUENCES_TO_PROCESS[0]
        example_paths = get_sequence_paths(SEMANTIC_KITTI_BASE_DIR, example_seq_id)
        if example_paths:
            for key, path in example_paths.items():
                print(f"  {key.capitalize()} path: {path} (Exists: {os.path.exists(path)})")
    print("----------------------------")