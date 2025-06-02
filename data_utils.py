# data_utils.py

import numpy as np
import open3d as o3d
import os
import glob

# Attempt to import config.py
try:
    import config # Assuming config.py is in the same directory or Python path
except ImportError:
    print("CRITICAL WARNING in data_utils.py: config.py not found. Using fallback NUM_CLASSES.")
    print("Ensure config.py is in the Python path for the main pipeline to work.")
    class FallbackConfigForDataUtils: # Minimal fallback for the test block
        NUM_CLASSES = 20
        MAX_SCANS_PER_SEQUENCE = 1 # For testing
        SEMANTIC_KITTI_BASE_DIR = "." # Placeholder
        SEQUENCES_TO_PROCESS = []
        def get_sequence_paths(self, base_dir, sequence_id): # Dummy for test block
             # This dummy won't actually find real data unless SEMANTIC_KITTI_BASE_DIR is valid
            velodyne_path = os.path.join(base_dir, "data_odometry_velodyne", "dataset", "sequences", sequence_id, "velodyne")
            labels_path = os.path.join(base_dir, "data_odometry_labels", "dataset", "sequences", sequence_id, "labels")
            return {"velodyne": velodyne_path, "labels": labels_path}

    config = FallbackConfigForDataUtils()


def get_scan_paths(velodyne_dir, labels_dir, max_scans=None):
    """
    Finds matching .bin (velodyne) and .label (labels) files in the given sequence-specific directories.
    Sorts them to ensure correspondence.
    """
    if not velodyne_dir or not labels_dir or not os.path.isdir(velodyne_dir) or not os.path.isdir(labels_dir):
        # This condition will be hit if paths from config.get_sequence_paths are invalid
        # print(f"Debug in get_scan_paths: Velodyne dir '{velodyne_dir}' or Labels dir '{labels_dir}' is invalid.")
        return [], []

    bin_files = sorted(glob.glob(os.path.join(velodyne_dir, '*.bin')))
    label_files = sorted(glob.glob(os.path.join(labels_dir, '*.label')))

    valid_bin_files = []
    valid_label_files = []
    
    label_basenames = {os.path.basename(f).split('.')[0]: f for f in label_files}

    for bin_f in bin_files:
        basename = os.path.basename(bin_f).split('.')[0]
        if basename in label_basenames:
            valid_bin_files.append(bin_f)
            valid_label_files.append(label_basenames[basename])
    
    if not valid_bin_files:
        # print(f"Debug in get_scan_paths: No matching .bin and .label files found in '{velodyne_dir}' and '{labels_dir}'.")
        return [], []

    if max_scans is not None and max_scans > 0:
        # print(f"Debug: Limiting to a maximum of {max_scans} scans for this sequence.")
        return valid_bin_files[:max_scans], valid_label_files[:max_scans]
        
    return valid_bin_files, valid_label_files

def load_points_labels(bin_path, label_path):
    """
    Loads LiDAR points (x,y,z) and semantic labels from specified file paths.
    """
    try:
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        labels = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
        labels = labels & 0xFFFF
        
        if points.shape[0] != labels.shape[0]:
            print(f"Warning: Mismatch in points ({points.shape[0]}) and labels ({labels.shape[0]}) for {bin_path}. Skipping this scan pair.")
            return None, None
        if points.shape[0] == 0:
            print(f"Warning: Empty point cloud in {bin_path}. Skipping this scan pair.")
            return None, None
            
        return points, labels
    except Exception as e:
        print(f"Error loading scan {bin_path} or {label_path}: {e}")
        return None, None

def mask_cuboid(points, cuboid_center, cuboid_size):
    """
    Creates a boolean mask for points inside an axis-aligned cuboid.
    """
    lower = cuboid_center - cuboid_size / 2
    upper = cuboid_center + cuboid_size / 2
    mask = np.all((points >= lower) & (points <= upper), axis=1)
    return mask

def fit_ground_plane_and_distances(points):
    """
    Fits a ground plane using RANSAC and calculates per-point distances to it.
    Returns distances and plane model parameters [a,b,c,d].
    """
    if points.shape[0] < 3:
        return np.zeros((points.shape[0], 1)), np.array([0.0, 0.0, 1.0, 0.0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    try:
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.2,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        [a, b, c, d_plane] = plane_model
        norm = np.sqrt(a**2 + b**2 + c**2)
        if norm == 0:
            return np.zeros((points.shape[0], 1)), np.array([0.0, 0.0, 1.0, 0.0])
        
        a, b, c, d_plane = a/norm, b/norm, c/norm, d_plane/norm
        distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d_plane)
        return distances.reshape(-1, 1), np.array([a, b, c, d_plane])
    except RuntimeError:
        return np.zeros((points.shape[0], 1)), np.array([0.0, 0.0, 1.0, 0.0])

def preprocess_scan(bin_path, label_path, num_classes_config):
    """
    Processes a single scan: loads data, creates occlusion, calculates priors.
    Returns a dictionary of processed numpy arrays, or None if loading fails.
    """
    points_orig_np, labels_orig_np = load_points_labels(bin_path, label_path)

    if points_orig_np is None or labels_orig_np is None:
        return None

    if np.any(labels_orig_np >= num_classes_config):
        # print(f"Warning: Labels in {os.path.basename(label_path)} (max: {np.max(labels_orig_np)}) exceed NUM_CLASSES-1 ({num_classes_config-1}). Clamping labels.")
        labels_orig_np = np.clip(labels_orig_np, 0, num_classes_config - 1)

    # Occlusion parameters randomized per scan for diversity
    occlusion_center_np = np.mean(points_orig_np, axis=0) + np.random.uniform(-10, 10, 3)
    occlusion_size_np = np.random.uniform(10, 20, 3)
    occlusion_mask_np = mask_cuboid(points_orig_np, occlusion_center_np, occlusion_size_np)
    
    num_occluded = occlusion_mask_np.sum()
    if num_occluded == 0 and points_orig_np.shape[0] > 1000:
        occlusion_center_np_fallback = np.mean(points_orig_np, axis=0)
        size_ptp_x = np.ptp(points_orig_np[:,0]) / 3 if points_orig_np.shape[0] > 0 else 5.0
        size_ptp_y = np.ptp(points_orig_np[:,1]) / 3 if points_orig_np.shape[0] > 0 else 5.0
        size_ptp_z = np.ptp(points_orig_np[:,2]) / 2 if points_orig_np.shape[0] > 0 and np.ptp(points_orig_np[:,2]) > 1e-3 else 3.0
        occlusion_size_np_fallback = np.array([size_ptp_x, size_ptp_y, size_ptp_z])
        
        occlusion_mask_np_fallback = mask_cuboid(points_orig_np, occlusion_center_np_fallback, occlusion_size_np_fallback)
        if occlusion_mask_np_fallback.sum() > 0:
            occlusion_mask_np = occlusion_mask_np_fallback
            occlusion_center_np = occlusion_center_np_fallback
            occlusion_size_np = occlusion_size_np_fallback
        # else:
            # print(f"Debug: Fallback central occlusion also empty for {os.path.basename(bin_path)}.")


    points_for_plane_fitting = points_orig_np[~occlusion_mask_np]
    if points_for_plane_fitting.shape[0] < 100:
        points_for_plane_fitting = points_orig_np # Use all if visible part is too small

    all_plane_dist_np_fit, ground_plane_params_np = fit_ground_plane_and_distances(points_for_plane_fitting)
    
    if ground_plane_params_np is not None and not np.all(ground_plane_params_np[:3] == 0):
        a, b, c, d_plane = ground_plane_params_np
        all_plane_dist_np = (a * points_orig_np[:, 0] + b * points_orig_np[:, 1] + c * points_orig_np[:, 2] + d_plane)
        all_plane_dist_np = all_plane_dist_np.reshape(-1,1)
    else:
        all_plane_dist_np = np.zeros((points_orig_np.shape[0], 1))
        if ground_plane_params_np is None:
            ground_plane_params_np = np.array([0.0,0.0,1.0,0.0])

    labels_one_hot_np = np.eye(num_classes_config)[labels_orig_np.astype(int)]

    return {
        "points_orig": points_orig_np, "labels_orig": labels_orig_np,
        "occlusion_mask": occlusion_mask_np, "occlusion_center": occlusion_center_np, "occlusion_size": occlusion_size_np,
        "all_plane_dist": all_plane_dist_np, "ground_plane_params": ground_plane_params_np,
        "labels_one_hot": labels_one_hot_np, "scan_filepath": bin_path
    }

if __name__ == '__main__':
    print("--- Testing data_utils.py ---")
    
    # This test block now needs to use the config structure for multiple sequences
    if 'FallbackConfigForDataUtils' in globals() and isinstance(config, FallbackConfigForDataUtils) :
        print("Using fallback config for data_utils test. Actual data paths may not be resolved.")
        # Manually set some test paths if needed for isolated testing with dummy structure
        # This is tricky without the actual config.py being correctly imported.
        # For now, it will likely print warnings or find no files if config isn't proper.
    
    processed_any_scan = False
    for seq_id_test in config.SEQUENCES_TO_PROCESS:
        print(f"\n--- Testing for Sequence ID: {seq_id_test} ---")
        # Get sequence-specific paths using the helper from config (if it exists and works)
        # or construct them manually if get_sequence_paths is not part of config.
        # For this example, we assume config.get_sequence_paths is available from the latest config.py
        
        # Check if config object has get_sequence_paths, otherwise try a direct construction
        if hasattr(config, 'get_sequence_paths') and callable(config.get_sequence_paths):
             sequence_paths_test = config.get_sequence_paths(config.SEMANTIC_KITTI_BASE_DIR, seq_id_test)
        else: # Fallback construction if get_sequence_paths is not in the imported config
            print("Warning: config.get_sequence_paths not found. Attempting direct path construction for test.")
            velodyne_dir_test = os.path.join(config.SEMANTIC_KITTI_BASE_DIR, "data_odometry_velodyne", "dataset", "sequences", seq_id_test, "velodyne")
            labels_dir_test = os.path.join(config.SEMANTIC_KITTI_BASE_DIR, "data_odometry_labels", "dataset", "sequences", seq_id_test, "labels")
            sequence_paths_test = {"velodyne": velodyne_dir_test, "labels": labels_dir_test}

        if not sequence_paths_test or not os.path.isdir(sequence_paths_test.get("velodyne","")):
            print(f"Velodyne directory for sequence {seq_id_test} not found or invalid. Skipping test for this sequence.")
            print(f"  Tried path: {sequence_paths_test.get('velodyne','') if sequence_paths_test else 'N/A'}")
            continue

        bin_paths, label_paths = get_scan_paths(
            sequence_paths_test["velodyne"], 
            sequence_paths_test["labels"], 
            max_scans=config.MAX_SCANS_PER_SEQUENCE
        )
        print(f"Found {len(bin_paths)} scan pairs to test in sequence {seq_id_test}.")

        if not bin_paths:
            continue

        for i, (bp, lp) in enumerate(zip(bin_paths, label_paths)):
            print(f"\n--- Processing test scan {i+1} from sequence {seq_id_test}: {os.path.basename(bp)} ---")
            # NUM_CLASSES needs to be correctly sourced from config
            num_classes_for_test = config.NUM_CLASSES if hasattr(config, 'NUM_CLASSES') else 20
            processed_data = preprocess_scan(bp, lp, num_classes_for_test)
            if processed_data:
                processed_any_scan = True
                print(f"Scan: {processed_data['scan_filepath']}")
                print(f"  Original points shape: {processed_data['points_orig'].shape}")
                print(f"  Occlusion mask sum: {processed_data['occlusion_mask'].sum()}")
                print(f"  Plane distances shape: {processed_data['all_plane_dist'].shape}")
                print(f"  Labels one-hot shape: {processed_data['labels_one_hot'].shape}")
            else:
                print(f"Failed to process scan: {bp}")
            if i >= 0 : # Process only one scan per sequence for this test to keep it short
                break 
        
    if not processed_any_scan:
         print("\nNo scans were processed during the test. Check config.py paths and dataset structure.")
    print("--- Finished testing data_utils.py ---")