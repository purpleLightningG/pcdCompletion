# preprocess_dataset.py

import os
import torch
import numpy as np
from tqdm import tqdm

try:
    import config
    from data_utils import get_scan_paths, preprocess_scan
except ImportError as e:
    print(f"CRITICAL Error importing necessary modules: {e}")
    exit()

def create_preprocessed_samples(base_dir, sequences, output_parent_dir, split_name):
    # ... (function setup and loops as before) ...
    output_dir = os.path.join(output_parent_dir, split_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory for {split_name} split: {output_dir}")

    total_samples_saved_this_run = 0
    total_samples_skipped = 0

    for seq_id in sequences:
        # ... (looping through sequences and scans as before) ...
        print(f"\nProcessing sequence: {seq_id} for {split_name} split...")
        if hasattr(config, 'get_sequence_paths') and callable(config.get_sequence_paths):
            sequence_paths = config.get_sequence_paths(base_dir, seq_id)
        else:
             velodyne_dir_sq = os.path.join(base_dir, "data_odometry_velodyne", "dataset", "sequences", seq_id, "velodyne")
             labels_dir_sq = os.path.join(base_dir, "data_odometry_labels", "dataset", "sequences", seq_id, "labels")
             sequence_paths = {"velodyne": velodyne_dir_sq, "labels": labels_dir_sq}
        
        if not sequence_paths or not os.path.isdir(sequence_paths.get("velodyne","")):
            continue

        bin_paths, label_paths = get_scan_paths(
            sequence_paths["velodyne"],
            sequence_paths["labels"],
            max_scans=config.MAX_SCANS_TO_PREPROCESS_PER_SEQUENCE
        )
        if not bin_paths:
            continue
        print(f"  Found {len(bin_paths)} scans to preprocess in sequence {seq_id}.")

        for scan_idx, (bin_path, label_path) in enumerate(tqdm(zip(bin_paths, label_paths), total=len(bin_paths), desc=f"Seq {seq_id} scans")):
            scan_filename_base = os.path.basename(bin_path).split('.')[0]

            for occ_instance_idx in range(config.NUM_OCCLUSIONS_PER_SCAN):
                sample_filename = f"seq{seq_id}_scan{scan_filename_base}_occ{occ_instance_idx}.pt"
                sample_save_path = os.path.join(output_dir, sample_filename)

                if os.path.exists(sample_save_path):
                    total_samples_skipped += 1
                    continue 
                
                processed_data_dict = preprocess_scan(bin_path, label_path, config.NUM_CLASSES)
                if processed_data_dict is None: continue

                points_orig_np = processed_data_dict["points_orig"]
                occlusion_mask = processed_data_dict["occlusion_mask"]
                all_plane_dist = processed_data_dict["all_plane_dist"]
                labels_one_hot = processed_data_dict["labels_one_hot"]

                visible_points = points_orig_np[~occlusion_mask]
                visible_labels_one_hot = labels_one_hot[~occlusion_mask]
                visible_plane_dist = all_plane_dist[~occlusion_mask]

                occluded_gt_points = points_orig_np[occlusion_mask]
                occluded_gt_labels_one_hot = labels_one_hot[occlusion_mask]
                occluded_gt_plane_dist = all_plane_dist[occlusion_mask]

                if occluded_gt_points.shape[0] == 0 or visible_points.shape[0] == 0:
                    continue
                
                #
                # Place the new line inside this dictionary definition:
                #
                sample_to_save = {
                    'visible_points': torch.from_numpy(visible_points).float(),
                    'visible_labels_one_hot': torch.from_numpy(visible_labels_one_hot).float(),
                    'visible_plane_dist': torch.from_numpy(visible_plane_dist).float(),
                    'occluded_gt_points': torch.from_numpy(occluded_gt_points).float(),
                    'occluded_gt_labels_one_hot': torch.from_numpy(occluded_gt_labels_one_hot).float(),
                    'occluded_gt_plane_dist': torch.from_numpy(occluded_gt_plane_dist).float(),
                    
                    # VVVVVV  ADD THE LINE HERE  VVVVVV
                    'original_full_points': torch.from_numpy(points_orig_np).float(),
                    # ^^^^^^  ADD THE LINE HERE  ^^^^^^

                    'metadata': { 
                        'original_scan_path': bin_path,
                        'sequence_id': seq_id,
                        'scan_id': scan_filename_base,
                        'occlusion_instance': occ_instance_idx,
                    }
                }
                
                try:
                    torch.save(sample_to_save, sample_save_path)
                    total_samples_saved_this_run += 1
                except Exception as e:
                    print(f"    Error saving sample {sample_save_path}: {e}")
            
    print(f"\nFinished preprocessing for {split_name} split.")
    # ... (summary print statements as before) ...


def main():
    # ... (main function as before) ...
    print("--- Starting Offline Dataset Preprocessing ---")
    os.makedirs(config.PREPROCESSED_DATA_DIR, exist_ok=True)
    if config.TRAIN_SEQUENCES:
        print("\n--- Preprocessing Training Data ---")
        create_preprocessed_samples(config.SEMANTIC_KITTI_BASE_DIR, config.TRAIN_SEQUENCES, config.PREPROCESSED_DATA_DIR, "train")
    if config.VAL_SEQUENCES:
        print("\n--- Preprocessing Validation Data ---")
        create_preprocessed_samples(config.SEMANTIC_KITTI_BASE_DIR, config.VAL_SEQUENCES, config.PREPROCESSED_DATA_DIR, "val")
    print("\n--- Offline Dataset Preprocessing Finished ---")

if __name__ == "__main__":
    np.random.seed(42) 
    main()
