# main_pipeline.py

import torch
import numpy as np
import os
import open3d as o3d # For saving point clouds
import matplotlib.pyplot as plt # For the final loss plot

try:
    import config
    from data_utils import get_scan_paths, preprocess_scan
    from model_components import ContextEncoder, DenoisingNetwork, DiffusionScheduler
    from training_utils import run_training_for_scan
    from inference_utils import sample_completion
    from visualization_utils import visualize_scan_completion
except ImportError as e:
    print(f"CRITICAL Error importing necessary modules: {e}")
    exit()

def main():
    print("--- Starting Point Cloud Completion Pipeline ---")
    print(f"Using device: {config.DEVICE}")
    # ... (other initial prints from your main_pipeline.py)

    # Create output directories if they don't exist
    os.makedirs(config.COMPLETED_SCANS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.TRAINING_PLOTS_DIR, exist_ok=True) # For the final plot

    print("\nInstantiating models...")
    context_encoder = ContextEncoder().to(config.DEVICE) # Ensure correct init from model_components
    denoising_net = DenoisingNetwork().to(config.DEVICE) # Ensure correct init
    scheduler = DiffusionScheduler(device=config.DEVICE) # Ensure correct init
    print("Models and scheduler instantiated.")

    all_scans_final_epoch_losses = [] # To collect loss from the last epoch of each scan's training
    processed_scan_counter = 0 # For x-axis of the final plot

    last_scan_data_for_visualization = None
    last_generated_points_for_visualization = None
    last_semantic_guidance_for_visualization = None

    for seq_id in config.SEQUENCES_TO_PROCESS:
        print(f"\n===== Processing Sequence: {seq_id} =====")
        if hasattr(config, 'get_sequence_paths') and callable(config.get_sequence_paths):
            sequence_paths = config.get_sequence_paths(config.SEMANTIC_KITTI_BASE_DIR, seq_id)
        else:
             velodyne_dir_sq = os.path.join(config.SEMANTIC_KITTI_BASE_DIR, "data_odometry_velodyne", "dataset", "sequences", seq_id, "velodyne")
             labels_dir_sq = os.path.join(config.SEMANTIC_KITTI_BASE_DIR, "data_odometry_labels", "dataset", "sequences", seq_id, "labels")
             sequence_paths = {"velodyne": velodyne_dir_sq, "labels": labels_dir_sq}

        if not sequence_paths or not os.path.isdir(sequence_paths.get("velodyne","")) or not os.path.isdir(sequence_paths.get("labels","")):
            print(f"Could not resolve paths for sequence {seq_id}. Skipping.")
            continue

        bin_paths, label_paths = get_scan_paths(
            sequence_paths["velodyne"], sequence_paths["labels"],
            max_scans=config.MAX_SCANS_PER_SEQUENCE
        )
        if not bin_paths:
            print(f"No scan files for sequence {seq_id}. Skipping.")
            continue
        
        print(f"Found {len(bin_paths)} scans to process in sequence {seq_id}.")

        for i, (bin_path, label_path) in enumerate(zip(bin_paths, label_paths)):
            scan_filename_base = os.path.basename(bin_path).split('.')[0] # e.g., "000000"
            scan_filename_full = os.path.basename(bin_path)
            print(f"\n--- Processing Scan {i+1}/{len(bin_paths)} (Seq {seq_id}): {scan_filename_full} ---")

            scan_data = preprocess_scan(bin_path, label_path, config.NUM_CLASSES)
            if not scan_data:
                print(f"Failed to preprocess {scan_filename_full}. Skipping.")
                continue
            
            points_orig_torch = torch.from_numpy(scan_data['points_orig']).float().to(config.DEVICE)
            occlusion_mask_torch = torch.from_numpy(scan_data['occlusion_mask']).bool().to(config.DEVICE)
            all_plane_dist_torch = torch.from_numpy(scan_data['all_plane_dist']).float().to(config.DEVICE)
            labels_one_hot_torch = torch.from_numpy(scan_data['labels_one_hot']).float().to(config.DEVICE)

            points_vis_torch = points_orig_torch[~occlusion_mask_torch]
            plane_dist_vis_torch = all_plane_dist_torch[~occlusion_mask_torch]
            labels_vis_one_hot_torch = labels_one_hot_torch[~occlusion_mask_torch]
            
            points_occ_gt_torch = points_orig_torch[occlusion_mask_torch]
            labels_occ_gt_one_hot_torch = labels_one_hot_torch[occlusion_mask_torch]
            plane_dist_occ_gt_torch = all_plane_dist_torch[occlusion_mask_torch]

            current_scan_had_occlusion = points_occ_gt_torch.shape[0] > 0
            generated_occluded_points_np = np.empty((0, config.POINT_DIM)) # Default if no occlusion/inference
            final_completed_cloud_np = scan_data['points_orig'] # Default to original if no completion

            if current_scan_had_occlusion:
                losses_this_scan, context_encoder, denoising_net = run_training_for_scan(
                    scan_data, context_encoder, denoising_net, scheduler, 
                    device=config.DEVICE, scan_filename=scan_filename_full
                )
                if losses_this_scan: # Check if training actually ran and produced losses
                    all_scans_final_epoch_losses.append(losses_this_scan[-1])
                    processed_scan_counter += 1

                num_points_to_generate = points_occ_gt_torch.shape[0]
                inference_semantic_guidance = labels_occ_gt_one_hot_torch 
                inference_plane_dist_guidance = plane_dist_occ_gt_torch
                
                # Adjust guidance (defensive coding from previous)
                if inference_semantic_guidance.shape[0] != num_points_to_generate:
                    inference_semantic_guidance = (inference_semantic_guidance[:num_points_to_generate] if inference_semantic_guidance.shape[0] > num_points_to_generate else
                                                   torch.cat([inference_semantic_guidance, inference_semantic_guidance[-1:].repeat(num_points_to_generate - inference_semantic_guidance.shape[0], 1)], dim=0) if inference_semantic_guidance.shape[0] > 0 and inference_semantic_guidance.shape[0] < num_points_to_generate else
                                                   torch.zeros(num_points_to_generate, config.NUM_CLASSES, device=config.DEVICE))
                if inference_plane_dist_guidance.shape[0] != num_points_to_generate:
                     inference_plane_dist_guidance = (inference_plane_dist_guidance[:num_points_to_generate] if inference_plane_dist_guidance.shape[0] > num_points_to_generate else
                                                   torch.cat([inference_plane_dist_guidance, inference_plane_dist_guidance[-1:].repeat(num_points_to_generate - inference_plane_dist_guidance.shape[0], 1)], dim=0) if inference_plane_dist_guidance.shape[0] > 0 and inference_plane_dist_guidance.shape[0] < num_points_to_generate else
                                                   torch.zeros(num_points_to_generate, config.PLANE_DIST_DIM, device=config.DEVICE))


                generated_occluded_points_np = sample_completion(
                    context_encoder, denoising_net, scheduler,
                    points_vis_torch, plane_dist_vis_torch, labels_vis_one_hot_torch,
                    num_points_to_generate,
                    inference_semantic_guidance, inference_plane_dist_guidance,
                    device=config.DEVICE
                )
                
                # Combine visible with generated for saving
                final_completed_cloud_np = np.vstack((points_vis_torch.cpu().numpy(), generated_occluded_points_np))

                last_scan_data_for_visualization = scan_data
                last_generated_points_for_visualization = generated_occluded_points_np
                last_semantic_guidance_for_visualization = inference_semantic_guidance.cpu().numpy()
            else:
                print(f"No occluded points in {scan_filename_full}. Skipping training/inference.")
                last_scan_data_for_visualization = scan_data
                last_generated_points_for_visualization = generated_occluded_points_np # empty
                last_semantic_guidance_for_visualization = None

            # Save the completed scan
            if final_completed_cloud_np.shape[0] > 0 : # Only save if there's something to save
                pcd_to_save = o3d.geometry.PointCloud()
                pcd_to_save.points = o3d.utility.Vector3dVector(final_completed_cloud_np)
                # Optionally, add colors before saving if you have them for the final_completed_cloud_np
                # For example, if you construct final_colors_np as in visualization_utils.py
                # pcd_to_save.colors = o3d.utility.Vector3dVector(final_colors_np)
                
                save_filename = f"seq{seq_id}_{scan_filename_base}_completed.pcd"
                save_path = os.path.join(config.COMPLETED_SCANS_OUTPUT_DIR, save_filename)
                o3d.io.write_point_cloud(save_path, pcd_to_save)
                print(f"Saved completed scan to: {save_path}")


            if config.VISUALIZE_PER_SCAN_DURING_PIPELINE:
                visualize_scan_completion(
                    scan_data, generated_occluded_points_np,
                    target_semantic_guidance_occ_for_coloring=(inference_semantic_guidance.cpu().numpy() if current_scan_had_occlusion and inference_semantic_guidance is not None else None),
                    show_window=True
                )
    
    print("\n--- Point Cloud Completion Pipeline Finished All Scans ---")

    # Plot overall losses
    if all_scans_final_epoch_losses:
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, processed_scan_counter + 1), all_scans_final_epoch_losses, marker='o', linestyle='-')
        plt.title("Final Epoch Loss per Processed Scan")
        plt.xlabel("Processed Scan Index")
        plt.ylabel("MSE Loss (Final Epoch of Scan Training)")
        plt.xticks(range(1, processed_scan_counter + 1))
        plt.grid(True)
        final_plot_path = os.path.join(config.TRAINING_PLOTS_DIR, "overall_scan_training_losses.png")
        plt.savefig(final_plot_path)
        print(f"Overall training loss plot saved to: {final_plot_path}")
        # plt.show() # Optionally show
        plt.close()

    if config.VISUALIZE_LAST_SCAN_POST_PIPELINE and last_scan_data_for_visualization is not None:
        print("\nVisualizing results for the LAST processed scan...")
        visualize_scan_completion(
            last_scan_data_for_visualization,
            last_generated_points_for_visualization,
            target_semantic_guidance_occ_for_coloring=last_semantic_guidance_for_visualization,
            show_window=True
        )
    elif config.VISUALIZE_LAST_SCAN_POST_PIPELINE:
        print("Post-pipeline visualization enabled, but no scan data was processed/stored for it.")

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()