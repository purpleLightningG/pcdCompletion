# training_utils.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
# import matplotlib.pyplot as plt # No longer needed here for plotting per scan
import os

try:
    import config
    from model_components import ContextEncoder, DenoisingNetwork, DiffusionScheduler
except ImportError:
    # ... (fallback config as before) ...
    print("Warning: Could not import from config or model_components in training_utils.py.")
    class FallbackConfig:
        DEVICE = torch.device("cpu"); LEARNING_RATE = 1e-4
        NUM_EPOCHS_PER_SCAN_CONCEPTUAL = 10; BATCH_SIZE_OCC_TRAINING = 32
        NUM_TIMESTEPS = 1000
    config = FallbackConfig()
    if 'ContextEncoder' not in globals(): ContextEncoder = type('ContextEncoder', (torch.nn.Module,), {})
    if 'DenoisingNetwork' not in globals(): DenoisingNetwork = type('DenoisingNetwork', (torch.nn.Module,), {})
    if 'DiffusionScheduler' not in globals(): DiffusionScheduler = type('DiffusionScheduler', (object,), {})


def train_single_scan_epoch(context_encoder, denoising_net, scheduler, optimizer,
                            points_vis_torch, plane_dist_vis_torch, labels_vis_one_hot_torch,
                            points_occ_gt_torch, plane_dist_occ_gt_torch, labels_occ_gt_one_hot_torch,
                            epoch_num, total_epochs):
    context_encoder.train()
    denoising_net.train()
    optimizer.zero_grad()
    
    if points_vis_torch.shape[0] == 0: return 0.0
    global_context_feat = context_encoder(points_vis_torch, plane_dist_vis_torch, labels_vis_one_hot_torch)
    if points_occ_gt_torch.shape[0] == 0: return 0.0

    num_occluded_gt = points_occ_gt_torch.shape[0]
    current_batch_size_occ = min(config.BATCH_SIZE_OCC_TRAINING, num_occluded_gt)
    
    if num_occluded_gt > current_batch_size_occ:
        perm = torch.randperm(num_occluded_gt, device=config.DEVICE)
        idx = perm[:current_batch_size_occ]
        current_points_occ_gt = points_occ_gt_torch[idx]
        current_labels_occ_gt = labels_occ_gt_one_hot_torch[idx]
        current_plane_dist_occ_gt = plane_dist_occ_gt_torch[idx]
    else:
        current_points_occ_gt = points_occ_gt_torch
        current_labels_occ_gt = labels_occ_gt_one_hot_torch
        current_plane_dist_occ_gt = plane_dist_occ_gt_torch

    t_indices = torch.randint(0, scheduler.num_timesteps, (current_points_occ_gt.shape[0],), device=config.DEVICE)
    t_norm_for_net = t_indices.float().mean() / scheduler.num_timesteps 
    noisy_points_occ, noise_gt = scheduler.add_noise(current_points_occ_gt, t_indices)
    predicted_noise = denoising_net(noisy_points_occ, t_norm_for_net, global_context_feat,
                                    current_labels_occ_gt, current_plane_dist_occ_gt)
    
    if predicted_noise.shape[0] == 0: return 0.0
    loss = F.mse_loss(predicted_noise, noise_gt)
    loss.backward()
    optimizer.step()
    return loss.item()

def run_training_for_scan(scan_data, context_encoder, denoising_net, scheduler, device=config.DEVICE, scan_filename="unknown_scan"):
    print(f"\n--- Starting Training for Scan: {os.path.basename(scan_filename)} ---")

    points_orig_torch = torch.from_numpy(scan_data['points_orig']).float().to(device)
    occlusion_mask_torch = torch.from_numpy(scan_data['occlusion_mask']).bool().to(device)
    all_plane_dist_torch = torch.from_numpy(scan_data['all_plane_dist']).float().to(device)
    labels_one_hot_torch = torch.from_numpy(scan_data['labels_one_hot']).float().to(device)

    points_vis_torch = points_orig_torch[~occlusion_mask_torch]
    plane_dist_vis_torch = all_plane_dist_torch[~occlusion_mask_torch]
    labels_vis_one_hot_torch = labels_one_hot_torch[~occlusion_mask_torch]

    points_occ_gt_torch = points_orig_torch[occlusion_mask_torch]
    labels_occ_gt_one_hot_torch = labels_one_hot_torch[occlusion_mask_torch]
    plane_dist_occ_gt_torch = all_plane_dist_torch[occlusion_mask_torch]
    
    if points_occ_gt_torch.shape[0] == 0:
        print(f"No occluded points in {os.path.basename(scan_filename)}. Skipping training for this scan.")
        return [], context_encoder, denoising_net # Return empty losses
    if points_vis_torch.shape[0] == 0:
        print(f"No visible points in {os.path.basename(scan_filename)}. Skipping training for this scan (no context).")
        return [], context_encoder, denoising_net

    params = list(context_encoder.parameters()) + list(denoising_net.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.LEARNING_RATE)

    losses_for_scan = []
    
    progress_bar = tqdm(range(config.NUM_EPOCHS_PER_SCAN_CONCEPTUAL), desc=f"Scan {os.path.basename(scan_filename)} Training")
    for epoch in progress_bar:
        loss_item = train_single_scan_epoch(
            context_encoder, denoising_net, scheduler, optimizer,
            points_vis_torch, plane_dist_vis_torch, labels_vis_one_hot_torch,
            points_occ_gt_torch, plane_dist_occ_gt_torch, labels_occ_gt_one_hot_torch,
            epoch, config.NUM_EPOCHS_PER_SCAN_CONCEPTUAL
        )
        losses_for_scan.append(loss_item)
        if (epoch + 1) % (max(1, config.NUM_EPOCHS_PER_SCAN_CONCEPTUAL // 10)) == 0:
             progress_bar.set_postfix({"Loss": f"{loss_item:.6f}"})
    
    print(f"--- Finished Training for Scan: {os.path.basename(scan_filename)} ---")
    
    # REMOVED PLOTTING LOGIC FROM HERE
    # It will be done once in main_pipeline.py

    return losses_for_scan, context_encoder, denoising_net


if __name__ == '__main__':
    # ... (The test block can remain, but it won't produce a plot directly from here anymore)
    # ... (You might want to print the returned losses_for_scan list in the test block)
    print("--- Testing training_utils.py ---")
    # (Ensure imports and setup for the test block are correct as per previous versions)
    try:
        import config
        from data_utils import get_scan_paths, preprocess_scan
        # Models already imported
    except ImportError:
        print("Error: Cannot run test. Missing config, data_utils, or model_components.")
        exit()

    if not config.SEQUENCES_TO_PROCESS:
        print("No sequences in config.SEQUENCES_TO_PROCESS. Test cannot run.")
        exit()
    
    first_seq_id = config.SEQUENCES_TO_PROCESS[0]
    if hasattr(config, 'get_sequence_paths'):
        seq_paths = config.get_sequence_paths(config.SEMANTIC_KITTI_BASE_DIR, first_seq_id)
    else: # Fallback
        velodyne_dir_test_seq = os.path.join(config.SEMANTIC_KITTI_BASE_DIR, "data_odometry_velodyne", "dataset", "sequences", first_seq_id, "velodyne")
        labels_dir_test_seq = os.path.join(config.SEMANTIC_KITTI_BASE_DIR, "data_odometry_labels", "dataset", "sequences", first_seq_id, "labels")
        seq_paths = {"velodyne": velodyne_dir_test_seq, "labels": labels_dir_test_seq}


    if not seq_paths or not os.path.isdir(seq_paths.get("velodyne","")):
        print(f"Velodyne dir for test seq {first_seq_id} not found. Test cannot run.")
        exit()

    bin_p_test, lbl_p_test = get_scan_paths(seq_paths["velodyne"], seq_paths["labels"], max_scans=1)
    if not bin_p_test:
        print(f"No scans found for test in seq {first_seq_id}.")
        exit()
    
    scan_data_t = preprocess_scan(bin_p_test[0], lbl_p_test[0], config.NUM_CLASSES)
    if not scan_data_t:
        print("Failed to preprocess test scan data.")
        exit()

    ctx_enc_t = ContextEncoder().to(config.DEVICE)
    den_net_t = DenoisingNetwork().to(config.DEVICE)
    sched_t = DiffusionScheduler(device=config.DEVICE)

    scan_losses_t, _, _ = run_training_for_scan(
        scan_data_t, ctx_enc_t, den_net_t, sched_t,
        device=config.DEVICE, scan_filename=bin_p_test[0]
    )
    if scan_losses_t:
        print(f"Test training for one scan completed. Final loss: {scan_losses_t[-1]:.6f}")
        print(f"All losses for the scan: {scan_losses_t}")
    else:
        print("Test training did not produce losses for the scan.")
    print("--- Finished testing training_utils.py (plotting per scan removed) ---")