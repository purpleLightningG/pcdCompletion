# train_full_scale.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    import config
    from model_components import ContextEncoder, DenoisingNetwork, DiffusionScheduler
    from kitti_completion_dataset import KittiCompletionDataset
except ImportError as e:
    print(f"CRITICAL Error importing necessary modules: {e}")
    exit()

# --- (Helper functions like pairwise_distance_sq, chamfer_distance_loss_chunked, etc. remain the same) ---
def pairwise_distance_sq(x, y):
    x_norm = (x**2).sum(1).view(-1, 1); y_norm = (y**2).sum(1).view(1, -1)
    return torch.clamp(x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1)), min=0.0)

def chamfer_distance_loss_chunked(p1, p2, chunk_size=1024):
    if p1.shape[0] == 0 or p2.shape[0] == 0: return torch.tensor(1e9 if p1.shape[0] != p2.shape[0] else 0.0, device=p1.device)
    dist_p1_p2 = 0.0
    for chunk in torch.split(p1, chunk_size):
        min_dist, _ = torch.min(pairwise_distance_sq(chunk, p2), dim=1); dist_p1_p2 += torch.sum(min_dist)
    loss_p1_p2 = dist_p1_p2 / p1.shape[0]
    dist_p2_p1 = 0.0
    for chunk in torch.split(p2, chunk_size):
        min_dist, _ = torch.min(pairwise_distance_sq(chunk, p1), dim=1); dist_p2_p1 += torch.sum(min_dist)
    loss_p2_p1 = dist_p2_p1 / p2.shape[0]
    return loss_p1_p2 + loss_p2_p1

def kitti_completion_collate_fn(batch_list):
    # ... (function as before)
    collated = {}; 
    if not batch_list: return collated
    keys = batch_list[0].keys()
    for key in keys:
        if key in ['visible_points', 'visible_labels_one_hot', 'visible_plane_dist', 'occluded_gt_points', 'occluded_gt_labels_one_hot', 'occluded_gt_plane_dist', 'original_full_points']:
            collated[key] = [sample[key] for sample in batch_list]
        elif key == 'metadata': collated[key] = [sample[key] for sample in batch_list]
        else:
            try: collated[key] = torch.utils.data.default_collate([sample[key] for sample in batch_list])
            except: collated[key] = [sample[key] for sample in batch_list]
    return collated

def save_checkpoint(epoch, model_ce, model_dn, optimizer, filename):
    state = {'epoch': epoch, 'context_encoder_state_dict': model_ce.state_dict(),
             'denoising_net_state_dict': model_dn.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    torch.save(state, os.path.join(config.CHECKPOINT_DIR, filename))
    print(f"Checkpoint saved to {os.path.join(config.CHECKPOINT_DIR, filename)}")


# --- UPDATED BATCH LOSS CALCULATION with Parametric Consistency Loss ---
def calculate_full_hybrid_loss(batch_data, context_encoder, denoising_net, scheduler, device):
    """
    Calculates the FULL HYBRID loss for a single batch.
    Loss = L_noise + lambda_cd * L_cd + lambda_parametric * L_parametric
    """
    # ... (unpacking lists of tensors as before) ...
    visible_points_list = batch_data['visible_points']
    occluded_gt_points_list = batch_data['occluded_gt_points']
    visible_labels_one_hot_list = batch_data['visible_labels_one_hot']
    visible_plane_dist_list = batch_data['visible_plane_dist']
    occluded_gt_labels_one_hot_list = batch_data['occluded_gt_labels_one_hot']
    occluded_gt_plane_dist_list = batch_data['occluded_gt_plane_dist']

    batch_size = len(visible_points_list)
    if batch_size == 0: return torch.tensor(0.0, device=device, requires_grad=True)

    total_loss_for_batch = torch.tensor(0.0, device=device)
    valid_samples_in_batch = 0

    for i in range(batch_size):
        sample_vis_pts = visible_points_list[i].to(device); sample_occ_gt_pts = occluded_gt_points_list[i].to(device)
        sample_vis_lbl = visible_labels_one_hot_list[i].to(device); sample_vis_pd = visible_plane_dist_list[i].to(device)
        sample_occ_gt_lbl = occluded_gt_labels_one_hot_list[i].to(device); sample_occ_gt_pd = occluded_gt_plane_dist_list[i].to(device)

        if sample_vis_pts.shape[0] == 0 or sample_occ_gt_pts.shape[0] == 0: continue

        global_context_feat = context_encoder(sample_vis_pts, sample_vis_pd, sample_vis_lbl)
        t_indices = torch.randint(0, scheduler.num_timesteps, (sample_occ_gt_pts.shape[0],), device=device)
        t_norm = t_indices.float().mean() / scheduler.num_timesteps
        noisy_points_occ, noise_gt = scheduler.add_noise(sample_occ_gt_pts, t_indices)
        
        # Denoising network now returns TWO outputs
        predicted_noise, predicted_plane_params = denoising_net(noisy_points_occ, t_norm, global_context_feat, sample_occ_gt_lbl, sample_occ_gt_pd)
        
        if predicted_noise.shape[0] == 0: continue
            
        # --- CALCULATE HYBRID LOSS ---
        # 1. Noise Prediction Loss (L_noise)
        loss_noise = torch.nn.functional.mse_loss(predicted_noise, noise_gt)
        
        # 2. Geometric Reconstruction Loss (L_cd)
        predicted_clean_points = scheduler.predict_x0_from_noise(noisy_points_occ, t_indices, predicted_noise)
        num_cd_pts = config.CD_LOSS_NUM_POINTS
        if predicted_clean_points.shape[0] > num_cd_pts: pred_points_subset = predicted_clean_points[torch.randperm(predicted_clean_points.shape[0])[:num_cd_pts]]
        else: pred_points_subset = predicted_clean_points
        if sample_occ_gt_pts.shape[0] > num_cd_pts: gt_points_subset = sample_occ_gt_pts[torch.randperm(sample_occ_gt_pts.shape[0])[:num_cd_pts]]
        else: gt_points_subset = sample_occ_gt_pts
        loss_cd = chamfer_distance_loss_chunked(pred_points_subset, gt_points_subset)
        
        # --- NEW: 3. Parametric Consistency Loss (L_parametric) ---
        loss_parametric = torch.tensor(0.0, device=device)
        # Check if the dominant semantic class for this sample is one of the planar classes
        dominant_class_idx = torch.argmax(torch.mean(sample_occ_gt_lbl, dim=0))
        if dominant_class_idx.item() in config.PLANAR_CLASS_IDS:
            # Get the predicted plane: a,b,c are the normal (n), d is the offset
            n = predicted_plane_params[:, :3] # Shape: [1, 3]
            d = predicted_plane_params[:, 3]  # Shape: [1]
            
            # Calculate the signed distance of each predicted point to the predicted plane
            # dist = (a*x + b*y + c*z + d). For a unit normal n, this is dot(p,n) + d
            # We use absolute distance for the loss.
            point_plane_dist = torch.abs(torch.matmul(predicted_clean_points, n.T).squeeze() + d)
            loss_parametric = torch.mean(point_plane_dist)

        # 4. Combine all losses
        total_loss_this_sample = loss_noise + config.LAMBDA_CD_LOSS * loss_cd + config.LAMBDA_PARAMETRIC_LOSS * loss_parametric
        
        total_loss_for_batch += total_loss_this_sample
        valid_samples_in_batch += 1
    
    if valid_samples_in_batch == 0: return torch.tensor(0.0, device=device, requires_grad=True)
    return total_loss_for_batch / valid_samples_in_batch


def main_train():
    # ... (Setup logic as before) ...
    print("--- Starting Full-Scale Training Pipeline (with Parametric Loss) ---")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True); os.makedirs(config.TRAINING_PLOTS_DIR, exist_ok=True)
    
    context_encoder = ContextEncoder().to(config.DEVICE)
    denoising_net = DenoisingNetwork().to(config.DEVICE)
    diffusion_scheduler = DiffusionScheduler(device=config.DEVICE)
    optimizer = optim.AdamW(list(context_encoder.parameters()) + list(denoising_net.parameters()), lr=config.LEARNING_RATE)
    start_epoch = 0

    try:
        train_dataset = KittiCompletionDataset(config.PREPROCESSED_DATA_DIR, 'train')
        val_dataset = KittiCompletionDataset(config.PREPROCESSED_DATA_DIR, 'val')
    except FileNotFoundError as e: print(f"Error loading dataset: {e}."); return
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.DATALOADER_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS_DATALOADER, collate_fn=kitti_completion_collate_fn, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.DATALOADER_BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS_DATALOADER, collate_fn=kitti_completion_collate_fn) if len(val_dataset) > 0 else None
    
    print(f"\nStarting training from epoch {start_epoch + 1} for {config.NUM_GLOBAL_EPOCHS} epochs...")
    epoch_train_losses, epoch_val_losses = [], []

    for epoch in range(start_epoch, config.NUM_GLOBAL_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.NUM_GLOBAL_EPOCHS} ---")
        context_encoder.train(); denoising_net.train()
        running_train_loss = 0.0
        
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training")
        for batch_data in train_progress_bar:
            optimizer.zero_grad()
            # Call the new loss function
            loss = calculate_full_hybrid_loss(batch_data, context_encoder, denoising_net, diffusion_scheduler, config.DEVICE)
            if loss.requires_grad: loss.backward(); optimizer.step()
            running_train_loss += loss.item()
            train_progress_bar.set_postfix({"Full Hybrid Loss": f"{loss.item():.4f}"})

        avg_train_loss = running_train_loss / len(train_dataloader)
        epoch_train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

        # Validation loop (also uses new loss function)
        if val_dataloader and (epoch + 1) % config.VALIDATE_EVERY_N_EPOCHS == 0:
            context_encoder.eval(); denoising_net.eval()
            running_val_loss = 0.0
            val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation")
            with torch.no_grad():
                for batch_data_val in val_progress_bar:
                    val_loss = calculate_full_hybrid_loss(batch_data_val, context_encoder, denoising_net, diffusion_scheduler, config.DEVICE)
                    running_val_loss += val_loss.item()
                    val_progress_bar.set_postfix({"Val Hybrid Loss": f"{val_loss.item():.4f}"})
            avg_val_loss = running_val_loss / len(val_dataloader)
            epoch_val_losses.append(avg_val_loss)
            print(f"Epoch {epoch + 1} - Average Validation Loss: {avg_val_loss:.4f}")
        else:
            epoch_val_losses.append(None)

        if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0 or (epoch + 1) == config.NUM_GLOBAL_EPOCHS:
            save_checkpoint(epoch, context_encoder, denoising_net, optimizer, f"checkpoint_epoch_{epoch+1}.pth.tar")
            save_checkpoint(epoch, context_encoder, denoising_net, optimizer, "latest_checkpoint.pth.tar")

    print("\n--- Full-Scale Training Finished ---")
    if epoch_train_losses:
        # ... (Final plotting logic as before, just update axis labels if desired)
        pass

if __name__ == "__main__":
    np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    main_train()
