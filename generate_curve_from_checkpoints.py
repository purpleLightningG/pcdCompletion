# generate_curve_from_checkpoints.py (Validation Loss Only Version)

import torch
from torch.utils.data import DataLoader
import os
import glob
import re
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    import config
    from model_components import ContextEncoder, DenoisingNetwork, DiffusionScheduler
    from kitti_completion_dataset import KittiCompletionDataset
except ImportError as e:
    print(f"CRITICAL Error importing necessary modules: {e}")
    print("Please ensure all project python files are in the Python path.")
    exit()

def kitti_eval_collate_fn(batch_list):
    """Custom collate function for evaluation."""
    collated = {}; 
    if not batch_list: return collated
    keys = batch_list[0].keys()
    for key in keys:
        if key in ['visible_points', 'visible_labels_one_hot', 'visible_plane_dist',
                   'occluded_gt_points', 'occluded_gt_labels_one_hot', 'occluded_gt_plane_dist',
                   'original_full_points']:
            collated[key] = [sample[key] for sample in batch_list]
        elif key == 'metadata': collated[key] = [sample[key] for sample in batch_list]
        else:
            try: collated[key] = torch.utils.data.default_collate([sample[key] for sample in batch_list])
            except: collated[key] = [sample[key] for sample in batch_list]
    return collated

def get_validation_loss_for_checkpoint(checkpoint_path, val_dataloader, device):
    """
    Loads a model checkpoint and computes the average validation loss.
    """
    print(f"\nEvaluating checkpoint: {os.path.basename(checkpoint_path)}")

    # 1. Instantiate fresh models for this evaluation
    context_encoder = ContextEncoder().to(device)
    denoising_net = DenoisingNetwork().to(device)
    scheduler = DiffusionScheduler(device=device)

    # 2. Load the state dicts from the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        context_encoder.load_state_dict(checkpoint['context_encoder_state_dict'])
        denoising_net.load_state_dict(checkpoint['denoising_net_state_dict'])
    except Exception as e:
        print(f"  Error loading checkpoint: {e}. Skipping this checkpoint.")
        return None

    context_encoder.eval()
    denoising_net.eval()
    
    running_val_loss = 0.0
    num_batches = 0

    def calculate_loss(batch_data):
        batch_size = len(batch_data['visible_points'])
        if batch_size == 0: return torch.tensor(0.0)
        total_loss = 0.0; valid_samples = 0
        for i in range(batch_size):
            vis_pts = batch_data['visible_points'][i].to(device)
            vis_lbl = batch_data['visible_labels_one_hot'][i].to(device)
            vis_pd = batch_data['visible_plane_dist'][i].to(device)
            occ_gt_pts = batch_data['occluded_gt_points'][i].to(device)
            occ_gt_lbl = batch_data['occluded_gt_labels_one_hot'][i].to(device)
            occ_gt_pd = batch_data['occluded_gt_plane_dist'][i].to(device)

            if vis_pts.shape[0] == 0 or occ_gt_pts.shape[0] == 0: continue
            
            ctx_feat = context_encoder(vis_pts, vis_pd, vis_lbl)
            t_indices = torch.randint(0, scheduler.num_timesteps, (occ_gt_pts.shape[0],), device=device)
            t_norm = t_indices.float().mean() / scheduler.num_timesteps
            noisy_occ, noise_gt = scheduler.add_noise(occ_gt_pts, t_indices)
            pred_noise = denoising_net(noisy_occ, t_norm, ctx_feat, occ_gt_lbl, occ_gt_pd)
            if pred_noise.shape[0] > 0:
                total_loss += torch.nn.functional.mse_loss(pred_noise, noise_gt); valid_samples += 1

        return total_loss / valid_samples if valid_samples > 0 else torch.tensor(0.0)

    with torch.no_grad():
        for batch_data in tqdm(val_dataloader, desc=f"  Validating Epoch {checkpoint.get('epoch', 'N/A')}", leave=False):
            val_loss = calculate_loss(batch_data)
            running_val_loss += val_loss.item()
            num_batches += 1

    avg_val_loss = running_val_loss / num_batches if num_batches > 0 else 0
    print(f"  - Calculated Average Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss


def main():
    print("--- Generating Validation Loss Curve from Checkpoints ---")
    
    checkpoint_dir = config.CHECKPOINT_DIR
    if not os.path.isdir(checkpoint_dir):
        print(f"Error: Checkpoint directory '{checkpoint_dir}' not found.")
        return

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth.tar"))
    if not checkpoint_files:
        print(f"No epoch checkpoints found in '{checkpoint_dir}' matching 'checkpoint_epoch_*.pth.tar'")
        return

    checkpoints_to_process = []
    for f_path in checkpoint_files:
        match = re.search(r'checkpoint_epoch_(\d+)\.pth\.tar', os.path.basename(f_path))
        if match:
            epoch_num = int(match.group(1))
            checkpoints_to_process.append((epoch_num, f_path))

    checkpoints_to_process.sort()
    print(f"Found {len(checkpoints_to_process)} epoch checkpoints to evaluate.")

    print("\nLoading validation dataset...")
    try:
        val_dataset = KittiCompletionDataset(config.PREPROCESSED_DATA_DIR, split='val')
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure preprocess_dataset.py has run for validation split.")
        return

    if len(val_dataset) == 0:
        print("Validation dataset is empty. Cannot generate curve.")
        return

    val_dataloader = DataLoader(val_dataset, batch_size=config.DATALOADER_BATCH_SIZE, shuffle=False,
                                num_workers=config.NUM_WORKERS_DATALOADER, collate_fn=kitti_eval_collate_fn)

    recalculated_val_losses = []
    processed_epochs = []

    for epoch, ckpt_path in checkpoints_to_process:
        val_loss = get_validation_loss_for_checkpoint(ckpt_path, val_dataloader, config.DEVICE)
        if val_loss is not None:
            recalculated_val_losses.append(val_loss)
            processed_epochs.append(epoch)
    
    # --- PLOTTING LOGIC (VALIDATION ONLY) ---
    if recalculated_val_losses:
        print("\nPlotting final validation loss curve...")
        plt.figure(figsize=(12, 7))
        
        # Plot the recalculated, accurate validation loss curve
        plt.plot(processed_epochs, recalculated_val_losses, label='Average Validation Loss', marker='x', linestyle='--', color='orange')
        
        # Highlight the best validation loss
        best_val_loss = min(recalculated_val_losses)
        best_epoch = processed_epochs[recalculated_val_losses.index(best_val_loss)]
        plt.axvline(x=best_epoch, color='r', linestyle=':', linewidth=1.5, label=f'Best Val Loss at Epoch {best_epoch} ({best_val_loss:.4f})')
        plt.scatter(best_epoch, best_val_loss, s=120, facecolors='none', edgecolors='r', linewidth=2.0, zorder=5)

        plt.title("Model Performance on Validation Set Over Epochs")
        plt.xlabel("Global Epoch")
        plt.ylabel("Average MSE Loss")
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(0, max(processed_epochs) + 5, 5)) # Ticks every 5 epochs
        plt.tight_layout()

        output_dir = config.TRAINING_PLOTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        final_plot_path = os.path.join(output_dir, "reconstructed_validation_loss_curve.png")
        
        try:
            plt.savefig(final_plot_path)
            print(f"\nSuccessfully saved validation loss curve to: {final_plot_path}")
        except Exception as e:
            print(f"\nError saving plot: {e}")

        # plt.show()
    else:
        print("Could not generate any validation loss data to plot.")

if __name__ == "__main__":
    main()