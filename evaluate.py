# evaluate.py

import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import open3d as o3d

try:
    import config
    from model_components import ContextEncoder, DenoisingNetwork, DiffusionScheduler
    from kitti_completion_dataset import KittiCompletionDataset
    from inference_utils import sample_completion 
except ImportError as e:
    print(f"CRITICAL Error importing necessary modules: {e}")
    exit()

# --- (All helper functions like chamfer_distance_loss_chunked, load_trained_checkpoint, etc., remain the same) ---
def pairwise_distance_sq(x, y):
    x_norm = (x**2).sum(1).view(-1, 1); y_norm = (y**2).sum(1).view(1, -1)
    return torch.clamp(x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1)), min=0.0)

def chamfer_distance_loss_chunked(p1, p2, chunk_size=1024):
    if p1.shape[0] == 0 or p2.shape[0] == 0:
        return torch.tensor(1e9 if p1.shape[0] != p2.shape[0] else 0.0, device=p1.device)
    dist_p1_p2 = 0.0
    for chunk in torch.split(p1, chunk_size):
        min_dist, _ = torch.min(pairwise_distance_sq(chunk, p2), dim=1)
        dist_p1_p2 += torch.sum(min_dist)
    loss_p1_p2 = dist_p1_p2 / p1.shape[0]
    dist_p2_p1 = 0.0
    for chunk in torch.split(p2, chunk_size):
        min_dist, _ = torch.min(pairwise_distance_sq(chunk, p1), dim=1)
        dist_p2_p1 += torch.sum(min_dist)
    loss_p2_p1 = dist_p2_p1 / p2.shape[0]
    return loss_p1_p2 + loss_p2_p1

def load_trained_checkpoint(model_ce, model_dn, ckpt_path, dev):
    if not os.path.isfile(ckpt_path): print(f"Error: Checkpoint not found: {ckpt_path}"); return False
    print(f"Loading checkpoint for evaluation: '{ckpt_path}'")
    try:
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=True)
        model_ce.load_state_dict(ckpt['context_encoder_state_dict'])
        model_dn.load_state_dict(ckpt['denoising_net_state_dict'])
        print(f"Checkpoint loaded successfully. Epoch: {ckpt.get('epoch', 'N/A')}"); return True
    except Exception as e: print(f"Error loading checkpoint: {e}"); return False

def kitti_eval_collate_fn(batch_list):
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


def evaluate_model(checkpoint_filename="latest_checkpoint.pth.tar"):
    print("--- Starting Full Model Evaluation ---")
    eval_pcd_output_dir = config.EVALUATION_PCD_OUTPUT_DIR
    os.makedirs(eval_pcd_output_dir, exist_ok=True)
    print(f"Saving comparison PCDs to: {eval_pcd_output_dir}")

    ctx_enc = ContextEncoder().to(config.DEVICE); den_net = DenoisingNetwork().to(config.DEVICE)
    scheduler = DiffusionScheduler(device=config.DEVICE)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, checkpoint_filename)
    if not load_trained_checkpoint(ctx_enc, den_net, ckpt_path, config.DEVICE): return

    ctx_enc.eval(); den_net.eval()
    print("Loading validation dataset...")
    try:
        val_dataset = KittiCompletionDataset(config.PREPROCESSED_DATA_DIR, split='val')
    except FileNotFoundError as e: print(f"Error: {e}"); return
    if len(val_dataset) == 0: print("Validation dataset empty."); return

    val_loader = DataLoader(val_dataset, batch_size=config.DATALOADER_BATCH_SIZE, shuffle=False, 
                            num_workers=config.NUM_WORKERS_DATALOADER, pin_memory=True, drop_last=False,
                            collate_fn=kitti_eval_collate_fn)
    print(f"Validation dataset: {len(val_dataset)} samples.")

    total_cd = 0.0; num_samples_evaluated = 0
    prog_bar = tqdm(val_loader, desc="Evaluating on Validation Set")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(prog_bar):
            visible_points_batch_list = batch_data['visible_points']
            visible_labels_batch_list = batch_data['visible_labels_one_hot']
            visible_plane_dist_batch_list = batch_data['visible_plane_dist']
            occluded_gt_points_batch_list = batch_data['occluded_gt_points']
            occluded_gt_labels_batch_list = batch_data['occluded_gt_labels_one_hot']
            occluded_gt_plane_dist_batch_list = batch_data['occluded_gt_plane_dist']
            original_full_points_batch_list = batch_data.get('original_full_points', [])
            metadata_batch_list = batch_data.get('metadata', [{} for _ in range(len(visible_points_batch_list))])

            batch_size_current = len(visible_points_batch_list)
            
            for i in range(batch_size_current):
                if config.MAX_SAMPLES_TO_EVALUATE is not None and \
                   num_samples_evaluated >= config.MAX_SAMPLES_TO_EVALUATE:
                    print(f"\nReached max evaluation samples limit of {config.MAX_SAMPLES_TO_EVALUATE}. Stopping evaluation.")
                    break
                
                vis_pts = visible_points_batch_list[i].to(config.DEVICE)
                vis_lbl = visible_labels_batch_list[i].to(config.DEVICE)
                vis_pd = visible_plane_dist_batch_list[i].to(config.DEVICE)
                occ_gt_pts = occluded_gt_points_batch_list[i].to(config.DEVICE)
                occ_gt_lbl_guide = occluded_gt_labels_batch_list[i].to(config.DEVICE)
                occ_gt_pd_guide = occluded_gt_plane_dist_batch_list[i].to(config.DEVICE)
                original_full_pts_sample = original_full_points_batch_list[i].to(config.DEVICE) if i < len(original_full_points_batch_list) else None
                meta = metadata_batch_list[i]
                seq_id = meta.get('sequence_id', 'sXX'); scan_id = meta.get('scan_id', f'b{batch_idx}_i{i}'); occ_inst = meta.get('occlusion_instance', 0)
                base_fn = f"seq{seq_id}_scan{scan_id}_occ{occ_inst}"

                if vis_pts.shape[0] == 0 or occ_gt_pts.shape[0] == 0: continue

                num_gen = occ_gt_pts.shape[0]
                
                # --- THIS IS THE CORRECTED LINE ---
                gen_occ_np = sample_completion(
                    ctx_enc, den_net, scheduler,
                    vis_pts, vis_pd, vis_lbl,
                    num_gen,
                    occ_gt_lbl_guide,
                    occ_gt_pd_guide,
                    device=config.DEVICE
                )
                # --- END CORRECTION ---
                
                # --- (PCD saving logic as before) ---
                if original_full_pts_sample is not None and original_full_pts_sample.shape[0] > 0:
                    pcd_orig = o3d.geometry.PointCloud(); pcd_orig.points = o3d.utility.Vector3dVector(original_full_pts_sample.cpu().numpy())
                    o3d.io.write_point_cloud(os.path.join(eval_pcd_output_dir, f"{base_fn}_00_original_full.pcd"), pcd_orig)
                pcd_vis = o3d.geometry.PointCloud(); pcd_vis.points = o3d.utility.Vector3dVector(vis_pts.cpu().numpy())
                o3d.io.write_point_cloud(os.path.join(eval_pcd_output_dir, f"{base_fn}_01_visible_input.pcd"), pcd_vis)
                pcd_gt_occ = o3d.geometry.PointCloud(); pcd_gt_occ.points = o3d.utility.Vector3dVector(occ_gt_pts.cpu().numpy())
                o3d.io.write_point_cloud(os.path.join(eval_pcd_output_dir, f"{base_fn}_02_gt_occluded_part.pcd"), pcd_gt_occ)
                if gen_occ_np.shape[0] > 0:
                    pcd_gen_occ = o3d.geometry.PointCloud(); pcd_gen_occ.points = o3d.utility.Vector3dVector(gen_occ_np)
                    o3d.io.write_point_cloud(os.path.join(eval_pcd_output_dir, f"{base_fn}_03_gen_occluded_part.pcd"), pcd_gen_occ)
                    full_completed_np = np.vstack((vis_pts.cpu().numpy(), gen_occ_np))
                    pcd_full_comp = o3d.geometry.PointCloud(); pcd_full_comp.points = o3d.utility.Vector3dVector(full_completed_np)
                    o3d.io.write_point_cloud(os.path.join(eval_pcd_output_dir, f"{base_fn}_04_completed_full.pcd"), pcd_full_comp)
                else:
                    o3d.io.write_point_cloud(os.path.join(eval_pcd_output_dir, f"{base_fn}_04_completed_full.pcd"), pcd_vis)
                # --- END PCD saving logic ---

                gen_pts_tensor = torch.from_numpy(gen_occ_np).float().to(config.DEVICE)
                cd = chamfer_distance_loss_chunked(gen_pts_tensor, occ_gt_pts)
                if not torch.isinf(cd) and not torch.isnan(cd):
                    total_cd += cd.item(); num_samples_evaluated += 1
                else: print(f"Warning: Invalid CD for {base_fn}")
            
            if config.MAX_SAMPLES_TO_EVALUATE is not None and \
               num_samples_evaluated >= config.MAX_SAMPLES_TO_EVALUATE:
                break
            
            prog_bar.set_postfix({"Avg CD": f"{(total_cd / num_samples_evaluated if num_samples_evaluated > 0 else 0):.4f}"})

    print("\n--- Evaluation Results ---")
    if num_samples_evaluated > 0: print(f"Average Chamfer Distance (CD) over {num_samples_evaluated} samples: {(total_cd / num_samples_evaluated):.6f}")
    else: print("No samples were successfully evaluated.")
    print(f"Saved comparison PCDs to: {eval_pcd_output_dir}")
    print("--- Evaluation Finished ---")

if __name__ == "__main__":
    default_ckpt = os.path.join(config.CHECKPOINT_DIR if hasattr(config, 'CHECKPOINT_DIR') else 'training_checkpoints', "latest_checkpoint.pth.tar")
    if not os.path.exists(default_ckpt): print(f"Default checkpoint '{default_ckpt}' not found.")
    else: evaluate_model(checkpoint_filename="latest_checkpoint.pth.tar")
