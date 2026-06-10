# inference_utils.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

try:
    import config
except ImportError:
    print("Warning: config.py not found in inference_utils.py.")
    class FallbackConfig:
        DEVICE = torch.device("cpu"); POINT_DIM = 3
    config = FallbackConfig()


@torch.no_grad()
def sample_completion(
    trained_context_encoder, 
    trained_denoising_net, 
    trained_scheduler,
    pts_vis_torch,
    pd_vis_torch,
    lbl_vis_one_hot_torch,
    num_points_to_generate_in_occ,
    target_semantic_guidance_occ,
    target_plane_dist_guidance_occ,
    device=config.DEVICE
):
    """
    Generates completed points for an occluded region using trained models.
    """
    trained_context_encoder.eval()
    trained_denoising_net.eval()
    
    if pts_vis_torch.shape[0] > 0:
        global_context_feat = trained_context_encoder(pts_vis_torch, pd_vis_torch, lbl_vis_one_hot_torch)
        visible_points_center = torch.mean(pts_vis_torch, dim=0, keepdim=True)
    else:
        global_context_feat = torch.zeros(1, config.EMBED_DIM, device=device)
        visible_points_center = torch.zeros(1, config.POINT_DIM, device=device)

    if num_points_to_generate_in_occ <= 0:
        return np.empty((0, config.POINT_DIM))

    point_dim_val = config.POINT_DIM
    current_points_gen_occ = torch.randn(num_points_to_generate_in_occ, point_dim_val).to(device) 
    
    for t_idx in reversed(range(trained_scheduler.num_timesteps)):
        t_norm_for_net = torch.tensor([t_idx / trained_scheduler.num_timesteps], device=device).float()
        
        # --- THE FIX IS HERE ---
        # Unpack the tuple output from the denoising network.
        # For inference, we only need the predicted_noise to denoise the points.
        predicted_noise, _ = trained_denoising_net(
            current_points_gen_occ, 
            t_norm_for_net, 
            global_context_feat,
            target_semantic_guidance_occ, 
            target_plane_dist_guidance_occ
        )
        # --- END FIX ---
        
        # Pass only the noise tensor to the denoise_step function
        current_points_gen_occ = trained_scheduler.denoise_step(
            predicted_noise, 
            current_points_gen_occ, 
            t_idx
        )
    
    re_centered_points = current_points_gen_occ + visible_points_center
    return re_centered_points.cpu().numpy()


if __name__ == '__main__':
    print("This script defines the sample_completion function and is not meant to be run directly.")
    print("Please run evaluate.py to test the full inference pipeline.")
