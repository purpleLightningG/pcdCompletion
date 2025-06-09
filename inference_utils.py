# inference_utils.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

try:
    import config
    # We don't need the model class definitions here, as the instances are passed in.
except ImportError:
    print("Warning: config.py not found in inference_utils.py. Using fallback device.")
    class FallbackConfig:
        DEVICE = torch.device("cpu")
        POINT_DIM = 3
    config = FallbackConfig()


@torch.no_grad()
def sample_completion(
    trained_context_encoder, 
    trained_denoising_net, 
    trained_scheduler,
    pts_vis_torch,                # Tensor [N_vis, 3]
    pd_vis_torch,                 # Tensor [N_vis, 1]
    lbl_vis_one_hot_torch,        # Tensor [N_vis, num_classes]
    num_points_to_generate_in_occ,
    target_semantic_guidance_occ, # Tensor [N_occ_gen, num_classes]
    target_plane_dist_guidance_occ, # Tensor [N_occ_gen, 1]
    device=config.DEVICE
):
    """
    Generates completed points for an occluded region using trained models.
    Includes a re-centering step to correctly position the generated cloud.
    """
    
    if trained_context_encoder is None or trained_denoising_net is None or trained_scheduler is None:
        print("Error in sample_completion: One or more models/scheduler are None.")
        return np.empty((0, config.POINT_DIM))

    trained_context_encoder.eval()
    trained_denoising_net.eval()
    
    # --- STEP 1: Get Context from Visible Points ---
    if pts_vis_torch.shape[0] == 0:
        print("Warning: No visible points provided for context. Completion might be poor.")
        embed_dim = config.EMBED_DIM # Assumes config is available
        global_context_feat = torch.zeros(1, embed_dim, device=device)
        # If no visible points, we cannot calculate a center. We'll have to generate at origin.
        visible_points_center = torch.zeros(1, config.POINT_DIM, device=device)
    else:
        global_context_feat = trained_context_encoder(pts_vis_torch, pd_vis_torch, lbl_vis_one_hot_torch)
        # Calculate the center of the visible points for re-positioning later
        visible_points_center = torch.mean(pts_vis_torch, dim=0, keepdim=True) # Shape: [1, 3]

    if num_points_to_generate_in_occ <= 0:
        return np.empty((0, config.POINT_DIM))

    # --- STEP 2: Iterative Denoising (Generation around origin) ---
    point_dim_val = config.POINT_DIM
    current_points_gen_occ = torch.randn(num_points_to_generate_in_occ, point_dim_val).to(device) 
    
    # The tqdm loop here can be verbose. For single inference, it's fine.
    # For batch evaluation in evaluate.py, it's better to have the progress bar there.
    for t_idx in reversed(range(trained_scheduler.num_timesteps)):
        t_norm_for_net = torch.tensor([t_idx / trained_scheduler.num_timesteps], device=device).float()
        
        predicted_noise = trained_denoising_net(
            current_points_gen_occ, 
            t_norm_for_net, 
            global_context_feat,
            target_semantic_guidance_occ, 
            target_plane_dist_guidance_occ
        )
        
        current_points_gen_occ = trained_scheduler.denoise_step(
            predicted_noise, 
            current_points_gen_occ, 
            t_idx
        )
    
    # --- STEP 3: Re-center the Generated Points (THE FIX!) ---
    # The `current_points_gen_occ` are generated around the origin (0,0,0).
    # We translate them to be centered around the visible part of the point cloud.
    re_centered_points = current_points_gen_occ + visible_points_center

    return re_centered_points.cpu().numpy()


if __name__ == '__main__':
    # ... (The test block in this file can remain as is for basic functionality check,
    # but the real test is seeing the output from evaluate.py)
    print("--- Testing inference_utils.py (with re-centering logic) ---")
    print("This test block only checks for runtime errors. The re-centering effect must be verified visually via evaluate.py -> view_comparison.py")
    # You would need to instantiate dummy models here to run this test block,
    # as they are not defined in this file.
    print("Please run evaluate.py to test the full inference pipeline with the re-centering fix.")
