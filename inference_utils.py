# inference_utils.py

import torch
import torch.nn.functional as F # For F.one_hot if used in testing
from tqdm import tqdm
import numpy as np
import os # For os.path.basename in test

# Assuming config.py and model_components.py are in the same directory or accessible
try:
    import config
    from model_components import ContextEncoder, DenoisingNetwork, DiffusionScheduler
except ImportError:
    print("Warning: Could not import from config or model_components. Ensure they are in the Python path.")
    # Define fallback minimal config if imports fail
    class FallbackConfig:
        DEVICE = torch.device("cpu")
        NUM_TIMESTEPS = 1000
        NUM_CLASSES = 20
        POINT_DIM = 3
        PLANE_DIST_DIM = 1
        EMBED_DIM = 256
        TIME_EMBED_DIM = 128
        HIDDEN_DIM_DENOISING = 256
        BETA_START = 0.0001
        BETA_END = 0.02

    config = FallbackConfig()
    # Dummy classes if model_components failed to load
    if 'ContextEncoder' not in globals(): ContextEncoder = type('ContextEncoder', (torch.nn.Module,), {})
    if 'DenoisingNetwork' not in globals(): DenoisingNetwork = type('DenoisingNetwork', (torch.nn.Module,), {})
    if 'DiffusionScheduler' not in globals(): DiffusionScheduler = type('DiffusionScheduler', (object,), {})


@torch.no_grad()
def sample_completion(
    trained_context_encoder, 
    trained_denoising_net, 
    trained_scheduler,
    pts_vis_torch,                # Tensor [N_vis, point_dim]
    pd_vis_torch,                 # Tensor [N_vis, plane_dist_dim]
    lbl_vis_one_hot_torch,        # Tensor [N_vis, num_classes]
    num_points_to_generate_in_occ,
    # Guidance for the occluded region during INFERENCE
    # These would typically be PREDICTED by other models in a full system.
    target_semantic_guidance_occ, # Tensor [N_occ_gen, num_classes]
    target_plane_dist_guidance_occ, # Tensor [N_occ_gen, plane_dist_dim]
    device=config.DEVICE
):
    """
    Generates completed points for an occluded region using trained models.

    Args:
        trained_context_encoder (nn.Module): The trained context encoder model.
        trained_denoising_net (nn.Module): The trained denoising network.
        trained_scheduler (DiffusionScheduler): The configured diffusion scheduler.
        pts_vis_torch (torch.Tensor): Visible points tensor.
        pd_vis_torch (torch.Tensor): Plane distances for visible points.
        lbl_vis_one_hot_torch (torch.Tensor): One-hot labels for visible points.
        num_points_to_generate_in_occ (int): Number of points to generate for the occluded part.
        target_semantic_guidance_occ (torch.Tensor): Semantic guidance for the points to be generated.
        target_plane_dist_guidance_occ (torch.Tensor): Plane distance guidance for points to be generated.
        device (torch.device): Device to perform computation on.

    Returns:
        np.ndarray: Generated points for the occluded region as a NumPy array.
    """
    
    if trained_context_encoder is None or trained_denoising_net is None or trained_scheduler is None:
        print("Error in sample_completion: One or more models/scheduler are None. Skipping sampling.")
        return np.empty((0, config.POINT_DIM if hasattr(config, 'POINT_DIM') else 3))

    trained_context_encoder.eval()
    trained_denoising_net.eval()
    
    if pts_vis_torch.shape[0] == 0:
        print("Warning: No visible points provided for context during sampling. Generated points might be poor.")
        # Use the embed_dim from the model instance if possible
        embed_dim = trained_context_encoder.embed_dim if hasattr(trained_context_encoder, 'embed_dim') else config.EMBED_DIM
        global_context_feat = torch.zeros(1, embed_dim, device=device)
    else:
        global_context_feat = trained_context_encoder(pts_vis_torch, pd_vis_torch, lbl_vis_one_hot_torch)
    
    if num_points_to_generate_in_occ <= 0:
        # print("Debug: num_points_to_generate_in_occ is 0 or negative. Returning empty array.")
        return np.empty((0, config.POINT_DIM if hasattr(config, 'POINT_DIM') else 3))

    # Start with random noise for the occluded region
    point_dim_val = config.POINT_DIM if hasattr(config, 'POINT_DIM') else 3
    current_points_gen_occ = torch.randn(num_points_to_generate_in_occ, point_dim_val).to(device)
    
    # print(f"Starting sampling for {num_points_to_generate_in_occ} points in occluded region...")
    # Using tqdm directly in the main loop if this function is called multiple times.
    # Or, can add a desc to tqdm if called once per scan.
    for t_idx in reversed(range(trained_scheduler.num_timesteps)): # tqdm here can be verbose if called per scan for many scans
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
        
    return current_points_gen_occ.cpu().numpy()


if __name__ == '__main__':
    print("--- Testing inference_utils.py ---")

    # This test requires config.py and model_components.py to be functional.
    try:
        import config # Re-import to ensure test uses its values
        # Models are already imported at the top of the file
    except ImportError:
        print("Error: Cannot run test. Missing config.py or model_components.py.")
        exit()

    # 1. Instantiate models and scheduler for testing
    print("Instantiating models for test...")
    context_encoder_test = ContextEncoder(
        point_dim=config.POINT_DIM, plane_dist_dim=config.PLANE_DIST_DIM, 
        semantic_dim=config.NUM_CLASSES, embed_dim=config.EMBED_DIM
    ).to(config.DEVICE)
    
    denoising_net_test = DenoisingNetwork(
        point_dim=config.POINT_DIM, context_dim=config.EMBED_DIM, 
        time_embed_dim=config.TIME_EMBED_DIM, semantic_cond_dim=config.NUM_CLASSES, 
        plane_dist_cond_dim=config.PLANE_DIST_DIM, hidden_dim=config.HIDDEN_DIM_DENOISING
    ).to(config.DEVICE)
    
    scheduler_test = DiffusionScheduler(
        num_timesteps=config.NUM_TIMESTEPS, beta_start=config.BETA_START, 
        beta_end=config.BETA_END, device=config.DEVICE
    )
    print("Models instantiated.")

    # 2. Create dummy input data for testing sample_completion
    N_vis_test = 100
    pts_vis_test = torch.randn(N_vis_test, config.POINT_DIM).to(config.DEVICE)
    pd_vis_test = torch.randn(N_vis_test, config.PLANE_DIST_DIM).to(config.DEVICE)
    lbl_vis_one_hot_test = F.one_hot(torch.randint(0, config.NUM_CLASSES, (N_vis_test,)), num_classes=config.NUM_CLASSES).float().to(config.DEVICE)
    
    num_to_generate_test = 50
    # Dummy guidance (in a real scenario, these would be predicted or intelligently derived)
    target_semantic_guidance_test = F.one_hot(torch.randint(0, config.NUM_CLASSES, (num_to_generate_test,)), num_classes=config.NUM_CLASSES).float().to(config.DEVICE)
    target_plane_dist_guidance_test = torch.randn(num_to_generate_test, config.PLANE_DIST_DIM).to(config.DEVICE)

    print(f"\nCalling sample_completion to generate {num_to_generate_test} points...")
    generated_points_np = sample_completion(
        context_encoder_test, denoising_net_test, scheduler_test,
        pts_vis_test, pd_vis_test, lbl_vis_one_hot_test,
        num_to_generate_test,
        target_semantic_guidance_test,
        target_plane_dist_guidance_test,
        device=config.DEVICE
    )

    print(f"sample_completion output shape: {generated_points_np.shape} (Expected: [{num_to_generate_test}, {config.POINT_DIM}])")
    assert generated_points_np.shape == (num_to_generate_test, config.POINT_DIM)

    # Test with no visible points
    print("\nCalling sample_completion with no visible points...")
    generated_points_no_vis_np = sample_completion(
        context_encoder_test, denoising_net_test, scheduler_test,
        torch.empty(0, config.POINT_DIM).to(config.DEVICE), 
        torch.empty(0, config.PLANE_DIST_DIM).to(config.DEVICE), 
        torch.empty(0, config.NUM_CLASSES).to(config.DEVICE),
        num_to_generate_test,
        target_semantic_guidance_test,
        target_plane_dist_guidance_test,
        device=config.DEVICE
    )
    print(f"sample_completion (no visible) output shape: {generated_points_no_vis_np.shape}")
    assert generated_points_no_vis_np.shape == (num_to_generate_test, config.POINT_DIM)
    
    # Test with num_points_to_generate_in_occ = 0
    print("\nCalling sample_completion with num_points_to_generate_in_occ = 0...")
    generated_points_zero_gen_np = sample_completion(
        context_encoder_test, denoising_net_test, scheduler_test,
        pts_vis_test, pd_vis_test, lbl_vis_one_hot_test,
        0, # num_points_to_generate_in_occ = 0
        torch.empty(0, config.NUM_CLASSES).to(config.DEVICE), # Empty guidance
        torch.empty(0, config.PLANE_DIST_DIM).to(config.DEVICE), # Empty guidance
        device=config.DEVICE
    )
    print(f"sample_completion (zero generate) output shape: {generated_points_zero_gen_np.shape}")
    assert generated_points_zero_gen_np.shape == (0, config.POINT_DIM)


    print("\n--- Finished testing inference_utils.py ---")
