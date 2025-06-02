# model_components.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Attempt to import from config.py.
try:
    import config
except ImportError:
    print("Warning: config.py not found. Using fallback default values for model parameters.")
    class FallbackConfig:
        NUM_CLASSES = 20
        POINT_DIM = 3
        PLANE_DIST_DIM = 1
        EMBED_DIM = 256
        TIME_EMBED_DIM = 128
        HIDDEN_DIM_DENOISING = 256
        NUM_TIMESTEPS = 1000
        BETA_START = 0.0001
        BETA_END = 0.02
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = FallbackConfig()


# 1. Context Encoder
class ContextEncoder(nn.Module):
    """
    Encodes the visible point cloud and its priors into a global context vector.
    Fuses geometric (coordinates), structural (plane distance), 
    and semantic (labels) information from visible points.
    """
    def __init__(self, point_dim=config.POINT_DIM, 
                 plane_dist_dim=config.PLANE_DIST_DIM, 
                 semantic_dim=config.NUM_CLASSES, 
                 embed_dim=config.EMBED_DIM):
        super().__init__()
        self.point_dim = point_dim
        self.plane_dist_dim = plane_dist_dim
        self.semantic_dim = semantic_dim
        self.embed_dim = embed_dim
        
        input_feat_dim = self.point_dim + self.plane_dist_dim + self.semantic_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_feat_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.embed_dim)
        )
        # print(f"ContextEncoder initialized with input_feat_dim: {input_feat_dim}, embed_dim: {self.embed_dim}")

    def forward(self, points_vis, plane_dist_vis, labels_vis_one_hot):
        if points_vis.shape[0] == 0:
            return torch.zeros(1, self.embed_dim, device=points_vis.device)

        combined_features = torch.cat([points_vis, plane_dist_vis, labels_vis_one_hot], dim=1)
        point_features = self.mlp(combined_features)
        
        global_feat, _ = torch.max(point_features, dim=0, keepdim=True)
        return global_feat

# 2. Denoising Network (Core of the Diffusion Model)
class DenoisingNetwork(nn.Module):
    """
    Predicts the noise to remove from the currently noisy occluded points.
    Conditioned on global context, diffusion timestep, AND
    target semantic labels & estimated geometric priors (plane_dist) for the occluded region.
    """
    def __init__(self, point_dim=config.POINT_DIM, 
                 context_dim=config.EMBED_DIM, 
                 time_embed_dim=config.TIME_EMBED_DIM, 
                 semantic_cond_dim=config.NUM_CLASSES, 
                 plane_dist_cond_dim=config.PLANE_DIST_DIM, 
                 hidden_dim=config.HIDDEN_DIM_DENOISING):
        super().__init__()
        self.point_dim = point_dim
        self.context_dim = context_dim
        self.time_embed_dim = time_embed_dim
        self.semantic_cond_dim = semantic_cond_dim
        self.plane_dist_cond_dim = plane_dist_cond_dim
        self.hidden_dim = hidden_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_embed_dim), # Expects input shape [batch_size, 1] or [1,1] for single time
            nn.ReLU(), 
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        input_feat_dim = (self.point_dim + self.time_embed_dim + self.context_dim + 
                          self.semantic_cond_dim + self.plane_dist_cond_dim)
        
        self.main_mlp = nn.Sequential(
            nn.Linear(input_feat_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.point_dim) 
        )
        # print(f"DenoisingNetwork initialized with input_feat_dim: {input_feat_dim}, output_dim (noise): {self.point_dim}")

    def forward(self, noisy_points_occ, t_norm, global_context_feat, 
                semantic_guidance_occ, plane_dist_guidance_occ):
        # noisy_points_occ: [N_occ, point_dim]
        # t_norm: scalar tensor or 1-element 1D tensor, normalized diffusion timestep [0, 1)
        # global_context_feat: [1, context_dim]
        # semantic_guidance_occ: [N_occ, semantic_cond_dim]
        # plane_dist_guidance_occ: [N_occ, plane_dist_cond_dim]

        if noisy_points_occ.shape[0] == 0:
            return torch.empty(0, self.point_dim, device=noisy_points_occ.device)

        N_occ = noisy_points_occ.shape[0]
        
        # Ensure t_norm is correctly shaped to [1, 1] for time_mlp
        # t_norm is expected to be a single time value for this batch of points
        time_input = t_norm.view(1, 1) # Reshape scalar or [1] to [1,1]
        time_embed = self.time_mlp(time_input) # Output shape: [1, self.time_embed_dim]
        
        time_embed_expanded = time_embed.repeat(N_occ, 1) # Shape: [N_occ, self.time_embed_dim]     
        
        context_feat_expanded = global_context_feat.repeat(N_occ, 1)
        
        combined_features = torch.cat([
            noisy_points_occ, 
            time_embed_expanded, 
            context_feat_expanded,
            semantic_guidance_occ,
            plane_dist_guidance_occ
        ], dim=1)
        
        predicted_noise = self.main_mlp(combined_features)
        return predicted_noise

# 3. Diffusion Scheduler
class DiffusionScheduler:
    """
    Manages the forward (noising) and reverse (denoising) diffusion processes.
    """
    def __init__(self, num_timesteps=config.NUM_TIMESTEPS, 
                 beta_start=config.BETA_START, 
                 beta_end=config.BETA_END, 
                 device=config.DEVICE):
        self.num_timesteps = num_timesteps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))

        self.posterior_mean_coef1 = self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        # print(f"DiffusionScheduler initialized for {num_timesteps} timesteps on device {device}.")


    def add_noise(self, x_start, t_indices, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t_indices].view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_indices].view(-1, 1)
        
        noisy_points = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
        return noisy_points, noise

    def denoise_step(self, model_output_noise, noisy_points_t, t_idx_scalar):
        pred_x0 = (noisy_points_t - self.sqrt_one_minus_alphas_cumprod[t_idx_scalar] * model_output_noise) / \
                  self.sqrt_alphas_cumprod[t_idx_scalar]
        # pred_x0 = torch.clamp(pred_x0, -50, 50) # Optional

        mean_prev_t = self.posterior_mean_coef1[t_idx_scalar] * pred_x0 + \
                      self.posterior_mean_coef2[t_idx_scalar] * noisy_points_t
        
        if t_idx_scalar == 0:
            return mean_prev_t
        
        log_variance = self.posterior_log_variance_clipped[t_idx_scalar]
        noise_for_step = torch.randn_like(noisy_points_t)
        return mean_prev_t + torch.exp(0.5 * log_variance) * noise_for_step


if __name__ == '__main__':
    print("--- Testing model_components.py (Corrected) ---")
    
    print("\nTesting ContextEncoder...")
    N_vis_test = 100
    points_vis_test = torch.randn(N_vis_test, config.POINT_DIM).to(config.DEVICE)
    plane_dist_vis_test = torch.randn(N_vis_test, config.PLANE_DIST_DIM).to(config.DEVICE)
    labels_vis_one_hot_test = F.one_hot(torch.randint(0, config.NUM_CLASSES, (N_vis_test,)), num_classes=config.NUM_CLASSES).float().to(config.DEVICE)
    
    context_encoder_test = ContextEncoder().to(config.DEVICE)
    global_feat_test = context_encoder_test(points_vis_test, plane_dist_vis_test, labels_vis_one_hot_test)
    print(f"ContextEncoder output shape: {global_feat_test.shape} (Expected: [1, {config.EMBED_DIM}])")
    assert global_feat_test.shape == (1, config.EMBED_DIM)

    print("\nTesting DenoisingNetwork...")
    N_occ_test = 50
    noisy_points_occ_test = torch.randn(N_occ_test, config.POINT_DIM).to(config.DEVICE)
    
    # Test with scalar t_norm
    t_norm_scalar_test = torch.rand(1).item() # Get Python scalar
    t_norm_scalar_tensor_test = torch.tensor(t_norm_scalar_test, device=config.DEVICE) # 0-dim tensor
    
    # Test with 1-element 1D t_norm
    t_norm_1d_tensor_test = torch.rand(1).to(config.DEVICE) # Shape [1]

    global_context_feat_test = torch.randn(1, config.EMBED_DIM).to(config.DEVICE)
    semantic_guidance_occ_test = F.one_hot(torch.randint(0, config.NUM_CLASSES, (N_occ_test,)), num_classes=config.NUM_CLASSES).float().to(config.DEVICE)
    plane_dist_guidance_occ_test = torch.randn(N_occ_test, config.PLANE_DIST_DIM).to(config.DEVICE)

    denoising_net_test = DenoisingNetwork().to(config.DEVICE)
    
    print("  Testing DenoisingNetwork with scalar t_norm...")
    predicted_noise_test_scalar_t = denoising_net_test(noisy_points_occ_test, t_norm_scalar_tensor_test, global_context_feat_test,
                                           semantic_guidance_occ_test, plane_dist_guidance_occ_test)
    print(f"  DenoisingNetwork (scalar t_norm) output shape: {predicted_noise_test_scalar_t.shape} (Expected: [{N_occ_test}, {config.POINT_DIM}])")
    assert predicted_noise_test_scalar_t.shape == (N_occ_test, config.POINT_DIM)

    print("  Testing DenoisingNetwork with 1D (1-element) t_norm...")
    predicted_noise_test_1d_t = denoising_net_test(noisy_points_occ_test, t_norm_1d_tensor_test, global_context_feat_test,
                                           semantic_guidance_occ_test, plane_dist_guidance_occ_test)
    print(f"  DenoisingNetwork (1D t_norm) output shape: {predicted_noise_test_1d_t.shape} (Expected: [{N_occ_test}, {config.POINT_DIM}])")
    assert predicted_noise_test_1d_t.shape == (N_occ_test, config.POINT_DIM)


    print("\nTesting DiffusionScheduler...")
    scheduler_test = DiffusionScheduler(device=config.DEVICE)
    x_start_test = torch.randn(N_occ_test, config.POINT_DIM).to(config.DEVICE)
    t_indices_test = torch.randint(0, config.NUM_TIMESTEPS, (N_occ_test,), device=config.DEVICE)
    
    noisy_points_added_test, noise_gt_test = scheduler_test.add_noise(x_start_test, t_indices_test)
    print(f"Scheduler add_noise output shape: {noisy_points_added_test.shape}")
    assert noisy_points_added_test.shape == x_start_test.shape

    t_idx_scalar_test = config.NUM_TIMESTEPS // 2
    # To test denoise_step, we need noisy_points_at_t_test where all points were noised with t_idx_scalar_test
    noisy_points_at_t_test, _ = scheduler_test.add_noise(x_start_test, torch.full_like(t_indices_test, t_idx_scalar_test))
    model_output_noise_sim_test = torch.randn_like(x_start_test)
    
    denoised_points_test = scheduler_test.denoise_step(model_output_noise_sim_test, noisy_points_at_t_test, t_idx_scalar_test)
    print(f"Scheduler denoise_step output shape: {denoised_points_test.shape}")
    assert denoised_points_test.shape == x_start_test.shape
    
    print("\n--- Finished testing model_components.py (Corrected) ---")