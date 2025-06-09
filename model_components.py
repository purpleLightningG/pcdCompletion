# model_components.py

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import config
except ImportError:
    # ... (FallbackConfig as before)
    print("Warning: config.py not found. Using fallback values.")
    class FallbackConfig:
        NUM_CLASSES = 20; POINT_DIM = 3; PLANE_DIST_DIM = 1; EMBED_DIM = 256
        TIME_EMBED_DIM = 128; HIDDEN_DIM_DENOISING = 256; NUM_TIMESTEPS = 1000
        BETA_START = 0.0001; BETA_END = 0.02
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = FallbackConfig()


# ContextEncoder and DenoisingNetwork classes remain unchanged from the PointNet-style version.
# ... (Paste your existing ContextEncoder and UPGRADED DenoisingNetwork classes here) ...
class ContextEncoder(nn.Module):
    def __init__(self, point_dim=config.POINT_DIM, plane_dist_dim=config.PLANE_DIST_DIM, semantic_dim=config.NUM_CLASSES, embed_dim=config.EMBED_DIM):
        super().__init__(); self.embed_dim = embed_dim
        input_feat_dim = point_dim + plane_dist_dim + semantic_dim
        self.mlp = nn.Sequential(nn.Linear(input_feat_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, self.embed_dim))
    def forward(self, points_vis, plane_dist_vis, labels_vis_one_hot):
        if points_vis.shape[0] == 0: return torch.zeros(1, self.embed_dim, device=points_vis.device)
        features = self.mlp(torch.cat([points_vis, plane_dist_vis, labels_vis_one_hot], dim=1))
        return torch.max(features, dim=0, keepdim=True)[0]

class DenoisingNetwork(nn.Module):
    def __init__(self, point_dim=config.POINT_DIM, context_dim=config.EMBED_DIM, time_embed_dim=config.TIME_EMBED_DIM, semantic_cond_dim=config.NUM_CLASSES, plane_dist_cond_dim=config.PLANE_DIST_DIM, hidden_dim=config.HIDDEN_DIM_DENOISING):
        super().__init__(); self.point_dim = point_dim
        self.shape_feature_extractor = nn.Sequential(nn.Linear(point_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
        self.time_mlp = nn.Sequential(nn.Linear(1, time_embed_dim), nn.ReLU(), nn.Linear(time_embed_dim, time_embed_dim))
        input_feat_dim = 256 + 256 + time_embed_dim + context_dim + semantic_cond_dim + plane_dist_cond_dim
        self.final_mlp = nn.Sequential(nn.Linear(input_feat_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, self.point_dim))
    def forward(self, noisy_points_occ, t_norm, global_context_feat, semantic_guidance_occ, plane_dist_guidance_occ):
        if noisy_points_occ.shape[0] == 0: return torch.empty(0, self.point_dim, device=noisy_points_occ.device)
        N_occ = noisy_points_occ.shape[0]
        pointwise_features = self.shape_feature_extractor(noisy_points_occ)
        current_shape_global_feat, _ = torch.max(pointwise_features, dim=0, keepdim=True)
        current_shape_global_feat_expanded = current_shape_global_feat.repeat(N_occ, 1)
        time_embed_expanded = self.time_mlp(t_norm.view(1, 1)).repeat(N_occ, 1)
        visible_context_expanded = global_context_feat.repeat(N_occ, 1)
        combined_features = torch.cat([pointwise_features, current_shape_global_feat_expanded, time_embed_expanded, visible_context_expanded, semantic_guidance_occ, plane_dist_guidance_occ], dim=1)
        return self.final_mlp(combined_features)


# Diffusion Scheduler (with one new method)
class DiffusionScheduler:
    def __init__(self, num_timesteps=config.NUM_TIMESTEPS, 
                 beta_start=config.BETA_START, 
                 beta_end=config.BETA_END, 
                 device=config.DEVICE):
        self.num_timesteps = num_timesteps; self.device = device
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

    def add_noise(self, x_start, t_indices, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t_indices].view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_indices].view(-1, 1)
        noisy_points = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
        return noisy_points, noise

    # --- NEW HELPER METHOD ---
    def predict_x0_from_noise(self, x_t, t_indices, noise_pred):
        """
        Calculates the predicted clean data x0 from the noisy data x_t and the predicted noise.
        Formula: x_0_pred = (x_t - sqrt(1 - alpha_cumprod_t) * pred_noise_t) / sqrt(alpha_cumprod_t)
        """
        # Ensure t_indices can be used to index into the schedule tensors
        # It might be a single value for a whole sample or per-point
        # We'll use view(-1, 1) to make it broadcastable
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t_indices].view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_indices].view(-1, 1)
        
        pred_x0 = (x_t - sqrt_one_minus_alpha_cumprod_t * noise_pred) / sqrt_alpha_cumprod_t
        return pred_x0

    def denoise_step(self, model_output_noise, noisy_points_t, t_idx_scalar):
        # This function now uses the new helper method for clarity
        pred_x0 = self.predict_x0_from_noise(noisy_points_t, torch.tensor([t_idx_scalar], device=self.device), model_output_noise)
        
        mean_prev_t = self.posterior_mean_coef1[t_idx_scalar] * pred_x0 + \
                      self.posterior_mean_coef2[t_idx_scalar] * noisy_points_t
        if t_idx_scalar == 0: return mean_prev_t
        log_variance = self.posterior_log_variance_clipped[t_idx_scalar]
        noise_for_step = torch.randn_like(noisy_points_t)
        return mean_prev_t + torch.exp(0.5 * log_variance) * noise_for_step


if __name__ == '__main__':
    # ... (Test block as before)
    pass # Test block can remain, its purpose is just to check for runtime errors
