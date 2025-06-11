# model_components.py

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import config
except ImportError:
    # ... (FallbackConfig as before)
    class FallbackConfig:
        NUM_CLASSES = 20; POINT_DIM = 3; PLANE_DIST_DIM = 1; EMBED_DIM = 256; TIME_EMBED_DIM = 128
        HIDDEN_DIM_DENOISING = 256; NUM_TIMESTEPS = 1000; BETA_START = 0.0001; BETA_END = 0.02
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = FallbackConfig()


# --- ContextEncoder (unchanged) ---
class ContextEncoder(nn.Module):
    def __init__(self, point_dim=config.POINT_DIM, plane_dist_dim=config.PLANE_DIST_DIM, semantic_dim=config.NUM_CLASSES, embed_dim=config.EMBED_DIM):
        super().__init__(); self.embed_dim = embed_dim
        self.mlp = nn.Sequential(nn.Linear(point_dim + plane_dist_dim + semantic_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, self.embed_dim))
    def forward(self, points_vis, plane_dist_vis, labels_vis_one_hot):
        if points_vis.shape[0] == 0: return torch.zeros(1, self.embed_dim, device=points_vis.device)
        return torch.max(self.mlp(torch.cat([points_vis, plane_dist_vis, labels_vis_one_hot], dim=1)), dim=0, keepdim=True)[0]


# --- DenoisingNetwork with Parametric Head (unchanged from last version) ---
class DenoisingNetwork(nn.Module):
    def __init__(self, point_dim=config.POINT_DIM, context_dim=config.EMBED_DIM, time_embed_dim=config.TIME_EMBED_DIM, semantic_cond_dim=config.NUM_CLASSES, plane_dist_cond_dim=config.PLANE_DIST_DIM, hidden_dim=config.HIDDEN_DIM_DENOISING):
        super().__init__(); self.point_dim = point_dim
        self.shape_feature_extractor = nn.Sequential(nn.Linear(point_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
        self.time_mlp = nn.Sequential(nn.Linear(1, time_embed_dim), nn.ReLU(), nn.Linear(time_embed_dim, time_embed_dim))
        self.noise_prediction_mlp = nn.Sequential(nn.Linear(256+256+time_embed_dim+context_dim+semantic_cond_dim+plane_dist_cond_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, self.point_dim))
        self.parametric_head = nn.Sequential(nn.Linear(256+context_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 4))
    def forward(self, noisy_points_occ, t_norm, global_context_feat, semantic_guidance_occ, plane_dist_guidance_occ):
        if noisy_points_occ.shape[0] == 0:
            return torch.empty(0, self.point_dim, device=noisy_points_occ.device), torch.empty(0, 4, device=noisy_points_occ.device)
        pointwise_features = self.shape_feature_extractor(noisy_points_occ)
        current_shape_global_feat, _ = torch.max(pointwise_features, dim=0, keepdim=True)
        N_occ = noisy_points_occ.shape[0]
        expanded_shape_feat = current_shape_global_feat.repeat(N_occ, 1)
        expanded_time_feat = self.time_mlp(t_norm.view(1, 1)).repeat(N_occ, 1)
        expanded_context_feat = global_context_feat.repeat(N_occ, 1)
        combined_features = torch.cat([pointwise_features, expanded_shape_feat, expanded_time_feat, expanded_context_feat, semantic_guidance_occ, plane_dist_guidance_occ], dim=1)
        predicted_noise = self.noise_prediction_mlp(combined_features)
        predicted_plane_params = self.parametric_head(torch.cat([current_shape_global_feat, global_context_feat], dim=1))
        return predicted_noise, predicted_plane_params


# --- DiffusionScheduler (with robust denoise_step) ---
class DiffusionScheduler:
    def __init__(self, num_timesteps=config.NUM_TIMESTEPS, beta_start=config.BETA_START, beta_end=config.BETA_END, device=config.DEVICE):
        self.num_timesteps = num_timesteps; self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device); self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0); self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x_start, t_indices, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t_indices].view(-1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t_indices].view(-1, 1)
        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise, noise

    def predict_x0_from_noise(self, x_t, t_indices, noise_pred):
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t_indices].view(-1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t_indices].view(-1, 1)
        return (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

    def denoise_step(self, model_output_noise, noisy_points_t, t_idx_scalar):
        # This function should be robust to receiving either a single integer or a tensor for t_idx_scalar
        # as long as predict_x0_from_noise handles indexing correctly.
        pred_x0 = self.predict_x0_from_noise(noisy_points_t, t_idx_scalar, model_output_noise)
        mean_prev_t = self.posterior_mean_coef1[t_idx_scalar] * pred_x0 + self.posterior_mean_coef2[t_idx_scalar] * noisy_points_t
        if t_idx_scalar == 0: return mean_prev_t
        log_variance = self.posterior_log_variance_clipped[t_idx_scalar]
        return mean_prev_t + torch.exp(0.5 * log_variance) * torch.randn_like(noisy_points_t)



if __name__ == '__main__':
    print("--- Testing model_components.py (with Parametric Head Denoiser) ---")
    N_occ_test = 50
    noisy_points_occ_test = torch.randn(N_occ_test, config.POINT_DIM).to(config.DEVICE)
    t_norm_test = torch.rand(1).to(config.DEVICE)
    global_context_feat_test = torch.randn(1, config.EMBED_DIM).to(config.DEVICE)
    semantic_guidance_occ_test = F.one_hot(torch.randint(0, config.NUM_CLASSES, (N_occ_test,)), num_classes=config.NUM_CLASSES).float().to(config.DEVICE)
    plane_dist_guidance_occ_test = torch.randn(N_occ_test, config.PLANE_DIST_DIM).to(config.DEVICE)
    
    denoising_net_test = DenoisingNetwork().to(config.DEVICE)
    
    predicted_noise_test, predicted_plane_params_test = denoising_net_test(noisy_points_occ_test, t_norm_test, global_context_feat_test,
                                                                         semantic_guidance_occ_test, plane_dist_guidance_occ_test)
    
    print(f"Noise prediction output shape: {predicted_noise_test.shape} (Expected: [{N_occ_test}, {config.POINT_DIM}])")
    assert predicted_noise_test.shape == (N_occ_test, config.POINT_DIM)
    
    print(f"Plane params output shape: {predicted_plane_params_test.shape} (Expected: [1, 4])")
    assert predicted_plane_params_test.shape == (1, 4)
    
    normal_norm = torch.linalg.norm(predicted_plane_params_test[0, :3])
    print(f"Norm of predicted plane normal vector: {normal_norm.item():.4f} (Should be ~1.0)")
    assert torch.allclose(normal_norm, torch.tensor(1.0)), "Plane normal should be a unit vector"

    print("--- Finished testing ---")
