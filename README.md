# pcdCompletion 🚗💨  
**Point Cloud / Depth Completion in PyTorch**

[![Stars](https://img.shields.io/github/stars/purpleLightningG/pcdCompletion?style=social)](https://github.com/purpleLightningG/pcdCompletion/stargazers)
[![Issues](https://img.shields.io/github/issues/purpleLightningG/pcdCompletion)](https://github.com/purpleLightningG/pcdCompletion/issues)
[![License](https://img.shields.io/badge/license-MIT-informational.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#installation)

End-to-end pipeline for **LiDAR / depth completion** using PyTorch, with dataset utilities, training, evaluation and visualization scripts.

This repo is aimed at:

- 🔰 **Beginners** who want a *clean, working* reference for point cloud / depth completion.
- 🔬 **Researchers** who want a small, hackable baseline.
- 🚗 **CV / AV people** who work with KITTI-style 3D data.

---

## Repository Structure

```text
pcdCompletion/
├── completed_scans_output/          # Example outputs from the completion model
├── training_plots/                  # Training curves / logs (for reference)
├── config.py                        # Central configuration (paths, hyperparams, etc.)
├── data_utils.py                    # I/O helpers, data formatting, augmentation
├── kitti_completion_dataset.py      # Dataset class for KITTI-style completion
├── model_components.py              # Model architectures / building blocks
├── training_utils.py                # Training loop helpers, metrics, schedulers
├── train_full_scale.py              # Main training script
├── evaluate.py                      # Evaluation on validation / test splits
├── generate_curve_from_checkpoints.py # Plot curves from saved checkpoints
├── inference_utils.py               # Inference helpers (single scan / batch)
├── main_pipeline.py                 # End-to-end pipeline: load → complete → save/visualize
├── preprocess_dataset.py            # Dataset preprocessing (sparse / dense preparation)
├── view_pcd.py                      # Visualize point clouds
├── view_comparison.py               # Side-by-side comparison of input vs completion
├── visualization_utils.py           # Shared visualization helpers
└── README.md
