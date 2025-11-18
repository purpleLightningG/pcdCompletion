# pcdCompletion 🚗💨  
**Point Cloud / Depth Completion in PyTorch**

[![Stars](https://img.shields.io/github/stars/purpleLightningG/pcdCompletion?style=social)](https://github.com/purpleLightningG/pcdCompletion/stargazers)
[![Issues](https://img.shields.io/github/issues/purpleLightningG/pcdCompletion)](https://github.com/purpleLightningG/pcdCompletion/issues)
[![License](https://img.shields.io/badge/license-MIT-informational.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#installation)

> End-to-end pipeline for LiDAR point cloud / depth completion in PyTorch, with dataset preprocessing, training, evaluation, and rich visualizations (KITTI-style depth completion).

This repo is meant to be **usable out-of-the-box** for:
- Students / beginners wanting a *clean reference* for point cloud / depth completion.
- Researchers who need a **baseline** they can extend.
- Practitioners who want a **ready pipeline** to train & visualize completion models.

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference & Visualization](#inference--visualization)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Features

- ✅ **PyTorch implementation** of point cloud / depth completion.
- ✅ **Dataset wrapper** for KITTI-style completion datasets (`kitti_completion_dataset.py`).
- ✅ **Modular model components** in `model_components.py`.
- ✅ **Training utilities** with logging & loss tracking (`train_full_scale.py`, `training_utils.py`).
- ✅ **Evaluation scripts** (`evaluate.py`, `generate_curve_from_checkpoints.py`).
- ✅ **Visualization tools** for:
  - Raw & completed point clouds (`view_pcd.py`, `visualization_utils.py`)
  - Side-by-side comparisons (`view_comparison.py`)
  - Training curves (`training_plots/`)
- ✅ Example **completed scans output** (`completed_scans_output/`) for quick inspection.

The code is structured to be **hackable** and **beginner-friendly**, with clear entrypoints for training and inference.

---

## Repository Structure

```text
pcdCompletion/
├── completed_scans_output/      # Example outputs from the completion model
├── training_plots/              # Training curves / logs (for reference)
├── config.py                    # Central configuration (paths, hyperparams, etc.)
├── data_utils.py                # I/O helpers, data formatting, augmentation utilities
├── kitti_completion_dataset.py  # Dataset class for KITTI-style completion
├── model_components.py          # Model architectures / building blocks
├── training_utils.py            # Training loop helpers, metrics, schedulers
├── train_full_scale.py          # Main training script
├── evaluate.py                  # Evaluation on validation / test splits
├── inference_utils.py           # Inference helpers (single scan / batch)
├── main_pipeline.py             # End-to-end pipeline script (load → complete → save/visualize)
├── preprocess_dataset.py        # Dataset preprocessing script (generate sparse/dense data)
├── view_pcd.py                  # Visualize point clouds
├── view_comparison.py           # Side-by-side comparison of input vs completion
├── visualization_utils.py       # Shared visualization helpers
└── README.md
