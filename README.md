# pcdCompletion 🚗💨  
**Point Cloud / Depth Completion in PyTorch**

[![Stars](https://img.shields.io/github/stars/purpleLightningG/pcdCompletion?style=social)](https://github.com/purpleLightningG/pcdCompletion/stargazers)
[![Issues](https://img.shields.io/github/issues/purpleLightningG/pcdCompletion)](https://github.com/purpleLightningG/pcdCompletion/issues)
[![License](https://img.shields.io/badge/license-MIT-informational.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#installation)

End-to-end pipeline for **LiDAR / depth completion** using PyTorch, with dataset utilities, training, evaluation and visualization scripts.

This repo is aimed at:

- 🔰 **Beginners** who want a clean, working reference for point cloud / depth completion.  
- 🔬 **Researchers** who want a small, hackable baseline.  
- 🚗 **CV / AV engineers** working with KITTI-style 3D data.

---

## 📁 Repository Structure

```text
pcdCompletion/
├── completed_scans_output/          # Example outputs
├── training_plots/                  # Training curves
├── config.py                        # Central configuration
├── data_utils.py                    # Data loading / formatting utilities
├── kitti_completion_dataset.py      # KITTI-style dataset class
├── model_components.py              # Model building blocks
├── training_utils.py                # Training loop utilities
├── train_full_scale.py              # Main training script
├── evaluate.py                      # Evaluation script
├── generate_curve_from_checkpoints.py  # Plot loss curves
├── inference_utils.py               # Inference helpers
├── main_pipeline.py                 # End-to-end pipeline
├── preprocess_dataset.py            # Dataset preprocessing
├── view_pcd.py                      # Visualize point clouds
├── view_comparison.py               # Compare sparse vs completed output
├── visualization_utils.py           # Visualization helpers
└── README.md
```

---

## ⭐ Features

- ✅ PyTorch implementation of point cloud / depth completion  
- ✅ KITTI-style dataset support  
- ✅ Modular model blocks for easy experimentation  
- ✅ Training + evaluation scripts  
- ✅ Visualization tools using Open3D  
- ✅ Example outputs and training plots included  

---

## ⚙️ Installation

```bash
git clone https://github.com/purpleLightningG/pcdCompletion.git
cd pcdCompletion

# (optional virtual environment)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 📦 Dataset Setup

This code assumes a **KITTI-style depth completion dataset layout** (sparse LiDAR depth + RGB + ground truth).

Steps:

1. Download the dataset (e.g., KITTI depth completion).  
2. Update paths inside `config.py`:  
   - training split  
   - validation / test split  
   - checkpoint/log directories  
3. To adapt to a custom dataset, modify:

```
kitti_completion_dataset.py
```

---

## 🏋️ Training

```bash
python train_full_scale.py
```

This will:

- Load the dataset  
- Build the model from `model_components.py`  
- Use training utilities in `training_utils.py`  
- Save checkpoints + loss curves  

---

## 📊 Evaluation

```bash
python evaluate.py --checkpoint path/to/checkpoint.pth
```

Plot learning curves:

```bash
python generate_curve_from_checkpoints.py \
  --log-dir path/to/checkpoints
```

---

## 🎯 Inference & Visualization

### Run the full pipeline:

```bash
python main_pipeline.py \
  --checkpoint path/to/checkpoint.pth \
  --input-dir path/to/sparse_scans \
  --output-dir completed_scans_output
```

### Visualize a completed scan:

```bash
python view_pcd.py --pcd path/to/scan.pcd
```

### Compare sparse vs completed:

```bash
python view_comparison.py \
  --input path/to/sparse_scan.pcd \
  --completed path/to/completed_scan.pcd
```

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

- Open an issue for bugs, clarifications, or feature requests.  
- Fork → create a branch → open a pull request with:  
  - what you changed  
  - how to reproduce  
  - impact on existing behaviour  

If you’re using this repo to learn depth completion, feel free to open an issue labeled `question`.

---

## 📝 Citation

```text
@misc{pcdCompletion,
  author       = {Shahriar Hossain},
  title        = {pcdCompletion: Point Cloud / Depth Completion in PyTorch},
  year         = {2025},
  howpublished = {\url{https://github.com/purpleLightningG/pcdCompletion}}
}
```

---

## 📜 License

This project is released under the MIT License.
