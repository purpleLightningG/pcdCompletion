# kitti_completion_dataset.py

import torch
from torch.utils.data import Dataset
import os
import glob

# Attempt to import config.py for PREPROCESSED_DATA_DIR (used in test block)
try:
    import config
except ImportError:
    print("Warning in kitti_completion_dataset.py: config.py not found. Test block might not function as expected.")
    # Define a minimal fallback for the test block if config isn't available
    class FallbackConfigForDatasetTest:
        PREPROCESSED_DATA_DIR = "preprocessed_completion_data" # Default, ensure this exists for test
        DEVICE = torch.device("cpu") # For moving data in test
    config = FallbackConfigForDatasetTest()


class KittiCompletionDataset(Dataset):
    """
    PyTorch Dataset class for loading preprocessed Semantic KITTI completion samples.
    Each sample is expected to be a .pt file containing a dictionary of tensors.
    """
    def __init__(self, preprocessed_data_root_dir, split='train', transform=None):
        """
        Args:
            preprocessed_data_root_dir (str): Path to the root directory where preprocessed
                                              data (e.g., PREPROCESSED_DATA_DIR from config) is stored.
            split (str): Which split to load ("train" or "val").
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.split_dir = os.path.join(preprocessed_data_root_dir, split)
        self.transform = transform

        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Preprocessed data directory for split '{split}' not found at: {self.split_dir}")

        self.sample_files = sorted(glob.glob(os.path.join(self.split_dir, "*.pt")))

        if not self.sample_files:
            print(f"Warning: No '.pt' sample files found in {self.split_dir}. Dataset will be empty.")
        else:
            print(f"Initialized KittiCompletionDataset for split '{split}': Found {len(self.sample_files)} samples in {self.split_dir}")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.sample_files)

    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the preprocessed sample tensors:
                'visible_points', 'visible_labels_one_hot', 'visible_plane_dist',
                'occluded_gt_points', 'occluded_gt_labels_one_hot', 'occluded_gt_plane_dist',
                'metadata' (optional)
        """
        if idx < 0 or idx >= len(self.sample_files):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.sample_files)} samples.")

        sample_file_path = self.sample_files[idx]
        
        try:
            # Load the dictionary of tensors saved by preprocess_dataset.py
            # ADD weights_only=True
            sample = torch.load(sample_file_path, map_location='cpu', weights_only=True) 
        except Exception as e:
            print(f"Error loading sample file {sample_file_path}: {e}")
            raise

        # The sample is already a dictionary of tensors as saved by preprocess_dataset.py
        # Example keys: 'visible_points', 'visible_labels_one_hot', 'visible_plane_dist',
        #               'occluded_gt_points', 'occluded_gt_labels_one_hot', 'occluded_gt_plane_dist',
        #               'metadata'

        if self.transform:
            sample = self.transform(sample) # Apply transforms if any

        return sample


if __name__ == '__main__':
    print("--- Testing kitti_completion_dataset.py ---")

    # This test assumes that preprocess_dataset.py has been run and
    # config.PREPROCESSED_DATA_DIR points to the correct location.
    # It also assumes config.py is available for PREPROCESSED_DATA_DIR.

    if 'FallbackConfigForDatasetTest' in globals() and isinstance(config, FallbackConfigForDatasetTest):
        print("Using fallback config for dataset test. Ensure PREPROCESSED_DATA_DIR is correctly set or exists.")
        # You might need to manually ensure the path config.PREPROCESSED_DATA_DIR + "/train" exists and has .pt files
        # For a robust standalone test, you might create a small dummy preprocessed_data folder.

    # Test loading the training split
    try:
        print(f"\nAttempting to load 'train' split from: {config.PREPROCESSED_DATA_DIR}")
        train_dataset = KittiCompletionDataset(
            preprocessed_data_root_dir=config.PREPROCESSED_DATA_DIR,
            split='train'
        )
        
        if len(train_dataset) > 0:
            print(f"Successfully loaded training dataset with {len(train_dataset)} samples.")
            
            # Get and inspect the first sample
            first_sample = train_dataset[0]
            print("\nFirst sample from training dataset:")
            for key, value in first_sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  '{key}': Tensor of shape {value.shape}, dtype {value.dtype}")
                elif isinstance(value, dict) and key == 'metadata': # Print metadata nicely
                    print(f"  '{key}':")
                    for meta_key, meta_val in value.items():
                        print(f"    '{meta_key}': {meta_val}")
                else:
                    print(f"  '{key}': {type(value)}")

            # Example of moving tensors to device (as would happen in a training loop)
            # device_to_test = config.DEVICE if hasattr(config, 'DEVICE') else torch.device('cpu')
            # print(f"\nMoving first sample tensors to device: {device_to_test}")
            # sample_on_device = {
            #     k: v.to(device_to_test) if isinstance(v, torch.Tensor) else v 
            #     for k, v in first_sample.items()
            # }
            # for key, value in sample_on_device.items():
            #     if isinstance(value, torch.Tensor):
            #         print(f"  '{key}' on device: {value.device}")

        else:
            print("Training dataset is empty. Ensure preprocess_dataset.py was run and created files in the 'train' subfolder.")

    except FileNotFoundError as e:
        print(f"Error during training dataset test: {e}")
        print("Please ensure preprocess_dataset.py has been run successfully and the path in config.PREPROCESSED_DATA_DIR is correct.")
    except Exception as e:
        print(f"An unexpected error occurred during training dataset test: {e}")


    # Test loading the validation split (if it exists)
    val_split_path = os.path.join(config.PREPROCESSED_DATA_DIR, "val")
    if os.path.exists(val_split_path) and os.listdir(val_split_path): # Check if val dir exists and is not empty
        try:
            print(f"\nAttempting to load 'val' split from: {config.PREPROCESSED_DATA_DIR}")
            val_dataset = KittiCompletionDataset(
                preprocessed_data_root_dir=config.PREPROCESSED_DATA_DIR,
                split='val'
            )
            if len(val_dataset) > 0:
                print(f"Successfully loaded validation dataset with {len(val_dataset)} samples.")
                # Get and inspect the first sample from validation
                first_val_sample = val_dataset[0]
                print("\nFirst sample from validation dataset (first key-value pair):")
                first_key = list(first_val_sample.keys())[0]
                first_value = first_val_sample[first_key]
                if isinstance(first_value, torch.Tensor):
                     print(f"  '{first_key}': Tensor of shape {first_value.shape}, dtype {first_value.dtype}")
                else:
                    print(f"  '{first_key}': {type(first_value)}")
            else:
                print("Validation dataset is empty, though the directory exists.")
        except FileNotFoundError as e:
            print(f"Error during validation dataset test: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during validation dataset test: {e}")
    else:
        print(f"\nValidation split directory '{val_split_path}' not found or is empty. Skipping validation dataset test.")

    print("\n--- Finished testing kitti_completion_dataset.py ---")