# view_pcd.py

import open3d as o3d
import os
import glob
import argparse

def visualize_pcd(pcd_path):
    """
    Loads and visualizes a single .pcd file.

    Args:
        pcd_path (str): The path to the .pcd file.
    """
    if not os.path.exists(pcd_path):
        print(f"Error: File not found at {pcd_path}")
        return

    print(f"Loading and visualizing: {pcd_path}")
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
    except Exception as e:
        print(f"Error reading PCD file {pcd_path}: {e}")
        return

    if not pcd.has_points():
        print(f"Warning: No points found in {pcd_path}")
        return

    # Basic visualization
    # You can customize the view options if needed
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Open3D - {os.path.basename(pcd_path)}")
    vis.add_geometry(pcd)
    
    # Set view options (optional)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.15, 0.15, 0.15]) # Dark grey background
    opt.point_size = 1.0
    if not pcd.has_colors():
        opt.default_color = np.asarray([0.7, 0.7, 0.7]) # Default color for PCDs without colors

    print("Press 'ESC' or close the window to continue...")
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="Visualize .pcd point cloud files.")
    parser.add_argument("path", type=str, help="Path to a single .pcd file or a directory containing .pcd files.")
    
    args = parser.parse_args()
    input_path = args.path

    if not os.path.exists(input_path):
        print(f"Error: The specified path does not exist: {input_path}")
        return

    if os.path.isfile(input_path):
        if input_path.lower().endswith(".pcd"):
            visualize_pcd(input_path)
        else:
            print(f"Error: Specified file is not a .pcd file: {input_path}")
    elif os.path.isdir(input_path):
        print(f"Searching for .pcd files in directory: {input_path}")
        pcd_files = sorted(glob.glob(os.path.join(input_path, "*.pcd")))
        
        if not pcd_files:
            print(f"No .pcd files found in {input_path}")
            return
            
        print(f"Found {len(pcd_files)} .pcd files. Visualizing them sequentially:")
        for pcd_file in pcd_files:
            visualize_pcd(pcd_file)
            # Optional: Add a small pause or prompt before showing the next one if desired
            # input("Press Enter to view the next PCD (if any)...")
    else:
        print(f"Error: Path is not a valid file or directory: {input_path}")

if __name__ == "__main__":
    main()