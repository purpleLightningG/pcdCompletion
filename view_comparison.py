# view_comparison.py
import open3d as o3d
import os
import argparse
import numpy as np

def visualize_completion_comparison(pcd_dir, base_filename_pattern):
    """
    Loads and visualizes a set of related PCD files for completion comparison.
    Example base_filename_pattern: "seq08_scan000000_occ0" (without the _XX_suffix.pcd)
    """
    filenames_map = {
        "original_full": f"{base_filename_pattern}_00_original_full.pcd",
        "visible_input": f"{base_filename_pattern}_01_visible_input.pcd",
        "gt_occluded_part": f"{base_filename_pattern}_02_gt_occluded_part.pcd",
        "gen_occluded_part": f"{base_filename_pattern}_03_gen_occluded_part.pcd",
        "completed_full": f"{base_filename_pattern}_04_completed_full.pcd",
    }

    loaded_pcds = {}
    print(f"Attempting to load files for base: {base_filename_pattern} from directory: {pcd_dir}")

    for key, filename in filenames_map.items():
        path = os.path.join(pcd_dir, filename)
        if os.path.exists(path):
            try:
                pcd = o3d.io.read_point_cloud(path)
                if pcd.has_points():
                    loaded_pcds[key] = pcd
                    print(f"  Loaded: {filename} ({len(pcd.points)} points)")
                else:
                    print(f"  Warning: {filename} is empty or has no points.")
                    loaded_pcds[key] = o3d.geometry.PointCloud() # Empty PCD if file exists but no points
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
        else:
            print(f"  File not found: {path}")

    if not any(pcd.has_points() for pcd in loaded_pcds.values() if pcd is not None): # Check if any valid PCD was loaded
        print("No valid PCDs with points loaded for the given base filename. Nothing to visualize.")
        return

    # Define colors
    colors = {
        "original_full": [0.6, 0.6, 0.6],  # Light Grey
        "visible_input_display": [0.0, 0.5, 1.0], # Bright Blue for visible part when shown with GT/Gen
        "gt_occluded_part_display": [1.0, 0.3, 0.3],  # Bright Red for GT occluded
        "gen_occluded_part_display": [0.3, 1.0, 0.3], # Bright Green for generated occluded
        "completed_full_visible_part": [0.2, 0.2, 0.7], # Darker Blue for visible in final composite
        "completed_full_generated_part": [0.2, 0.7, 0.2], # Darker Green for generated in final composite
        "default_color_if_no_parts": [0.5, 0.0, 0.5] # Purple (e.g., for completed_full if parts unknown)
    }
    
    geometries = []
    current_x_offset = 0.0
    
    # Determine a suitable x_shift based on the original point cloud's extent
    x_shift_increment = 50.0 # Default shift
    if "original_full" in loaded_pcds and loaded_pcds["original_full"].has_points():
        original_bounds = loaded_pcds["original_full"].get_axis_aligned_bounding_box()
        x_extent = original_bounds.get_extent()[0]
        if x_extent > 1.0: # Avoid tiny shifts for very small extents
            x_shift_increment = x_extent * 1.2 
    elif "visible_input" in loaded_pcds and loaded_pcds["visible_input"].has_points():
        vis_bounds = loaded_pcds["visible_input"].get_axis_aligned_bounding_box()
        x_extent = vis_bounds.get_extent()[0]
        if x_extent > 1.0:
            x_shift_increment = x_extent * 1.5 # May need larger shift if only visible is present for scale

    # 1. Original Full
    if "original_full" in loaded_pcds and loaded_pcds["original_full"].has_points():
        pcd1 = o3d.geometry.PointCloud(loaded_pcds["original_full"]) # Make a copy for manipulation
        pcd1.paint_uniform_color(colors["original_full"])
        pcd1.translate([current_x_offset, 0, 0])
        geometries.append(pcd1)
        current_x_offset += x_shift_increment
        
    # 2. Visible Input + GT Occluded Part
    pcd2_combined = o3d.geometry.PointCloud()
    if "visible_input" in loaded_pcds and loaded_pcds["visible_input"].has_points():
        pcd_vis_part = o3d.geometry.PointCloud(loaded_pcds["visible_input"])
        pcd_vis_part.paint_uniform_color(colors["visible_input_display"])
        pcd2_combined += pcd_vis_part
    if "gt_occluded_part" in loaded_pcds and loaded_pcds["gt_occluded_part"].has_points():
        pcd_gt_occ_part = o3d.geometry.PointCloud(loaded_pcds["gt_occluded_part"])
        pcd_gt_occ_part.paint_uniform_color(colors["gt_occluded_part_display"])
        pcd2_combined += pcd_gt_occ_part
    if pcd2_combined.has_points():
        pcd2_combined.translate([current_x_offset, 0, 0])
        geometries.append(pcd2_combined)
        current_x_offset += x_shift_increment

    # 3. Visible Input + Generated Occluded Part
    pcd3_combined = o3d.geometry.PointCloud()
    if "visible_input" in loaded_pcds and loaded_pcds["visible_input"].has_points():
        pcd_vis_part_again = o3d.geometry.PointCloud(loaded_pcds["visible_input"])
        pcd_vis_part_again.paint_uniform_color(colors["visible_input_display"])
        pcd3_combined += pcd_vis_part_again
    if "gen_occluded_part" in loaded_pcds and loaded_pcds["gen_occluded_part"].has_points():
        pcd_gen_occ_part = o3d.geometry.PointCloud(loaded_pcds["gen_occluded_part"])
        pcd_gen_occ_part.paint_uniform_color(colors["gen_occluded_part_display"])
        pcd3_combined += pcd_gen_occ_part
    if pcd3_combined.has_points():
        pcd3_combined.translate([current_x_offset, 0, 0])
        geometries.append(pcd3_combined)
        current_x_offset += x_shift_increment

    # 4. Completed Full (with differentiated colors for visible and generated parts)
    if "completed_full" in loaded_pcds and loaded_pcds["completed_full"].has_points():
        pcd4_completed_full = o3d.geometry.PointCloud(loaded_pcds["completed_full"])
        # Try to color visible and generated parts differently
        num_vis_pts = len(loaded_pcds["visible_input"].points) if "visible_input" in loaded_pcds and loaded_pcds["visible_input"].has_points() else 0
        num_gen_pts = len(loaded_pcds["gen_occluded_part"].points) if "gen_occluded_part" in loaded_pcds and loaded_pcds["gen_occluded_part"].has_points() else 0
        
        if num_vis_pts > 0 and num_gen_pts > 0 and len(pcd4_completed_full.points) == (num_vis_pts + num_gen_pts):
            comp_colors = np.zeros((len(pcd4_completed_full.points), 3))
            comp_colors[:num_vis_pts] = colors["completed_full_visible_part"]
            comp_colors[num_vis_pts:] = colors["completed_full_generated_part"]
            pcd4_completed_full.colors = o3d.utility.Vector3dVector(comp_colors)
        else:
            pcd4_completed_full.paint_uniform_color(colors["default_color_if_no_parts"]) # Fallback color

        pcd4_completed_full.translate([current_x_offset, 0, 0])
        geometries.append(pcd4_completed_full)

    if geometries:
        print("\nDisplaying comparison. Press 'ESC' or close window.")
        print("Layout (left to right, if all files found):")
        print("1. Original Full (Grey)")
        print("2. Visible (Blue) + GT Occluded (Red)")
        print("3. Visible (Blue) + Generated Occluded (Green)")
        print("4. Final Completed (Visible Dark Blue, Generated Dark Green / or Purple)")
        o3d.visualization.draw_geometries(geometries, window_name=f"Completion Comparison: {base_filename_pattern}")
    else:
        print("No valid PCDs were loaded to visualize for the specified base filename.")


def main():
    parser = argparse.ArgumentParser(description="Visualize a set of related point cloud completion files.")
    parser.add_argument("pcd_dir", type=str, 
                        help="Directory where the set of related .pcd files are stored (e.g., EVALUATION_PCD_OUTPUT_DIR).")
    parser.add_argument("base_filename", type=str, 
                        help="The base filename pattern (e.g., 'seq08_scan000000_occ0') without the _XX_suffix.pcd part.")
    
    args = parser.parse_args()

    if not os.path.isdir(args.pcd_dir):
        print(f"Error: PCD directory not found: {args.pcd_dir}")
        return

    visualize_completion_comparison(args.pcd_dir, args.base_filename)

if __name__ == "__main__":
    main()