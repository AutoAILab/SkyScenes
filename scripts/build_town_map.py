import argparse
import os
import open3d as o3d
import numpy as np

def merge_town_maps(input_dir, output_dir, voxel_size=0.1):
    """
    Finds all town_accumulated.ply files in input_dir, groups them by town name, and merges them.
    """
    print(f"Searching for town_accumulated.ply in {input_dir}...")
    
    town_groups = {} # town_name -> list of ply_paths
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file == "town_accumulated.ply":
                # Town name is usually the folder name: /.../Town10HD/town_accumulated.ply
                town_name = os.path.basename(root)
                if town_name not in town_groups:
                    town_groups[town_name] = []
                town_groups[town_name].append(os.path.join(root, file))
    
    if not town_groups:
        print("No town_accumulated.ply files found!")
        return

    os.makedirs(output_dir, exist_ok=True)

    for town_name, ply_paths in town_groups.items():
        print(f"\nMerging segments for {town_name} ({len(ply_paths)} files)...")
        combined_pcd = o3d.geometry.PointCloud()
        
        for ply_path in ply_paths:
            print(f"  Loading {ply_path}...")
            pcd = o3d.io.read_point_cloud(ply_path)
            combined_pcd += pcd
            combined_pcd = combined_pcd.voxel_down_sample(voxel_size)
            print(f"  Current points: {len(combined_pcd.points)}")

        output_file = os.path.join(output_dir, f"{town_name}_merged.ply")
        print(f"Saving merged map for {town_name} to {output_file}...")
        o3d.io.write_point_cloud(output_file, combined_pcd)
    
    print("\nAll town merges complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatically merge LiDAR segments grouped by town.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory containing generated sequences")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged town maps")
    parser.add_argument("--voxel_size", type=float, default=0.1, help="Voxel size for downsampling")
    
    args = parser.parse_args()
    merge_town_maps(args.input_dir, args.output_dir, args.voxel_size)
