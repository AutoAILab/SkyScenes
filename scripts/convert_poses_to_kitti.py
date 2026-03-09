#!/usr/bin/env python3
import numpy as np
import json
import os
import re
import argparse
from tqdm import tqdm

def parse_carla_transform(transform_str):
    """
    Parses CARLA Transform string into location and rotation dictionaries.
    Example: Transform(Location(x=50.2, y=-12.5, z=35.0), Rotation(pitch=-45.0, yaw=90.0, roll=0.0))
    """
    # Handle both direct Transform string and the one with prefix like 'sensor.camera.rgb(id=18) Transform(...)'
    if 'Transform(' in transform_str:
        transform_str = transform_str[transform_str.find('Transform('):]

    # Regex to extract x, y, z, pitch, yaw, roll
    loc_match = re.search(r'Location\(x=([\d.-]+), y=([\d.-]+), z=([\d.-]+)\)', transform_str)
    rot_match = re.search(r'Rotation\(pitch=([\d.-]+), yaw=([\d.-]+), roll=([\d.-]+)\)', transform_str)
    
    if not loc_match or not rot_match:
        raise ValueError(f"Could not parse transform string: {transform_str}")
    
    return {
        'x': float(loc_match.group(1)),
        'y': float(loc_match.group(2)),
        'z': float(loc_match.group(3)),
        'pitch': float(rot_match.group(1)),
        'yaw': float(rot_match.group(2)),
        'roll': float(rot_match.group(3))
    }

def carla_to_matrix(loc_rot):
    """
    Converts CARLA location/rotation to a 4x4 transformation matrix in CARLA world.
    CARLA LHS: X-Forward, Y-Right, Z-Up.
    Rotation order: Roll (X), Pitch (Y), Yaw (Z) in intrinsic/local coords?
    Actually CARLA uses: R = R_yaw * R_pitch * R_roll
    """
    yaw = np.radians(loc_rot['yaw'])
    pitch = np.radians(loc_rot['pitch'])
    roll = np.radians(loc_rot['roll'])
    
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    
    matrix = np.eye(4)
    matrix[0, 3] = loc_rot['x']
    matrix[1, 3] = loc_rot['y']
    matrix[2, 3] = loc_rot['z']
    
    # Rotation matrix construction (matches CARLA's Transform.get_matrix())
    matrix[0, 0] = cp * cy
    matrix[0, 1] = cy * sp * sr - sy * cr
    matrix[0, 2] = -cy * sp * cr - sy * sr
    matrix[1, 0] = sy * cp
    matrix[1, 1] = sy * sp * sr + cy * cr
    matrix[1, 2] = -sy * sp * cr + cy * sr
    matrix[2, 0] = sp
    matrix[2, 1] = -cp * sr
    matrix[2, 2] = cp * cr
    
    return matrix

def transform_to_kitti(carla_matrix):
    """
    Transforms a CARLA world matrix to a KITTI world matrix.
    CARLA (C): X-Forward, Y-Right, Z-Up
    KITTI (K): Z-Forward, X-Right, Y-Down
    
    Mapping:
    X_k = Y_c
    Y_k = -Z_c
    Z_k = X_c
    """
    # T_C2K transforms a vector from CARLA coords to KITTI coords
    T_C2K = np.array([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    
    # M_k = T_C2K @ M_c @ inv(T_C2K)
    T_K2C = np.linalg.inv(T_C2K)
    kitti_matrix = T_C2K @ carla_matrix @ T_K2C
    return kitti_matrix

def verify_pose(matrix):
    """
    Check if the rotation part is orthonormal.
    """
    R = matrix[:3, :3]
    det = np.linalg.det(R)
    is_ortho = np.allclose(R.T @ R, np.eye(3), atol=1e-4)
    return np.allclose(det, 1.0, atol=1e-4) and is_ortho

def main():
    parser = argparse.ArgumentParser(description="Convert SkyScenes metadata to KITTI Odometry format.")
    parser.add_argument("--input_dir", required=True, help="Path to sequence metaData directory.")
    parser.add_argument("--output_file", required=True, help="Path to output .txt file.")
    parser.add_argument("--sensor_index", type=int, default=0, help="Index of the sensor to use (0 is usually RGB camera).")
    args = parser.parse_args()

    files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.json')])
    if not files:
        print(f"No JSON files found in {args.input_dir}")
        return

    poses_kitti = []

    for filename in tqdm(files, desc="Converting poses"):
        with open(os.path.join(args.input_dir, filename), 'r') as f:
            data = json.load(f)
        
        # Extract sensor transform (usually the first one in "sensors" list is the RGB camera)
        sensor_data = data["sensors"][args.sensor_index]
        # sensor_data is a string with "Actor(...) \n Transform(...)"
        # We need the Transform part
        transform_str = sensor_data.split('\n')[1]
        
        loc_rot = parse_carla_transform(transform_str)
        m_carla = carla_to_matrix(loc_rot)
        m_kitti = transform_to_kitti(m_carla)
        
        if not verify_pose(m_kitti):
            print(f"Warning: Invalid pose detected at {filename}")
        
        # Flatten the 3x4 part (row-major)
        pose_row = m_kitti[:3, :4].reshape(-1)
        poses_kitti.append(pose_row)

    # Save to KITTI format (12 values per line, space-separated)
    np.savetxt(args.output_file, np.array(poses_kitti), fmt='%.8e', delimiter=' ')
    print(f"Saved {len(poses_kitti)} poses to {args.output_file}")

if __name__ == "__main__":
    main()
