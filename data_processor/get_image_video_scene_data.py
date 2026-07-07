import torch
import collections
import numpy as np
import torchvision.transforms.functional as F
from tqdm import tqdm
import open3d as o3d
import matplotlib.pyplot as plt
import math
import random
from einops import einsum
from PIL import Image
import cv2
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

import numpy as np
from scipy.spatial import cKDTree
import heapq
from scipy.ndimage import distance_transform_edt
import numpy as np
import cv2
import torch
from collections import defaultdict
from scipy.spatial import cKDTree
import os


def calibrate_scene_orientation(pcd_scene, pcd_ground, target_axis=np.array([0, 0, 1])):
    """
    Calibrate scene orientation based on the ground point cloud.
    Improvement: Added anti-rollover logic. When the angle between the detected plane 
    normal vector and the Z-axis is too large (recognized as a wall), abandon rotation 
    correction and only perform Z-axis height alignment.
    """
    print("Calibrating scene coordinate system based on navigable area...")
    
    if len(pcd_ground.points) < 10:
        print("Warning: Too few ground points, calibration cannot be performed.")
        return pcd_scene, pcd_ground

    # 1. RANSAC plane fitting
    plane_model, inliers = pcd_ground.segment_plane(distance_threshold=0.02,
                                                    ransac_n=3,
                                                    num_iterations=1000)
    [a, b, c, d] = plane_model
    current_normal = np.array([a, b, c])
    
    # Normalize the normal vector
    norm_val = np.linalg.norm(current_normal)
    if norm_val == 0:
        return pcd_scene, pcd_ground
    current_normal = current_normal / norm_val

    # 2. Ensure the normal vector points upwards (assuming original Z is roughly upwards)
    if np.dot(current_normal, target_axis) < 0:
        current_normal = -current_normal
        d = -d # The plane equation coefficient must also be inverted

    print(f"Detected RANSAC plane normal: {current_normal}")

    # --- [Key Modification] Safety check: Prevent recognizing walls as ground ---
    # Calculate the cosine of the angle between the current normal vector and the target axis (Z-axis)
    # dot_product = 1 means completely parallel, 0 means perpendicular
    alignment_score = np.abs(np.dot(current_normal, target_axis))
    
    # Threshold setting: cos(30 degrees) ≈ 0.866, cos(45 degrees) ≈ 0.707
    # If the similarity is less than 0.8, it indicates a deviation of more than 36 degrees from the Z-axis, highly likely fitting to a wall or slope
    is_valid_ground = alignment_score > 0.8

    T = np.eye(4)
    
    if is_valid_ground:
        print(f"  -> Normal vector determined as valid ground (Alignment Score: {alignment_score:.3f}), performing rotation correction.")
        # 3. Calculate rotation matrix (rotate current_normal to target_axis)
        v = np.cross(current_normal, target_axis)
        c_val = np.dot(current_normal, target_axis)
        
        if np.linalg.norm(v) < 1e-6:
            R = np.eye(3)
        else:
            k = v / np.linalg.norm(v)
            theta = math.acos(np.clip(c_val, -1.0, 1.0))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(k * theta)
        
        T[:3, :3] = R
    else:
        print(f"  -> [Warning] The detected plane might be a wall (Alignment Score: {alignment_score:.3f} < 0.8).")
        print("  -> Skipping rotation correction, only performing translation alignment, trusting original Z-axis direction.")
        # Keep the rotation matrix as identity matrix
        R = np.eye(3)

    # Apply rotation (if any)
    pcd_scene.transform(T)
    pcd_ground.transform(T)

    # 4. Calculate translation (pull ground to Z=0)
    # If we trust the RANSAC plane (is_valid_ground=True), we can precisely calibrate Z using the plane equation
    # If we don't trust the plane (might be a wall), we roughly calibrate Z using the centroid or percentiles of the ground point cloud
    
    translation = np.zeros(3)
    if is_valid_ground:
        # After rotation, the normal vector becomes [0,0,1], and the plane equation becomes z + new_d = 0 => z = -new_d
        # We retake the center point of the ground to be more robust
        ground_center = pcd_ground.get_center()
        translation = np.array([0, 0, -ground_center[2]])
    else:
        # If what was fitted was a wall, since there's no rotation, we cannot use the previous d.
        # Directly count the Z values of the ground point cloud, take the median or low percentile as the ground height
        points = np.asarray(pcd_ground.points)
        if len(points) > 0:
            # Take the median of the Z-axis as the ground height
            z_median = np.median(points[:, 2])
            translation = np.array([0, 0, -z_median])

    pcd_scene.translate(translation)
    pcd_ground.translate(translation)
    
    print("Scene calibration completed.")
    return pcd_scene, pcd_ground


class NavigableAreaSegmenter:
    def __init__(self):
        print("Loading Mask2Former segmentation model...")
        self.processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.navigable_ids = [
            3,   # Floor
            9,   # Rug
            11,  # Sidewalk
            12,  # Earth
            13,  # Road
            29,  # Path
            94,  # Step
        ]
        self.door_ids = [14]
        print("Segmentation model loaded successfully.")

    def predict(self, rgb_image: np.ndarray, depth_map: np.ndarray = None):
        """
        Returns: (navigable_mask, door_mask)
        """
        if rgb_image is None:
            return None, None
            
        h, w = rgb_image.shape[:2]
        image_pil = Image.fromarray(rgb_image)
        inputs = self.processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        prediction = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(h, w)]
        )[0] 
        segmentation = prediction.cpu().numpy()
        
        # 1. Ground Mask (General)
        navigable_mask = np.isin(segmentation, self.navigable_ids).astype(np.uint8) * 255
        
        # 2. Door Mask (Specifically for opening doors in Structured3D)
        door_mask = np.isin(segmentation, self.door_ids).astype(np.uint8) * 255

        if depth_map is not None:
            if depth_map.shape[:2] != (h, w):
                depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_NEAREST)
            valid_depth = (depth_map > 0.2) & (depth_map < 10.0)
            valid_mask = valid_depth.astype(np.uint8)
            
            navigable_mask = cv2.bitwise_and(navigable_mask, navigable_mask, mask=valid_mask)
            door_mask = cv2.bitwise_and(door_mask, door_mask, mask=valid_mask)

        return self._post_process_mask(navigable_mask), self._post_process_mask(door_mask)

    def _post_process_mask(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask
    

def clean_navigable_cloud_advanced(pcd, 
                                   height_threshold=0.05, 
                                   obstacle_buffer=0.2):
    """
    Navigable area cleaning.
    """
    if len(pcd.points) < 100:
        return pcd

    print(f"Cleaning point cloud (original points: {len(pcd.points)})...")
    
    pcd_proc = pcd.voxel_down_sample(voxel_size=0.05)
    pcd_proc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    normals = np.asarray(pcd_proc.normals)
    abs_z = np.abs(normals[:, 2])
    
    horizontal_mask = abs_z > 0.5
    horizontal_indices = np.where(horizontal_mask)[0]
    pcd_horizontal = pcd_proc.select_by_index(horizontal_indices)
    
    print(f"  -> Normal filtering: Removed {len(pcd_proc.points) - len(horizontal_indices)} suspected wall points.")

    if len(pcd_horizontal.points) < 10:
        print("Warning: No ground points left after filtering. Calibration might have failed severely or the scene has no ground.")
        return pcd

    # RANSAC ground plane fitting (only fit within horizontal points)
    plane_model, inliers = pcd_horizontal.segment_plane(distance_threshold=0.03,
                                                        ransac_n=3,
                                                        num_iterations=2000)
    [a, b, c, d] = plane_model
    print(f"  -> Fitted ground plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    

    if abs(c) < 0.5:
        print("  -> [Error] Still fitted to a wall! Abandoning RANSAC result, forcing Z=0 plane assumption.")
        a, b, c, d = 0, 0, 1, 0 
        
    points_all = np.asarray(pcd.points)
    distances = (points_all[:, 0] * a + points_all[:, 1] * b + points_all[:, 2] * c + d)
    
    is_ground = np.abs(distances) < height_threshold
    is_obstacle = np.abs(distances) > height_threshold
    
    obstacle_indices = np.where(is_obstacle)[0]
    
    if len(obstacle_indices) == 0:
        return pcd.select_by_index(np.where(is_ground)[0])


    print(f"  -> Originally detected obstacle points: {len(obstacle_indices)}")
    
    obs_pcd_temp = pcd.select_by_index(obstacle_indices)
    
    _, valid_obs_mask = obs_pcd_temp.remove_radius_outlier(nb_points=8, radius=0.10)
    
    if len(valid_obs_mask) > 0:
        valid_indices_local = np.asarray(valid_obs_mask)
        filtered_obstacle_indices = obstacle_indices[valid_indices_local]
    else:
        filtered_obstacle_indices = []
        
    print(f"  -> Valid solid obstacle points used for removing ground after filtering noise: {len(filtered_obstacle_indices)}")
    
    if len(filtered_obstacle_indices) == 0:
        return pcd.select_by_index(np.where(is_ground)[0])

    print(f"  -> Removing ground noise near obstacles...")
    
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points_to_remove = set(obstacle_indices)
    
    # Only use "real obstacles" as centers to search for ground to delete
    check_points = points_all[filtered_obstacle_indices]
    
    # Downsample query points to accelerate
    if len(check_points) > 2000:
        idx_sample = np.random.choice(len(check_points), 2000, replace=False)
        check_points = check_points[idx_sample]

    for pt in check_points:
        # Search all points within the radius and mark them for deletion
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pt, obstacle_buffer)
        if k > 0:
            points_to_remove.update(idx)
            
    # Generate final result
    all_indices = set(range(len(points_all)))
    keep_indices = list(all_indices - points_to_remove)
    
    final_indices = np.array(keep_indices).astype(np.int32)
    final_points = points_all[final_indices]
    
    final_dists = (final_points[:, 0] * a + final_points[:, 1] * b + final_points[:, 2] * c + d)
    strict_ground_mask = np.abs(final_dists) < height_threshold
    
    real_final_indices = final_indices[strict_ground_mask]
    
    result_pcd = pcd.select_by_index(real_final_indices)
    print(f"Cleaning completed, remaining points: {len(result_pcd.points)}")
    return result_pcd


def validate_scene_quality(pcd, voxel_size=0.05, min_area=5):
    """
    Check scene quality.
    
    Args:
        pcd: The pcd here should be the ground point cloud processed by clean_navigable_cloud_advanced
        voxel_size: Voxel size of the point cloud (meters), used for estimating area
        min_area: Minimum allowed ground area (square meters)
        
    Returns:
        is_valid (bool): Whether to keep this scene
        largest_cluster_pcd (PointCloud): Main ground after filtering fragments (returns None if invalid)
        area (float): Estimated area
    """
    if pcd is None or len(pcd.points) < 50:
        return False

    # 1. Ensure uniform point cloud distribution (based on voxel downsampling)
    # This step is crucial for area estimation, ensuring each point represents a similar physical area
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    if len(pcd_down.points) < 10:
        return False

    # 2. Connectivity analysis (DBSCAN)
    # eps: Clustering radius. Set to 2-3 times voxel_size to ensure adjacent voxels can connect
    # min_points: Minimum number of points to form a cluster
    labels = np.array(pcd_down.cluster_dbscan(eps=voxel_size * 2.5, min_points=10, print_progress=False))
    
    if len(labels) == 0 or labels.max() < 0:
        return False

    # 3. Extract the largest cluster (Main Floor)
    # Count the occurrences of each label (ignoring -1 noise)
    counts = np.bincount(labels[labels >= 0])
    largest_cluster_idx = np.argmax(counts)
    
    # Extract point indices belonging to the largest cluster
    valid_indices = np.where(labels == largest_cluster_idx)[0]
    
    # 4. Calculate area
    # Approximate area = number of points * (voxel edge length ^ 2)
    # This is a rough estimate, but highly effective after uniform voxelization
    num_points = len(valid_indices)
    estimated_area = num_points * (voxel_size ** 2)
    
    print(f"  -> The largest connected ground contains {num_points} points, estimated area: {estimated_area:.2f} m^2")

    if estimated_area < min_area:
        print(f"  -> [Rejected] Area is too small (< {min_area} m^2)")
        return False
    else:
        # Create a point cloud containing only the main ground
        return True


def fill_navigable_area(pcd_ground, pcd_all, grid_size=0.05, max_fill_gap=2.0):
    """
    Adaptive filling algorithm (prevents noise from digging holes).
    """
    pts_ground = np.asarray(pcd_ground.points)
    pts_all = np.asarray(pcd_all.points)

    if len(pts_ground) < 10:
        return pcd_ground

    print(f"Adaptively filling ground (Grid: {grid_size}m, Adaptive Plane)...")

    # 1. Dynamically fit current ground plane
    try:
        plane_model, inliers = pcd_ground.segment_plane(distance_threshold=0.03,
                                                        ransac_n=3,
                                                        num_iterations=500)
        [a, b, c, d] = plane_model
        if c < 0:
            a, b, c, d = -a, -b, -c, -d
    except Exception as e:
        print(f"  -> Plane fitting failed: {e}")
        a, b, c, d = 0, 0, 1, -np.median(pts_ground[:, 2])

    # 2. Identify obstacles
    dists_all = (pts_all[:, 0] * a + pts_all[:, 1] * b + pts_all[:, 2] * c + d)
    min_h = 0.1
    max_h = 1.8
    is_obstacle = (dists_all > min_h) & (dists_all < max_h)
    pts_obs = pts_all[is_obstacle]

    # 3. Build 2D projection grid
    all_xy_points = np.vstack((pts_ground, pts_obs))
    min_x, min_y = np.min(all_xy_points[:, :2], axis=0)
    max_x, max_y = np.max(all_xy_points[:, :2], axis=0)
    
    padding = max_fill_gap * 2
    min_x -= padding; min_y -= padding
    max_x += padding; max_y += padding 

    w = int((max_x - min_x) / grid_size)
    h = int((max_y - min_y) / grid_size)
    
    def to_grid(pts):
        us = ((pts[:, 0] - min_x) / grid_size).astype(int)
        vs = ((pts[:, 1] - min_y) / grid_size).astype(int)
        valid = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
        return us[valid], vs[valid]

    # Map 1: Ground
    grid_ground = np.zeros((h, w), dtype=np.uint8)
    us_g, vs_g = to_grid(pts_ground)
    grid_ground[vs_g, us_g] = 255
    
    # Map 2: Obstacles
    grid_obs = np.zeros((h, w), dtype=np.uint8)
    if len(pts_obs) > 0:
        us_o, vs_o = to_grid(pts_obs)
        grid_obs[vs_o, us_o] = 255

    # --- Obstacle grid denoising ---
    # Use connected components analysis to remove extremely small obstacle spots (e.g., < 4 pixels)
    # 4 pixels in a 0.05m grid is about 100cm^2 (10cm x 10cm)
    min_obstacle_pixels = 8 
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(grid_obs, connectivity=8)
    
    clean_grid_obs = np.zeros_like(grid_obs)
    for i in range(1, num_labels): # 0 是背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_obstacle_pixels:
            clean_grid_obs[labels == i] = 255
            
    grid_obs = clean_grid_obs

    # 4. Morphological processing
    # 4.1 Dilate obstacles (establish restricted areas)
    obs_buffer = 0.15 
    k_obs_size = int(obs_buffer / grid_size) * 2 + 1
    kernel_obs = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_obs_size, k_obs_size))
    grid_obs_dilated = cv2.dilate(grid_obs, kernel_obs)

    # 4.2 Close operation on ground (fill holes)
    k_fill_size = int(max_fill_gap / grid_size)
    if k_fill_size % 2 == 0: k_fill_size += 1
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_fill_size, k_fill_size))
    grid_ground_closed = cv2.morphologyEx(grid_ground, cv2.MORPH_CLOSE, kernel_close)

    # 5. Generate filling regions
    fill_mask = cv2.bitwise_and(grid_ground_closed, cv2.bitwise_not(grid_obs_dilated))
    new_mask = cv2.bitwise_and(fill_mask, cv2.bitwise_not(grid_ground))

    # 6. Back-project into 3D space
    v_new, u_new = np.where(new_mask > 0)
    if len(v_new) == 0:
        return pcd_ground

    x_new = u_new * grid_size + min_x + grid_size/2
    y_new = v_new * grid_size + min_y + grid_size/2
    
    # Prefer using the plane equation to calculate Z
    if abs(c) < 0.1:
        z_new = np.full_like(x_new, np.median(pts_ground[:, 2]))
    else:
        z_new = -(a * x_new + b * y_new + d) / c

    new_points = np.vstack((x_new, y_new, z_new)).T
    new_colors = np.tile([0.5, 0.5, 0.5], (len(new_points), 1))

    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(new_points)
    pcd_new.colors = o3d.utility.Vector3dVector(new_colors)

    print(f"  -> Adaptive filling completed, newly added points: {len(new_points)}")
    return pcd_ground + pcd_new

def remove_floating_chunks(pcd, min_cluster_points=200):
    # eps: Clustering neighborhood distance. If voxel_size is 0.01, eps set to 0.03 is appropriate
    labels = np.array(pcd.cluster_dbscan(eps=0.03, min_points=5))
    
    if len(labels) == 0: return pcd
    
    # Count the number of points in each cluster
    counts = np.bincount(labels[labels >= 0])
    if len(counts) == 0: return pcd
    
    # Only keep clusters larger than the threshold
    valid_labels = np.where(counts > min_cluster_points)[0]
    mask = np.isin(labels, valid_labels)
    
    return pcd.select_by_index(np.where(mask)[0])


# Initialize global segmenter
segmenter = NavigableAreaSegmenter()

def normalize(vector):
    return vector / np.linalg.norm(vector)

def parse_camera_info(camera_info, height, width):
    """ extract intrinsic and extrinsic matrix
    """
    lookat = normalize(camera_info[3:6])
    up = normalize(camera_info[6:9])

    W = lookat
    U = np.cross(W, up)
    V = np.cross(W, U)

    rot = np.vstack((U, V, W))
    trans = camera_info[:3] / 1000.

    xfov = camera_info[9]
    yfov = camera_info[10]

    K = np.diag([1, 1, 1])

    K[0, 2] = width / 2
    K[1, 2] = height / 2

    K[0, 0] = K[0, 2] / np.tan(xfov)
    K[1, 1] = K[1, 2] / np.tan(yfov)

    return rot, trans, K



def create_pcd_from_panorama(color_img, depth_img, camera_center, depth_scale=1000.0):
    """
    Generate 3D point cloud or Mesh based on panorama RGB and Depth.
    
    Args:
        color_img: RGB image (H, W, 3)
        depth_img: Depth image (H, W)
        camera_center: Camera position in world coordinates [x, y, z]
        depth_scale: Scaling factor for depth map values (Structured3D is usually mm, so divide by 1000 to convert to meters)
        mesh_strategy: 'point_cloud' or 'mesh' (triangular mesh)
    """
    H, W = depth_img.shape
    
    # 1. Construct longitude and latitude grid (Spherical Coordinates)
    # The definition here depends on the projection method of the panorama. Structured3D uses standard Equirectangular.
    # u corresponds to longitude (lon), v corresponds to latitude (lat)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Normalize pixel coordinates and map to radians
    # lon: [-pi, pi], lat: [pi/2, -pi/2] (Note the Y-axis direction, usually latitude changes from positive to negative from top to bottom of the image)
    lon = (u / W) * 2 * np.pi - np.pi
    lat = -((v / H) * np.pi - (np.pi / 2)) 

    # 2. Convert depth values to distances
    # Assuming depth_img stores the Euclidean distance from the camera to the point
    # Note: If depth is a Z-buffer (vertical depth), the formula needs to be adjusted. Panoramas usually use Euclidean distance.
    dist = depth_img.astype(np.float32) / depth_scale
    
    # Filter invalid depths (e.g., 0 or too far)
    mask = (dist > 0.01) & (dist < 100.0)
    
    # 3. Spherical coordinates to Cartesian coordinates
    # Coordinate system assumption: In Structured3D, the Z-axis is upwards (inferred from your original code delta_height = [0,0, h])
    # Standard mathematical derivation:
    # x = r * cos(lat) * cos(lon) (or sin, depend on forward axis)
    # y = r * cos(lat) * sin(lon)
    # z = r * sin(lat)
    
    # The mapping here needs to be adjusted according to the specific orientation of the dataset.
    # Usually, the center of the panorama (W/2) is directly in front.
    x = dist * np.cos(lat) * np.sin(lon)
    y = dist * np.cos(lat) * np.cos(lon)
    z = dist * np.sin(lat)
    
    # Stack x, y, z
    points = np.stack([x, y, z], axis=-1) # (H, W, 3)
    
    # Add camera center offset, convert back to world coordinates
    points += camera_center / depth_scale
    
    # Flatten arrays for Open3D processing
    flat_points = points[mask]
    flat_colors = color_img[mask].astype(np.float32) / 255.0


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(flat_points)
    pcd.colors = o3d.utility.Vector3dVector(flat_colors)
    return pcd


def process_realsee_scene(data_root, output_dir, segmenter):
    viewpoints_dir = os.path.join(data_root, "viewpoints")
    if not os.path.exists(viewpoints_dir):
        print(f"Error: {viewpoints_dir} not found.")
        return

    viewpoints = sorted([d for d in os.listdir(viewpoints_dir) if os.path.isdir(os.path.join(viewpoints_dir, d))])
    print(f"Found {len(viewpoints)} viewpoints in {data_root}.")

    # Use lists to temporarily store point clouds of all viewpoints
    scene_points_all = []
    scene_colors_all = []
    
    scene_points_nav = []
    scene_colors_nav = []

    for vp_id in tqdm(viewpoints, desc="Processing Viewpoints"):
        vp_dir = os.path.join(viewpoints_dir, vp_id)
        
        depth_path = os.path.join(vp_dir, "depth_image.png")
        scale_path = os.path.join(vp_dir, "depth_scale.txt")
        color_path = os.path.join(vp_dir, "panoImage_1600.jpg")
        mask_path = os.path.join(vp_dir, "pano_mask.png")
        ext_path = os.path.join(vp_dir, "extrinsics.txt")
        
        if not (os.path.exists(depth_path) and os.path.exists(scale_path) and 
                os.path.exists(color_path) and os.path.exists(ext_path)):
            continue

        try:
            with open(ext_path, 'r') as f:
                ext_vals = [float(x) for x in f.read().strip().split()]
                ext = np.array(ext_vals).reshape(4, 4)
            
            with open(scale_path, 'r') as f:
                depth_scale = float(f.read().strip())
                
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) # uint16
            color_img = cv2.imread(color_path, cv2.IMREAD_COLOR) # BGR
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) # RGB
        except Exception as e:
            print(f"Read error {vp_id}: {e}")
            continue

        h, w = depth_img.shape
        if color_img.shape[:2] != (h, w):
            color_img = cv2.resize(color_img, (w, h), interpolation=cv2.INTER_LINEAR)

        original_mask = None
        if os.path.exists(mask_path):
            original_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if len(original_mask.shape) == 3: original_mask = original_mask[:, :, 0]
            if original_mask.shape[:2] != (h, w):
                original_mask = cv2.resize(original_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        navigable_mask, door_mask = segmenter.predict(color_img)
        
        full_nav_mask = np.logical_or(navigable_mask > 0, door_mask > 0).astype(np.uint8) * 255

        iy, ix = np.indices((h, w))
        
        # Spherical coords
        yaw = (iy.astype(np.float32) / h - 0.5) * np.pi          # Lat: [-pi/2, pi/2]
        pitch = (ix.astype(np.float32) / w - 0.5) * 2.0 * np.pi  # Lon: [-pi, pi]
        
        # Filter 1: Angle limits (Cut poles)
        limit = 30 * np.pi / 180.0
        valid_geo = (yaw >= (-np.pi/2 + limit)) & (yaw <= (np.pi/2 - limit))
        
        # Filter 2: Valid Depth
        dist = depth_img.astype(np.float32) / depth_scale
        valid_dist = (depth_img > 0) & (dist >= 0.4) & (dist < 10.0)
        
        # Filter 3: Original Mask
        valid_pixel = np.ones_like(valid_geo, dtype=bool)
        if original_mask is not None:
            valid_pixel = (original_mask == 255)
            
        mask_all = valid_geo & valid_dist & valid_pixel
        
        mask_nav = mask_all & (full_nav_mask > 0)

        def project_points(mask_in):
            if np.sum(mask_in) == 0:
                return None, None
            
            y_s = yaw[mask_in]
            p_s = pitch[mask_in]
            d_s = dist[mask_in]
            c_s = color_img[mask_in] # RGB
            
            # Local Cartesian (RealSee definition: Y is Up/Down in local spherical?)
            # x_local = dist * cos(yaw) * sin(pitch)
            # y_local = dist * sin(yaw)
            # z_local = dist * cos(yaw) * cos(pitch)
            
            cos_y = np.cos(y_s)
            sin_y = np.sin(y_s)
            cos_p = np.cos(p_s)
            sin_p = np.sin(p_s)
            
            x_local = d_s * cos_y * sin_p
            y_local = d_s * sin_y
            z_local = d_s * cos_y * cos_p
            
            # Transform to World
            pts_local = np.stack([x_local, y_local, z_local, np.ones_like(x_local)], axis=1)
            pts_world = pts_local @ ext.T
            pts_world = pts_world[:, :3]
            
            # Coordinate Swap (RealSee Specific)
            # (x, y, z) -> (x, z, -y)
            pts_world_swapped = np.stack([
                pts_world[:, 0],
                pts_world[:, 2],
                -pts_world[:, 1]
            ], axis=1)
            
            return pts_world_swapped, c_s.astype(np.float32) / 255.0

        pts_all, cols_all = project_points(mask_all)
        if pts_all is not None:
            scene_points_all.append(pts_all)
            scene_colors_all.append(cols_all)
            
        pts_nav, cols_nav = project_points(mask_nav)
        if pts_nav is not None:
            scene_points_nav.append(pts_nav)
            scene_colors_nav.append(cols_nav)

    if not scene_points_all:
        print(f"Warning: No valid points for {data_root}")
        return

    print("Merging point clouds...")
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(np.concatenate(scene_points_all, axis=0))
    pcd_all.colors = o3d.utility.Vector3dVector(np.concatenate(scene_colors_all, axis=0))
    
    pcd_nav = o3d.geometry.PointCloud()
    if scene_points_nav:
        pcd_nav.points = o3d.utility.Vector3dVector(np.concatenate(scene_points_nav, axis=0))
        pcd_nav.colors = o3d.utility.Vector3dVector(np.concatenate(scene_colors_nav, axis=0))

    print("Post-processing Full Scene...")
    pcd_all = pcd_all.voxel_down_sample(voxel_size=0.01) 
    # pcd_all = remove_floating_chunks(pcd_all, min_cluster_points=100)
    
    print("Post-processing Navigable Scene...")
    pcd_nav = pcd_nav.voxel_down_sample(voxel_size=0.05)
    
    pcd_nav = fill_navigable_area(pcd_nav, pcd_all)
    
    pcd_nav = clean_navigable_cloud_advanced(pcd_nav, obstacle_buffer=0.3)
    
    if not validate_scene_quality(pcd_nav, min_area=5.0):
        print(f"Scene {os.path.basename(data_root)} rejected due to low quality navigable area.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    scene_name = os.path.basename(data_root.rstrip('/'))
    save_path_all = os.path.join(output_dir, f"{scene_name}.pcd")
    save_path_nav = os.path.join(output_dir, f"{scene_name}_navigable.pcd")
    
    #o3d.visualization.draw_geometries([pcd_all], window_name="Full Scene RGB PointCloud")
    #o3d.visualization.draw_geometries([pcd_nav], window_name="Navigable Area PointCloud (Voxel Filtered)")

    print(f"Saving to {save_path_all} ...")
    o3d.io.write_point_cloud(save_path_all, pcd_all, write_ascii=False, compressed=True)
    o3d.io.write_point_cloud(save_path_nav, pcd_nav, write_ascii=False, compressed=True)
    print("Done.")

RealSee3D_path = "data/scene_datasets/RealSee3D/real_world_data"
all_scenes = sorted([
        os.path.join(RealSee3D_path, d) for d in os.listdir(RealSee3D_path) 
        if os.path.isdir(os.path.join(RealSee3D_path, d))
    ])
RealSee3D_path = "data/scene_datasets/RealSee3D/synthetic_data"
all_scenes += sorted([
        os.path.join(RealSee3D_path, d) for d in os.listdir(RealSee3D_path) 
        if os.path.isdir(os.path.join(RealSee3D_path, d))
    ])
    
output_dir = "data/nav_map/RealSee3D"
for scene_path in tqdm(all_scenes, desc="Total Progress"):
    if os.path.isdir(os.path.join(scene_path, "viewpoints")) and not os.path.exists(os.path.join(output_dir, f"{os.path.basename(scene_path.rstrip('/'))}.pcd")):
        process_realsee_scene(scene_path, output_dir, segmenter)


scene_list = []
for i in range(3500):
    scene_id = 'data/scene_datasets/Structured3D/Structured3D/scene_'+str(i).rjust(5, "0")+'/2D_rendering' 
    scene_list.append(scene_id)

for scene_id in tqdm(scene_list):
    room_list = [scene_id+'/'+item+'/perspective/full' for item in os.listdir(scene_id)]
    image_list = []
    for i in room_list:
        if os.path.exists(i):
            for j in os.listdir(i):
                image_list.append(i+'/'+j)

    pcd_all = o3d.geometry.PointCloud() 
    pcd_navigable = o3d.geometry.PointCloud() 
    pano_id = None
    for image_id in image_list:
        try:
            if "/".join(image_id.split("/")[:-1]) == pano_id:
                pano_id = None
            else:
                pano_id = "/".join(image_id.split("/")[:-1])

            camera_info = np.loadtxt(os.path.join(image_id, 'camera_pose.txt'))
            rot, trans, intrinsic = parse_camera_info(camera_info,720, 1280)
            extrinsic = np.eye(4)
            extrinsic[:3,:3] = rot
            extrinsic = np.linalg.inv(extrinsic)
            extrinsic[:3,3:4] = trans.reshape(3,1)
            R = extrinsic[:3,:3]
            T = trans.reshape(3,1)

            color_path = image_id + '/rgb_rawlight.png'
            depth_path = image_id + '/depth.png'
            semantic_path = image_id + '/semantic.png'

            color_cv = cv2.imread(color_path)
            color_cv = cv2.cvtColor(color_cv, cv2.COLOR_BGR2RGB)
            depth_cv = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) # uint16

            semantic_cv = cv2.imread(semantic_path)
            semantic_cv = cv2.cvtColor(semantic_cv, cv2.COLOR_BGR2RGB)

            diff = semantic_cv.astype(np.float32) - np.array([152, 223, 138]).astype(np.float32)
            gt_floor_mask = (np.sum(np.abs(diff), axis=-1) < 5).astype(np.uint8) * 255
            
            depth_meters = depth_cv.astype(np.float32) / 1000.0
            _, pred_door_mask = segmenter.predict(color_cv, depth_meters)
            
            if pred_door_mask is not None:
                kernel_door = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                pred_door_mask = cv2.dilate(pred_door_mask, kernel_door, iterations=1)

            depth_all_cv = depth_cv.copy()
            if pred_door_mask is not None:
                depth_all_cv[pred_door_mask > 0] = 0
                
            depth_nav_cv = depth_cv.copy()
            
            is_floor = (gt_floor_mask > 0)
            is_door = (pred_door_mask > 0) if pred_door_mask is not None else np.zeros_like(is_floor)
            
            valid_nav_mask = np.logical_or(is_floor, is_door)

            depth_nav_cv[~valid_nav_mask] = 0


            color_o3d = o3d.geometry.Image(color_cv)
            
            depth_all_o3d = o3d.geometry.Image(depth_all_cv)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_all_o3d, depth_scale=1000.0, depth_trunc=1000.0, convert_rgb_to_intensity=False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(1280,720,intrinsic[0][0],intrinsic[1][1],intrinsic[0][2],intrinsic[1][2])
            )
            
            points = np.asarray(pcd.points)
            points = (R @ points.T + T).T
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd_all += pcd

            depth_nav_o3d = o3d.geometry.Image(depth_nav_cv)
            rgbd_nav = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_nav_o3d, depth_scale=1000.0, depth_trunc=10.0, convert_rgb_to_intensity=False)
            pcd_nav = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_nav,
                o3d.camera.PinholeCameraIntrinsic(1280,720,intrinsic[0][0],intrinsic[1][1],intrinsic[0][2],intrinsic[1][2])
            )
            
            points = np.asarray(pcd_nav.points)
            points = (R @ points.T + T).T
            pcd_nav.points = o3d.utility.Vector3dVector(points)
            pcd_navigable += pcd_nav

            depth_door_cv = depth_nav_cv.copy()
            depth_door_cv[~is_door] = 0
            if depth_door_cv.sum().item() > 0:
                depth_door_o3d = o3d.geometry.Image(depth_door_cv)
                rgbd_door = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_door_o3d, depth_scale=1000.0, depth_trunc=10.0, convert_rgb_to_intensity=False)
                pcd_door = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_door,
                    o3d.camera.PinholeCameraIntrinsic(1280,720,intrinsic[0][0],intrinsic[1][1],intrinsic[0][2],intrinsic[1][2])
                )
                points = np.asarray(pcd_door.points)
                points = (R @ points.T + T).T
                points[:,2] = points[:,2].min()
                pcd_door.points = o3d.utility.Vector3dVector(points)
                pcd_navigable += pcd_door


            if pano_id is not None:
                pano_color_path = (pano_id + '/rgb_rawlight.png').replace("perspective","panorama")
                pano_depth_path = (pano_id + '/depth.png').replace("perspective","panorama")
                pano_semantic_path = (pano_id + '/semantic.png').replace("perspective","panorama")

                pano_color_cv = cv2.imread(pano_color_path)
                pano_color_cv = cv2.cvtColor(pano_color_cv, cv2.COLOR_BGR2RGB)
                pano_depth_cv = cv2.imread(pano_depth_path, cv2.IMREAD_UNCHANGED)
                pano_semantic_cv = cv2.imread(pano_semantic_path)
                pano_semantic_cv = cv2.cvtColor(pano_semantic_cv, cv2.COLOR_BGR2RGB)

                pano_diff = pano_semantic_cv.astype(np.float32) - np.array([152, 223, 138]).astype(np.float32)
                pano_gt_floor_mask = (np.sum(np.abs(pano_diff), axis=-1) < 5).astype(np.uint8) * 255

                pano_depth_meters = pano_depth_cv.astype(np.float32) / 1000.0
                _, pano_pred_door_mask = segmenter.predict(pano_color_cv, pano_depth_meters)

                cam_path = pano_id.replace("perspective/full","panorama/camera_xyz.txt")
                camera_center = np.loadtxt(cam_path)
                

                if pano_pred_door_mask is not None:
                    kernel_door = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                    pano_pred_door_mask = cv2.dilate(pano_pred_door_mask, kernel_door, iterations=1)

                pano_depth_all = pano_depth_cv.copy()
                if pano_pred_door_mask is not None:
                    pano_depth_all[pano_pred_door_mask > 0] = 0
                
                pano_pcd = create_pcd_from_panorama(pano_color_cv, pano_depth_all, camera_center)
                pcd_all += pano_pcd

                pano_depth_nav = pano_depth_cv.copy()
                pano_is_floor = (pano_gt_floor_mask > 0)
                pano_is_door = (pano_pred_door_mask > 0) if pano_pred_door_mask is not None else np.zeros_like(pano_is_floor)
                
                pano_valid_mask = np.logical_or(pano_is_floor, pano_is_door)
                pano_depth_nav[~pano_valid_mask] = 0
                
                pano_pcd_nav = create_pcd_from_panorama(pano_color_cv, pano_depth_nav, camera_center)
                pcd_navigable += pano_pcd_nav

                pano_depth_door = pano_depth_nav.copy()
                pano_depth_door[~pano_is_door] = 0
                if pano_depth_door.sum().item() > 0:
                    pano_pcd_door = create_pcd_from_panorama(pano_color_cv, pano_depth_door, camera_center)
                    points = np.asarray(pano_pcd_door.points)
                    points[:,2] = points[:,2].min()
                    pano_pcd_door.points = o3d.utility.Vector3dVector(points)
                    pcd_navigable += pano_pcd_door
        except:
            print(image_id,"not found.")


    pcd_all = pcd_all.voxel_down_sample(voxel_size=0.005)
    pcd_navigable_down = pcd_navigable.voxel_down_sample(voxel_size=0.05)
    pcd_all = remove_floating_chunks(pcd_all)
    #pcd_all, pcd_navigable_down = calibrate_scene_orientation(pcd_all,pcd_navigable_down)
    pcd_navigable_down = clean_navigable_cloud_advanced(pcd_navigable_down, obstacle_buffer=0.3)
    pcd_navigable_down = fill_navigable_area(pcd_navigable_down, pcd_all)

    if not validate_scene_quality(pcd_navigable_down):
        continue

    #o3d.visualization.draw_geometries([pcd_all], window_name="Full Scene RGB PointCloud")
    #o3d.visualization.draw_geometries([pcd_navigable_down], window_name="Navigable Area PointCloud (Voxel Filtered)")

    o3d.io.write_point_cloud("data/nav_map/Structured3D/"+scene_id.split("/")[-2]+".pcd", pcd_all, write_ascii=False, compressed=True)
    o3d.io.write_point_cloud("data/nav_map/Structured3D/"+scene_id.split("/")[-2]+"_navigable.pcd", pcd_navigable_down, write_ascii=False, compressed=True)


scene_list = []
for i in range(800):
    path = 'data/scene_datasets/ScanNet/scannet_train_images/frames_square/'
    scene = 'scene'+str(i).rjust(4, "0")+'_00/'    
    scene_list.append(path+scene)

for scene_id in tqdm(scene_list):
    image_list = []
    for image_id in range(1000):
        image_id = image_id * 20
        image_path = scene_id + 'color/' + str(image_id) + ".jpg"
        if not os.path.exists(image_path):
            break
        image_list.append(str(image_id))

    tsdf_full = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=0.05,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    tsdf_nav = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=0.05,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for image_id in tqdm(image_list, desc=f"Processing Scene {scene_id}"):
        intrinsic_matrix = np.eye(3)
        with open(scene_id + 'intrinsic_depth.txt', 'r') as file:
            lines = [line.strip().split() for line in file]
            intrinsic_matrix[0, 0] = float(lines[0][0])
            intrinsic_matrix[1, 1] = float(lines[1][1])
            intrinsic_matrix[0, 2] = float(lines[0][2])
            intrinsic_matrix[1, 2] = float(lines[1][2])

        extrinsic = np.eye(4)
        with open(scene_id + 'pose/' + image_id + '.txt', 'r') as file:
            extrinsic_raw = [line.strip() for line in file]
        for i in range(4):
            for j in range(4):
                extrinsic[i][j] = float(extrinsic_raw[i].split()[j])

        color_path = scene_id + 'color/' + image_id + ".jpg"
        depth_path = scene_id + 'depth/' + image_id + ".png"
        
        color_cv = cv2.imread(color_path)
        color_cv = cv2.cvtColor(color_cv, cv2.COLOR_BGR2RGB)
        depth_cv = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) # uint16

        gray = cv2.cvtColor(color_cv, cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur_score < 120.0: 
            continue

        target_w, target_h = 320, 240
        color_cv = cv2.resize(color_cv, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        depth_cv = cv2.resize(depth_cv, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        fx, fy = intrinsic_matrix[0, 0] / 2, intrinsic_matrix[1, 1] / 2
        cx, cy = intrinsic_matrix[0, 2] / 2, intrinsic_matrix[1, 2] / 2
        
        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(target_w, target_h, fx, fy, cx, cy)

        depth_meters = depth_cv.astype(np.float32) / 1000.0
        nav_mask, _ = segmenter.predict(color_cv, depth_meters)

        color_o3d = o3d.geometry.Image(color_cv)
        depth_o3d = o3d.geometry.Image(depth_cv)

        rgbd_full = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, 
            depth_scale=1000.0, 
            depth_trunc=4.0,
            convert_rgb_to_intensity=False
        )

        if nav_mask is not None:
            depth_nav_cv = depth_cv.copy()
            depth_nav_cv[nav_mask == 0] = 0
            depth_nav_o3d = o3d.geometry.Image(depth_nav_cv)
            
            rgbd_nav = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_nav_o3d, 
                depth_scale=1000.0, 
                depth_trunc=4.0, 
                convert_rgb_to_intensity=False
            )
        else:
            rgbd_nav = None

        if np.isfinite(extrinsic).all():
            try:
                world_to_cam = np.linalg.inv(extrinsic)
                
                tsdf_full.integrate(rgbd_full, o3d_intrinsic, world_to_cam)
                
                if rgbd_nav is not None:
                    tsdf_nav.integrate(rgbd_nav, o3d_intrinsic, world_to_cam)
            except np.linalg.LinAlgError:
                print(f"Pose inversion failed for {image_id}")
        
    pcd_all = tsdf_full.extract_point_cloud()
    
    pcd_navigable = tsdf_nav.extract_point_cloud()

    pcd_all = pcd_all.voxel_down_sample(voxel_size=0.01)
    pcd_navigable_down = pcd_navigable.voxel_down_sample(voxel_size=0.05)

    pcd_all = remove_floating_chunks(pcd_all)
    #pcd_all, pcd_navigable_down = calibrate_scene_orientation(pcd_all,pcd_navigable_down)
    pcd_navigable_down = fill_navigable_area(pcd_navigable_down, pcd_all)
    pcd_navigable_down = clean_navigable_cloud_advanced(pcd_navigable_down)

    if not validate_scene_quality(pcd_navigable_down):
        continue

    #o3d.visualization.draw_geometries([pcd_all], window_name="Full Scene RGB PointCloud")
    #o3d.visualization.draw_geometries([pcd_navigable_down], window_name="Navigable Area PointCloud (Voxel Filtered)")

    o3d.io.write_point_cloud("data/nav_map/ScanNet/"+scene_id.split("/")[-2]+".pcd", pcd_all, write_ascii=False, compressed=True)
    o3d.io.write_point_cloud("data/nav_map/ScanNet/"+scene_id.split("/")[-2]+"_navigable.pcd", pcd_navigable_down, write_ascii=False, compressed=True)

 

# ==============================================================================
# 2. 3RScan (Modified with TSDF and Segmentation), 3RScan is not used in Image2Sim, can ignore it
# ==============================================================================
print("\n========== Processing 3RScan ==========")
scene_list = os.listdir('data/scene_datasets/3RScan/scenes')

for scene_id in tqdm(scene_list, desc="3RScan Scenes"):
    image_list = []
    base_path = 'data/scene_datasets/3RScan/scenes/'+scene_id+'/sequence/'
    
    # Check if sequence exists
    if not os.path.exists(base_path):
        continue

    for image_id in range(1000):  
        image_path_chk = base_path + 'frame-'+str(image_id).zfill(6)+'.color.jpg'
        if not os.path.exists(image_path_chk):
            break
        image_list.append(base_path + 'frame-'+str(image_id).zfill(6))

    if not image_list:
        continue

    # Initialize TSDF Volumes
    tsdf_full = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01, sdf_trunc=0.05, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    tsdf_nav = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01, sdf_trunc=0.05, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    # Pre-load Intrinsic (assuming constant for sequence)
    # The original code parses intrinsics per image, but usually it's constant per sequence.
    # We will stick to the original logic of reading it inside, but optimization suggests reading once.
    # However, to be safe and match original logic structure:
    intrinsic_matrix_data = None
    try:
        with open(base_path + '_info.txt', 'r') as file:  
            intrinsic_raw_lines = [line.strip() for line in file]
        # Line 9 contains intrinsic info in 3RScan _info.txt usually
        intrinsic_vals = intrinsic_raw_lines[9].split(" ")[2:]
        intrinsic_matrix_data = np.eye(4)
        for i in range(4):  
            for j in range(4): 
                intrinsic_matrix_data[i][j] = float(intrinsic_vals[i*4+j])
    except Exception as e:
        print(f"Error reading intrinsics for {scene_id}: {e}")
        continue

    # 3RScan Target Resolution (from original code)
    TARGET_W, TARGET_H = 224, 172

    # Create Open3D Intrinsic Object
    # Note: If the intrinsic in _info.txt is for full res, and we resize, we need to scale fx, fy, cx, cy.
    # The original code used these intrinsics directly with (224, 172). 
    # We assume the intrinsic values in _info.txt MATCH the resized resolution or the user logic was implicit.
    # To correspond exactly to previous logic:
    fx, fy = intrinsic_matrix_data[0][0], intrinsic_matrix_data[1][1]
    cx, cy = intrinsic_matrix_data[0][2], intrinsic_matrix_data[1][2]
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(TARGET_W, TARGET_H, fx, fy, cx, cy)

    for image_path_base in tqdm(image_list, leave=False, desc="Integrating"):
        # 1. Load Extrinsics
        extrinsic = np.eye(4)
        try:
            with open(image_path_base + '.pose.txt', 'r') as file:  
                extrinsic_raw = [line.strip() for line in file]
            for i in range(4):  
                for j in range(4): 
                    extrinsic[i][j] = float(extrinsic_raw[i].split()[j])
        except:
            continue
            
        # Check Pose validity
        if not np.isfinite(extrinsic).all():
            continue

        # 3RScan .pose.txt usually stores Camera-to-World (T_wc)
        # TSDF integration requires World-to-Camera (T_cw)
        try:
            world_to_cam = np.linalg.inv(extrinsic)
        except np.linalg.LinAlgError:
            continue

        # 2. Load Images (with strict resolution control)
        try:
            # Color: Resize to 224x172
            color_pil = Image.open(image_path_base + ".color.jpg").resize((TARGET_W, TARGET_H))
            color_np = np.asarray(color_pil)
            
            # Depth: Resize to 224x172 (Nearest Neighbor to preserve values)
            depth_pil = Image.open(image_path_base + ".depth.pgm").resize((TARGET_W, TARGET_H), Image.NEAREST)
            depth_np = np.asarray(depth_pil).astype(np.uint16)
        except Exception as e:
            continue

        # Blur Check (Optional but recommended)
        gray = cv2.cvtColor(color_np, cv2.COLOR_RGB2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 100.0:
            continue

        # 3. Predict Navigable Mask
        depth_meters = depth_np.astype(np.float32) / 1000.0
        nav_mask, _ = segmenter.predict(color_np, depth_meters)

        # 4. Integrate Full Scene
        color_o3d = o3d.geometry.Image(color_np)
        depth_o3d = o3d.geometry.Image(depth_np)
        
        rgbd_full = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=4.0, convert_rgb_to_intensity=False)
        tsdf_full.integrate(rgbd_full, o3d_intrinsic, world_to_cam)

        # 5. Integrate Navigable Area
        if nav_mask is not None:
            depth_nav_np = depth_np.copy()
            depth_nav_np[nav_mask == 0] = 0
            depth_nav_o3d = o3d.geometry.Image(depth_nav_np)
            
            rgbd_nav = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_nav_o3d, depth_scale=1000.0, depth_trunc=4.0, convert_rgb_to_intensity=False)
            tsdf_nav.integrate(rgbd_nav, o3d_intrinsic, world_to_cam)

    # Post-Processing
    pcd_all = tsdf_full.extract_point_cloud()
    pcd_navigable = tsdf_nav.extract_point_cloud()

    if len(pcd_all.points) > 0:
        pcd_all = pcd_all.voxel_down_sample(voxel_size=0.01)
        pcd_navigable = pcd_navigable.voxel_down_sample(voxel_size=0.05)
        
        # Calibrate & Clean
        pcd_all = remove_floating_chunks(pcd_all)
        #pcd_all, pcd_navigable = calibrate_scene_orientation(pcd_all, pcd_navigable)
        pcd_navigable = fill_navigable_area(pcd_navigable, pcd_all)
        pcd_navigable = clean_navigable_cloud_advanced(pcd_navigable)
        
        if validate_scene_quality(pcd_navigable):
            save_path = f"data/nav_map/3RScan/{scene_id}.pcd"
            save_path_nav = f"data/nav_map/3RScan/{scene_id}_navigable.pcd"
            
            # Ensure dir exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            #o3d.visualization.draw_geometries([pcd_all], window_name="Full Scene RGB PointCloud")
            #o3d.visualization.draw_geometries([pcd_navigable], window_name="Navigable Area PointCloud (Voxel Filtered)")
            o3d.io.write_point_cloud(save_path, pcd_all, write_ascii=False, compressed=True)
            o3d.io.write_point_cloud(save_path_nav, pcd_navigable, write_ascii=False, compressed=True)
            print(f"Saved {scene_id}")


# ==============================================================================
# 3. ARKitScenes (Modified with TSDF and Segmentation)
# ==============================================================================
print("\n========== Processing ARKitScenes ==========")

# ARKit Helper Functions
def convert_angle_axis_to_matrix3(angle_axis):
    """Return a Matrix3 for the angle axis.
    Arguments:
        angle_axis {Point3} -- a rotation in angle axis form.
    """
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix

# from ARKit Scene, some with modifications
def TrajStringToMatrix(traj_str):
    """ convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)

    Returns:
        Rt: rotation matrix, translation matrix
    """

    tokens = traj_str.split()
    assert len(tokens) == 7
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return Rt


def st2_camera_intrinsics(filename):
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])


scene_list = os.listdir('data/scene_datasets/ARKitScenes/3dod/Training')
image_list = []
for scene_id in tqdm(scene_list):
    
    image_path = 'data/scene_datasets/ARKitScenes/3dod/Training/'+scene_id+'/'+scene_id+'_frames/lowres_wide'
    image_list = os.listdir(image_path)
    image_list.sort()
    extrinsic_file = 'data/scene_datasets/ARKitScenes/3dod/Training/'+scene_id+'/'+scene_id+'_frames/lowres_wide.traj'
    with open(extrinsic_file, 'r') as file:  
        extrinsic_list = [line.strip() for line in file]

    image_ids = [i for i in range(len(image_list))]
    random.shuffle(image_ids)
    image_ids = image_ids[:200]
    image_list = [image_path+'/'+image_list[i] for i in image_ids]
    extrinsic_list = [extrinsic_list[i] for i in image_ids]

    pcd_all = o3d.geometry.PointCloud()
    # TSDF Initialization
    tsdf_full = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01, sdf_trunc=0.04, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    tsdf_nav = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01, sdf_trunc=0.04, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    extrinsic_id = 0
    TARGET_W, TARGET_H = 256, 192
    for image_path in tqdm(image_list):
        intrinsic_file = 'data/scene_datasets/ARKitScenes/3dod/Training/'+scene_id+'/'+scene_id+'_frames'+'/lowres_wide_intrinsics/' + image_path.split('/')[-1][:-4]+'.pincam'
        with open(intrinsic_file, 'r') as file:  
            intrinsic_raw = [line.split() for line in file]
        intrinsic = st2_camera_intrinsics(intrinsic_raw[0])
        
        fx, fy = intrinsic[0][0], intrinsic[1][1]
        cx, cy = intrinsic[0][2], intrinsic[1][2]
        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(TARGET_W, TARGET_H, fx, fy, cx, cy)

        extrinsic = TrajStringToMatrix(extrinsic_list[extrinsic_id])
        extrinsic_id += 1

        try:
            world_to_cam = np.linalg.inv(extrinsic)
        except np.linalg.LinAlgError:
            continue

        color_cv = cv2.imread(image_path)
        color_cv = cv2.cvtColor(color_cv, cv2.COLOR_BGR2RGB)
        depth_cv = cv2.imread(image_path.replace("lowres_wide","lowres_depth"), cv2.IMREAD_UNCHANGED) # uint16 usually

        # Blur Check
        gray = cv2.cvtColor(color_cv, cv2.COLOR_RGB2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 80.0: # ARKit frames can be motion blurry
            continue

        # 4. Predict Navigable Mask
        depth_meters = depth_cv.astype(np.float32) / 1000.0
        nav_mask, _ = segmenter.predict(color_cv, depth_meters)

        # 5. Integrate
        color_o3d = o3d.io.read_image(image_path)
        depth_o3d = o3d.io.read_image(image_path.replace("lowres_wide","lowres_depth"))

        rgbd_full = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
        tsdf_full.integrate(rgbd_full, o3d_intrinsic, world_to_cam)

        if nav_mask is not None:
            depth_nav_cv = depth_cv.copy()
            depth_nav_cv[nav_mask == 0] = 0
            depth_nav_o3d = o3d.geometry.Image(depth_nav_cv)
            
            rgbd_nav = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_nav_o3d, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
            tsdf_nav.integrate(rgbd_nav, o3d_intrinsic, world_to_cam)

    # Post-Processing
    pcd_all = tsdf_full.extract_point_cloud()
    pcd_navigable = tsdf_nav.extract_point_cloud()

    if len(pcd_all.points) > 0:
        pcd_all = pcd_all.voxel_down_sample(voxel_size=0.01)
        pcd_navigable = pcd_navigable.voxel_down_sample(voxel_size=0.05)
        
        # Calibrate & Clean
        pcd_all = remove_floating_chunks(pcd_all)
        #pcd_all, pcd_navigable = calibrate_scene_orientation(pcd_all, pcd_navigable)
        pcd_navigable = fill_navigable_area(pcd_navigable, pcd_all)
        pcd_navigable = clean_navigable_cloud_advanced(pcd_navigable)
        
        if validate_scene_quality(pcd_navigable):
            save_path = f"data/nav_map/ARKitScenes/{scene_id}.pcd"
            save_path_nav = f"data/nav_map/ARKitScenes/{scene_id}_navigable.pcd"
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            #o3d.visualization.draw_geometries([pcd_all], window_name="Full Scene RGB PointCloud")
            #o3d.visualization.draw_geometries([pcd_navigable], window_name="Navigable Area PointCloud (Voxel Filtered)")
            o3d.io.write_point_cloud(save_path, pcd_all, write_ascii=False, compressed=True)
            o3d.io.write_point_cloud(save_path_nav, pcd_navigable, write_ascii=False, compressed=True)
            print(f"Saved {scene_id}")
