import os
import glob
import struct
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy.spatial.transform import Rotation as R
import random
from scipy.spatial import cKDTree


# ==============================================================================
# 1. Geometry and camera tools
# ==============================================================================
def quaternion_multiply(q1, q2):
    """
    Hamilton Product: q_out = q1 * q2
    output [w, x, y, z]
    q1: (..., 4) Rotation of Pose
    q2: (..., 4) Rotation of Local Gaussian
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)

def apply_rotation_to_gs(gs_features, pose_matrix, device):
    """
    From Local Gaussian Rotation to World Space
    gs_features: (N, 8) -> [scale(3), rot(4), opacity(1)]
    pose_matrix: (4, 4) numpy array, camera-to-world
    """
    # 1. (Camera -> World)
    R_cam = pose_matrix[:3, :3]
    
    q_cam_xyzw = R.from_matrix(R_cam).as_quat()
    q_cam_wxyz = np.array([q_cam_xyzw[3], q_cam_xyzw[0], q_cam_xyzw[1], q_cam_xyzw[2]])
    
    q_cam_tensor = torch.tensor(q_cam_wxyz, device=device, dtype=torch.float32)
    
    # 2. Local Gaussian Rotation [w, x, y, z]
    q_local = gs_features[:, 3:7] 
    
    # 3. q_world = q_cam * q_local
    q_world = quaternion_multiply(q_cam_tensor.unsqueeze(0), q_local)
    
    # 4. Normalize
    q_world = F.normalize(q_world, p=2, dim=-1, eps=1e-4)
    
    # 5. Write
    gs_features_world = gs_features.clone()
    gs_features_world[:, 3:7] = q_world
    
    return gs_features_world

def get_pose_matrix_from_quat(t_vec, quat):
    r = R.from_quat(quat)
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = t_vec
    return T

def normalize(vector):
    return vector / np.linalg.norm(vector)

def parse_structured3d_camera_info(camera_info, height, width):
    lookat = normalize(camera_info[3:6])
    up = normalize(camera_info[6:9])

    W = lookat
    U = np.cross(W, up)
    V = np.cross(W, U)

    rot = np.vstack((U, V, W))
    trans = camera_info[:3] / 1000.0 # mm to meters

    xfov = camera_info[9]
    yfov = camera_info[10]

    K = np.eye(3)
    K[0, 2] = width / 2
    K[1, 2] = height / 2
    K[0, 0] = K[0, 2] / np.tan(max(float(xfov), 1e-5))
    K[1, 1] = K[1, 2] / np.tan(max(float(yfov), 1e-5))

    # Extrinsics (World to Cam) -> Inverse to get Cam to World
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rot
    extrinsic_inv = np.linalg.inv(extrinsic) # T_c2w (rotation part)
    
    # Translation
    T_c2w = np.eye(4)
    T_c2w[:3, :3] = extrinsic_inv[:3, :3]
    T_c2w[:3, 3] = trans
    
    return T_c2w, K


def get_position_heading_from_pose(frame, device="cuda"):
    dataset_type = frame['type']
    pose_matrix = frame['pose']
    if frame['type'] == 'realsee':
        pos_np = pose_matrix[:3, 3] 
        pos_np_swapped = np.array([pos_np[0], pos_np[2], -pos_np[1]])
        pos_tensor = torch.tensor(pos_np_swapped, device=device, dtype=torch.float32).unsqueeze(0)
    elif "structured3d" in frame['type']:
        if pose_matrix.shape == (4, 4):
            pos_np = pose_matrix[:3, 3]
        else:
            pos_np = pose_matrix / 1000.0
            pose_matrix = np.eye(4)
            pose_matrix[:3, 3] = pos_np
        pos_tensor = torch.tensor(pos_np, device=device, dtype=torch.float32).unsqueeze(0)
    else:
        pos_np = pose_matrix[:3, 3]
        pos_tensor = torch.tensor(pos_np, device=device, dtype=torch.float32).unsqueeze(0)

    if dataset_type in ["matterport", "hm3d", "gibson"]:
        R = pose_matrix[:3, :3]
        local_forward = np.array([0, 0, -1])
        world_forward = R @ local_forward
        heading = (np.pi / 2) - np.arctan2(world_forward[1], world_forward[0])
    elif dataset_type == "realsee":
        raw_R = pose_matrix[:3, :3]
        local_fwd = np.array([0, 0, 1]) 
        raw_fwd = raw_R @ local_fwd
        forward_world = np.array([raw_fwd[0], raw_fwd[2], -raw_fwd[1]])
        heading = np.pi / 2 - np.arctan2(forward_world[1], forward_world[0])
    elif "structured3d" in dataset_type:
        R = pose_matrix[:3, :3]
        local_forward = np.array([0, 0, -1])
        world_forward = R @ local_forward
        heading = np.arctan2(world_forward[1], world_forward[0])
    else:
        print("Don't support such dataset:", dataset_type, ", the supported dataset: matterport, realsee, structured3d")
        heading = 0.0
    
    heading_tensor = torch.tensor([heading], device=device, dtype=torch.float32)
    return pos_tensor, heading_tensor

def read_dpt(path):
    """read Matterport .dpt depth file"""
    if "png" in path or "PNG" in path:
        depth_img_cv = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        data = depth_img_cv.astype(np.float32) / 1000.0
    else:
        with open(path, 'rb') as fid:
            tag = struct.unpack('f', fid.read(4))[0]
            width = struct.unpack('i', fid.read(4))[0]
            height = struct.unpack('i', fid.read(4))[0]
            data = np.fromfile(fid, np.float32).reshape(height, width)
    return data


def remove_flying_points_kdtree(xyz, rgb, feats, gs, radius=0.1, min_neighbors=4):
    if xyz is None or xyz.shape[0] < min_neighbors + 1:
        return xyz, rgb, feats, gs
    
    xyz_np = xyz.detach().cpu().numpy()
    
    tree = cKDTree(xyz_np)
    
    neighbors_counts = tree.query_ball_point(xyz_np, r=radius, return_length=True, workers=-1)
    
    valid_mask = neighbors_counts >= (min_neighbors + 1)
    valid_mask_tensor = torch.tensor(valid_mask, device=xyz.device, dtype=torch.bool)
    
    clean_xyz = xyz[valid_mask_tensor]
    clean_rgb = rgb[valid_mask_tensor]
    clean_feats = feats[valid_mask_tensor] if feats is not None else None
    clean_gs = gs[valid_mask_tensor] if gs is not None else None
    
    return clean_xyz, clean_rgb, clean_feats, clean_gs


def convert_angle_axis_to_matrix3(angle_axis):
    matrix, _ = cv2.Rodrigues(angle_axis)
    return matrix

def TrajStringToMatrix(traj_str):
    tokens = traj_str.split()
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return Rt

def st2_camera_intrinsics(filename):
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])


# ==============================================================================
# 2. Data loader for scenes
# ==============================================================================

def load_matterport_data(scene_path, max_batch_size=1):
    rgb_files = sorted(glob.glob(os.path.join(scene_path, "*_rgb.png")))
    frames_data = []
    
    sample_k = min(max_batch_size, len(rgb_files))
    rgb_files = random.sample(rgb_files, k=sample_k) if sample_k > 0 else []
    
    for rgb_path in rgb_files:
        uuid = os.path.basename(rgb_path).split('_rgb')[0]
        depth_path = os.path.join(scene_path, f"{uuid}_depth.dpt")
        pose_path = os.path.join(scene_path, f"{uuid}_pose.txt")
        
        if not os.path.exists(depth_path) or not os.path.exists(pose_path):
            continue
            
        with open(pose_path, 'r') as f:
            content = f.read().replace(',', ' ').strip()
            values = [float(x) for x in content.split()]
        t_vec = np.array(values[:3])
        quat = values[3:]
        pose_matrix = get_pose_matrix_from_quat(t_vec, quat)
        
        frames_data.append({
            'type': 'matterport',
            'uuid': uuid,
            'rgb_path': rgb_path,
            'depth_path': depth_path,
            'pose': pose_matrix,
            'intrinsics': None,
            'trainable': True
        })
    return frames_data


def load_hm3d_data(scene_path, max_batch_size=1):
    rgb_files = sorted(glob.glob(os.path.join(scene_path, "*_rgb.png")))
    frames_data = []
    
    sample_k = min(max_batch_size, len(rgb_files))
    rgb_files = random.sample(rgb_files, k=sample_k) if sample_k > 0 else []
    
    for rgb_path in rgb_files:
        uuid = os.path.basename(rgb_path).split('_rgb')[0]
        depth_path = os.path.join(scene_path, f"{uuid}_depth.png")
        pose_path = os.path.join(scene_path, f"{uuid}_pose.txt")
        
        if not os.path.exists(depth_path) or not os.path.exists(pose_path):
            continue
            
        with open(pose_path, 'r') as f:
            content = f.read().replace(',', ' ').strip()
            values = [float(x) for x in content.split()]
        t_vec = np.array(values[:3])
        quat = values[3:]
        pose_matrix = get_pose_matrix_from_quat(t_vec, quat)
        
        frames_data.append({
            'type': 'hm3d',
            'uuid': uuid,
            'rgb_path': rgb_path,
            'depth_path': depth_path,
            'pose': pose_matrix,
            'intrinsics': None,
            'trainable': True
        })
    return frames_data


def load_gibson_data(scene_path, max_batch_size=1):
    rgb_files = sorted(glob.glob(os.path.join(scene_path, "*_rgb.png")))
    frames_data = []
    
    sample_k = min(max_batch_size, len(rgb_files))
    rgb_files = random.sample(rgb_files, k=sample_k) if sample_k > 0 else []
    
    for rgb_path in rgb_files:
        uuid = os.path.basename(rgb_path).split('_rgb')[0]
        depth_path = os.path.join(scene_path, f"{uuid}_depth.png")
        pose_path = os.path.join(scene_path, f"{uuid}_pose.txt")
        
        if not os.path.exists(depth_path) or not os.path.exists(pose_path):
            continue
            
        with open(pose_path, 'r') as f:
            content = f.read().replace(',', ' ').strip()
            values = [float(x) for x in content.split()]
        t_vec = np.array(values[:3])
        quat = values[3:]
        pose_matrix = get_pose_matrix_from_quat(t_vec, quat)
        
        frames_data.append({
            'type': 'gibson',
            'uuid': uuid,
            'rgb_path': rgb_path,
            'depth_path': depth_path,
            'pose': pose_matrix,
            'intrinsics': None,
            'trainable': True
        })
    return frames_data


def load_realsee_data(scene_path, max_batch_size=1):
    viewpoints_dir = os.path.join(scene_path, "viewpoints")
    if not os.path.exists(viewpoints_dir):
        viewpoints_dir = scene_path
        
    if os.path.exists(os.path.join(viewpoints_dir, "viewpoints")): 
        viewpoints_dir = os.path.join(viewpoints_dir, "viewpoints")

    if not os.path.exists(viewpoints_dir):
        return []

    viewpoints = sorted([d for d in os.listdir(viewpoints_dir) if os.path.isdir(os.path.join(viewpoints_dir, d))])
    frames_data = []
    
    sample_k = min(max_batch_size, len(viewpoints))
    viewpoints = random.sample(viewpoints, k=sample_k) if sample_k > 0 else []
    
    for vp_id in viewpoints:
        vp_dir = os.path.join(viewpoints_dir, vp_id)
        depth_path = os.path.join(vp_dir, "depth_image.png")
        scale_path = os.path.join(vp_dir, "depth_scale.txt")
        color_path = os.path.join(vp_dir, "panoImage_1600.jpg")
        ext_path = os.path.join(vp_dir, "extrinsics.txt")
        
        if not (os.path.exists(depth_path) and os.path.exists(scale_path) and 
                os.path.exists(color_path) and os.path.exists(ext_path)):
            continue

        with open(ext_path, 'r') as f:
            ext_vals = [float(x) for x in f.read().strip().split()]
            pose_matrix = np.array(ext_vals).reshape(4, 4)
        with open(scale_path, 'r') as f:
            depth_scale = float(f.read().strip())

        frames_data.append({
            'type': 'realsee',
            'uuid': vp_id,
            'rgb_path': color_path,
            'depth_path': depth_path,
            'depth_scale': depth_scale,
            'pose': pose_matrix, 
            'intrinsics': None,
            'trainable': True
        })
        
    return frames_data


def load_structured3d_data(scene_path, max_batch_size=1):
    base_render_path = os.path.join(scene_path, '2D_rendering')
    if not os.path.exists(base_render_path):
        return []
        
    frames_data = []
    room_list = sorted([os.path.join(base_render_path, item) for item in os.listdir(base_render_path)])
    
    sample_k = min(max_batch_size, len(room_list))
    room_list = random.sample(room_list, k=sample_k) if sample_k > 0 else []
    
    for room_dir in room_list:
        if not os.path.isdir(room_dir): continue
        room_id = os.path.basename(room_dir)

        # 1. Panorama (Trainable)
        pano_dir = os.path.join(room_dir, 'panorama')
        if os.path.exists(pano_dir):
            rgb_path = os.path.join(pano_dir, 'full/rgb_rawlight.png')
            depth_path = os.path.join(pano_dir, 'full/depth.png')
            cam_xyz_path = os.path.join(pano_dir, 'camera_xyz.txt')
            
            if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(cam_xyz_path):
                camera_center = np.loadtxt(cam_xyz_path)
                frames_data.append({
                    'type': 'structured3d_panorama',
                    'uuid': f"{room_id}_panorama",
                    'rgb_path': rgb_path,
                    'depth_path': depth_path,
                    'pose': camera_center,
                    'intrinsics': None,
                    'trainable': True 
                })

        # 2. Perspective (Memory Only)
        persp_full_dir = os.path.join(room_dir, 'perspective', 'full')
        if os.path.exists(persp_full_dir):
            view_ids = sorted(os.listdir(persp_full_dir))
            for vid in view_ids:
                view_dir = os.path.join(persp_full_dir, vid)
                if not os.path.isdir(view_dir): continue
                
                rgb_path = os.path.join(view_dir, 'rgb_rawlight.png')
                depth_path = os.path.join(view_dir, 'depth.png')
                pose_file = os.path.join(view_dir, 'camera_pose.txt')
                
                if not (os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(pose_file)):
                    continue
                    
                h, w = 720, 1280 
                camera_info = np.loadtxt(pose_file)
                pose_matrix, K = parse_structured3d_camera_info(camera_info, h, w) 
                
                frames_data.append({
                    'type': 'structured3d_perspective',
                    'uuid': f"{room_id}_view_{vid}",
                    'rgb_path': rgb_path,
                    'depth_path': depth_path,
                    'pose': pose_matrix,
                    'intrinsics': K,
                    'trainable': False
                })

    return frames_data


def load_scannet_data(scene_path, max_batch_size=1):
    frames_data = []
    color_dir = os.path.join(scene_path, 'color')
    depth_dir = os.path.join(scene_path, 'depth')
    pose_dir = os.path.join(scene_path, 'pose')
    intrinsic_file = os.path.join(scene_path, 'intrinsic_depth.txt')

    if not os.path.exists(intrinsic_file) or not os.path.exists(color_dir):
        return []

    intrinsic_matrix = np.eye(3)
    with open(intrinsic_file, 'r') as file:
        lines = [line.strip().split() for line in file]
        intrinsic_matrix[0, 0] = float(lines[0][0])
        intrinsic_matrix[1, 1] = float(lines[1][1])
        intrinsic_matrix[0, 2] = float(lines[0][2])
        intrinsic_matrix[1, 2] = float(lines[1][2])

    image_files = sorted(glob.glob(os.path.join(color_dir, "*.jpg")))
    sample_k = min(max_batch_size, len(image_files))
    image_files = random.sample(image_files, k=sample_k) if sample_k > 0 else []

    for img_path in image_files:
        img_id = os.path.basename(img_path).split('.')[0]
        depth_path = os.path.join(depth_dir, f"{img_id}.png")
        pose_path = os.path.join(pose_dir, f"{img_id}.txt")

        if not os.path.exists(depth_path) or not os.path.exists(pose_path): continue

        extrinsic = np.eye(4)
        try:
            with open(pose_path, 'r') as file:
                extrinsic_raw = [line.strip() for line in file]
            for i in range(4):
                for j in range(4):
                    extrinsic[i][j] = float(extrinsic_raw[i].split()[j])
        except: continue
        if not np.isfinite(extrinsic).all(): continue

        frames_data.append({
            'type': 'scannet_perspective',
            'uuid': f"scannet_{img_id}",
            'rgb_path': img_path,
            'depth_path': depth_path,
            'pose': extrinsic,
            'intrinsics': intrinsic_matrix,
            'trainable': True
        })
    return frames_data


def load_3rscan_data(scene_path, max_batch_size=1):
    frames_data = []
    seq_dir = os.path.join(scene_path, 'sequence')
    info_file = os.path.join(seq_dir, '_info.txt')

    if not os.path.exists(info_file):
        return []

    try:
        with open(info_file, 'r') as file:
            intrinsic_raw_lines = [line.strip() for line in file]
        intrinsic_vals = intrinsic_raw_lines[9].split(" ")[2:]
        intrinsic = np.eye(3)
        intrinsic[0, 0] = float(intrinsic_vals[0])
        intrinsic[1, 1] = float(intrinsic_vals[5])
        intrinsic[0, 2] = float(intrinsic_vals[2])
        intrinsic[1, 2] = float(intrinsic_vals[6])
    except:
        return []

    image_files = sorted(glob.glob(os.path.join(seq_dir, "*.color.jpg")))
    sample_k = min(max_batch_size, len(image_files))
    image_files = random.sample(image_files, k=sample_k) if sample_k > 0 else []

    for img_path in image_files:
        base_name = img_path.replace(".color.jpg", "")
        depth_path = base_name + ".depth.pgm"
        pose_path = base_name + ".pose.txt"

        if not os.path.exists(depth_path) or not os.path.exists(pose_path): continue

        extrinsic = np.eye(4)
        try:
            with open(pose_path, 'r') as file:
                extrinsic_raw = [line.strip() for line in file]
            for i in range(4):
                for j in range(4):
                    extrinsic[i][j] = float(extrinsic_raw[i].split()[j])
        except: continue
        if not np.isfinite(extrinsic).all(): continue

        frames_data.append({
            'type': '3rscan_perspective',
            'uuid': os.path.basename(base_name),
            'rgb_path': img_path,
            'depth_path': depth_path,
            'pose': extrinsic,
            'intrinsics': intrinsic,
            'trainable': True
        })
    return frames_data


def load_arkitscenes_data(scene_path, max_batch_size=1):
    frames_data = []
    scene_id = os.path.basename(os.path.normpath(scene_path))

    frames_dir = os.path.join(scene_path, f"{scene_id}_frames")
    color_dir = os.path.join(frames_dir, 'lowres_wide')
    depth_dir = os.path.join(frames_dir, 'lowres_depth')
    intrinsics_dir = os.path.join(frames_dir, 'lowres_wide_intrinsics') 
    traj_file = os.path.join(frames_dir, 'lowres_wide.traj')

    if not os.path.exists(frames_dir):
        color_dir = os.path.join(scene_path, 'lowres_wide')
        depth_dir = os.path.join(scene_path, 'lowres_depth')
        if not os.path.exists(color_dir):
            color_dir = os.path.join(scene_path, 'vga_wide')

    if not os.path.exists(intrinsics_dir):
        intrinsics_dir = os.path.join(scene_path, 'lowres_wide_intrinsics') 
        traj_file = os.path.join(scene_path, 'lowres_wide.traj')

        if not os.path.exists(intrinsics_dir):
            intrinsics_dir = os.path.join("/dev/shm/Training", scene_id, f"{scene_id}_frames", 'lowres_wide_intrinsics') 
            traj_file = os.path.join("/dev/shm/Training", scene_id, f"{scene_id}_frames", 'lowres_wide.traj')

            if not os.path.exists(intrinsics_dir):
                intrinsics_dir = os.path.join("/dev/shm/Training", scene_id, 'lowres_wide_intrinsics') 
                traj_file = os.path.join("/dev/shm/Training", scene_id, 'lowres_wide.traj')

    if not os.path.exists(color_dir) or not os.path.exists(traj_file):
        return []

    with open(traj_file, 'r') as file:
        extrinsic_list = [line.strip() for line in file]

    image_files = sorted(os.listdir(color_dir))
    
    total_frames = min(len(image_files), len(extrinsic_list))
    indices = list(range(total_frames))
    sample_k = min(max_batch_size, total_frames)
    sampled_indices = random.sample(indices, k=sample_k) if sample_k > 0 else []

    for idx in sampled_indices:
        img_name = image_files[idx]
        img_path = os.path.join(color_dir, img_name)
        depth_name = img_name.replace("lowres_wide", "lowres_depth").replace("vga_wide", "lowres_depth").replace("jpg","png")
        depth_path = os.path.join(depth_dir, depth_name)
        intrinsic_file = os.path.join(intrinsics_dir, img_name[:-4] + '.pincam')
        if not os.path.exists(depth_path) or not os.path.exists(intrinsic_file): continue

        intrinsic = st2_camera_intrinsics(intrinsic_file)
        pose_matrix = TrajStringToMatrix(extrinsic_list[idx])

        frames_data.append({
            'type': 'arkitscenes_perspective',
            'uuid': img_name[:-4],
            'rgb_path': img_path,
            'depth_path': depth_path,
            'pose': pose_matrix,
            'intrinsics': intrinsic,
            'trainable': True
        })
    return frames_data
    

def voxel_down_sample_torch_cuda(xyz, rgb, feats, gs, voxel_size=0.01):
    """
    PyTorch CUDA Voxel Downsample
    """
    if xyz.shape[0] == 0:
        return xyz, rgb, feats, gs

    min_bound = xyz.min(dim=0)[0]
    grid_indices = torch.div(xyz - min_bound, voxel_size, rounding_mode='floor').long()

    grid_max = grid_indices.max(dim=0)[0] + 1
    keys = grid_indices[:, 0] + \
           grid_indices[:, 1] * grid_max[0] + \
           grid_indices[:, 2] * grid_max[0] * grid_max[1]

    unique_keys, inverse_indices = torch.unique(keys, return_inverse=True, sorted=False)
    n_voxels = unique_keys.shape[0]

    new_xyz = torch.zeros((n_voxels, 3), device=xyz.device, dtype=xyz.dtype)
    new_rgb = torch.zeros((n_voxels, 3), device=rgb.device, dtype=rgb.dtype)
    new_feats = torch.zeros((n_voxels, feats.shape[1]), device=feats.device, dtype=feats.dtype)
    new_gs = torch.zeros((n_voxels, 8), device=gs.device, dtype=gs.dtype)
    counts = torch.zeros((n_voxels, 1), device=xyz.device, dtype=xyz.dtype)

    new_xyz.index_add_(0, inverse_indices, xyz)
    new_rgb.index_add_(0, inverse_indices, rgb)
    new_feats.index_add_(0, inverse_indices, feats)
    new_gs.index_add_(0, inverse_indices, gs)
    
    ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
    counts.index_add_(0, inverse_indices, ones)
    counts = torch.clamp(counts, min=1)

    naive_mean_gs = new_gs / counts
    
    ref_quats = F.normalize(naive_mean_gs[:, 3:7], p=2, dim=1) 
    gathered_refs = ref_quats[inverse_indices] 
    orig_quats = gs[:, 3:7] 
    dots = (orig_quats * gathered_refs).sum(dim=1, keepdim=True) 
    flip_sign = torch.sign(dots) 
    flip_sign[flip_sign == 0] = 1 
    aligned_quats = orig_quats * flip_sign
    
    new_rot_acc = torch.zeros((n_voxels, 4), device=gs.device, dtype=gs.dtype)
    new_rot_acc.index_add_(0, inverse_indices, aligned_quats)
    
    final_xyz = new_xyz / counts
    final_rgb = new_rgb / counts
    final_feats = new_feats / counts
    final_gs = naive_mean_gs.clone()
    final_rot_mean = new_rot_acc / counts
    final_rot_norm = F.normalize(final_rot_mean, p=2, dim=1, eps=1e-12)
    final_gs[:, 3:7] = final_rot_norm

    return final_xyz, final_rgb, final_feats, final_gs


def fill_sparse_lidar_depth(depth_map, kernel_size=32, iterations=16):
    """
    An ultra-fast depth diffusion algorithm designed specifically for extremely sparse LiDAR scanlines (e.g., RealSee).
    Implements spatial smooth interpolation based on Normalized Convolution.
    
    Parameters:
        depth_map: (H, W) float depth map.
        kernel_size: Convolution kernel size, determining the search radius for valid surrounding pixels in a single pass.
        iterations: Number of diffusion iterations. More iterations can fill wider gaps.
    """
    # 1. Extract valid depths and mask, handling NaN and zero values
    valid_mask = ((depth_map > 0.01) & ~np.isnan(depth_map)).astype(np.float32)
    
    curr_depth = np.nan_to_num(depth_map.copy(), nan=0.0)
    curr_mask = valid_mask.copy()
    
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    
    # 2. Iterative outward smooth diffusion
    for _ in range(iterations):
        # Convolve depth values and valid weights (Mask) respectively
        sum_depth = cv2.filter2D(curr_depth, -1, kernel)
        sum_mask = cv2.filter2D(curr_mask, -1, kernel)
        
        # Calculate the weighted mean of valid pixels within the neighborhood
        avg_depth = np.zeros_like(curr_depth)
        valid_area = sum_mask > 0
        avg_depth[valid_area] = sum_depth[valid_area] / sum_mask[valid_area]
        
        # Fill the estimated mean only into regions that are currently still missing
        missing = curr_mask == 0
        curr_depth[missing] = avg_depth[missing]
        
        # Update the mask (filled areas can serve as valid data sources to diffuse further outward in the next iteration)
        curr_mask[missing] = (sum_mask[missing] > 0).astype(np.float32)
        
    # 3. Strictly preserve the original valid data
    final_depth = np.where(valid_mask > 0, depth_map, curr_depth)
    
    return final_depth


# ==============================================================================
# 3. Build Scene Memory - CUDA ACCELERATED
# ==============================================================================

def process_frame_to_points_cuda(frame_data, device, model=None, height=512, width=1024, inpaint_depth=False):
    ftype = frame_data['type']
    
    if ftype == 'structured3d_panorama':
        depth_path = frame_data['depth_path']
        rgb_path = frame_data['rgb_path']
        camera_center_np = frame_data['pose'] / 1000. 
        camera_center = torch.tensor(camera_center_np, device=device, dtype=torch.float32)

        depth_img_np = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        color_img_np = cv2.imread(rgb_path)
        if depth_img_np is None or color_img_np is None: return None, None, None, None, None
        
        color_img_np = cv2.cvtColor(color_img_np, cv2.COLOR_BGR2RGB)
        h, w = height, width
        if color_img_np.shape[:2] != (h, w):
            color_img_np = cv2.resize(color_img_np, (w, h), interpolation=cv2.INTER_LINEAR)
        if depth_img_np.shape[:2] != (h, w):
            depth_img_np = cv2.resize(depth_img_np, (w, h), interpolation=cv2.INTER_NEAREST)

        dist = torch.tensor(depth_img_np, device=device, dtype=torch.float32) / 1000.0
        colors = torch.tensor(color_img_np, device=device, dtype=torch.float32) 

        grid_v, grid_u = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        u, v = grid_u.float(), grid_v.float()

        lon = (u / w) * 2 * np.pi - np.pi
        lat = -((v / h) * np.pi - (np.pi / 2)) 

        if inpaint_depth:
            dist = model.inpaint_depth_2d(colors.unsqueeze(0), dist.unsqueeze(0))

        mask = (dist > 0.01) & (dist < 10.0)
        
        cos_lat = torch.cos(lat)
        x = dist * cos_lat * torch.sin(lon)
        y = dist * cos_lat * torch.cos(lon)
        z = dist * torch.sin(lat)
        
        points = torch.stack([x, y, z], dim=-1) 
        points = points + camera_center 
        
        flat_points = points[mask]
        flat_colors = colors[mask] / 255.

        if model is not None:

            feats_map, gs_map, patch_tokens = model.extract_features_from_image(colors.unsqueeze(0), dist.unsqueeze(0)) 
            feats_flat = feats_map.squeeze().permute(1, 2, 0)[mask]
            gs_flat = gs_map.squeeze().permute(1, 2, 0)[mask]
            return flat_points, flat_colors, feats_flat, gs_flat, patch_tokens
        
        return flat_points, flat_colors, None, None, None
        
    elif ftype == 'structured3d_perspective':
        depth_cv = cv2.imread(frame_data['depth_path'], cv2.IMREAD_UNCHANGED)
        rgb_cv = cv2.imread(frame_data['rgb_path'])
        if depth_cv is None or rgb_cv is None: return None, None, None, None, None
        
        # Robustness Guard: Intrinsic matrices are usually aligned with the depth map. 
        # If the RGB resolution does not match, force alignment.
        dh, dw = depth_cv.shape
        rh, rw = rgb_cv.shape[:2]
        if (rh, rw) != (dh, dw):
            rgb_cv = cv2.resize(rgb_cv, (dw, dh), interpolation=cv2.INTER_LINEAR)
            
        rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)
        h, w = dh, dw

        pose = torch.tensor(frame_data['pose'], device=device, dtype=torch.float32) 
        K = frame_data['intrinsics']
        intrinsics = torch.tensor(K, device=device).unsqueeze(0)
        
        depth_tensor = torch.tensor(depth_cv, device=device, dtype=torch.float32) / 1000.0 
        rgb_tensor = torch.tensor(rgb_cv, device=device, dtype=torch.float32)

        i, j = torch.meshgrid(torch.linspace(0, w-1, w, device=device), 
                              torch.linspace(0, h-1, h, device=device), indexing='xy')
        
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        x = (i - cx) * depth_tensor / fx
        y = (j - cy) * depth_tensor / fy
        z = depth_tensor
        
        points_cam = torch.stack([x, y, z], dim=-1).view(-1, 3)
        valid_mask = (depth_tensor > 0) & (depth_tensor < 10.0)
        flat_mask = valid_mask.view(-1)
        
        R = pose[:3, :3]
        t = pose[:3, 3]
        flat_points = points_cam @ R.T + t
        flat_colors = rgb_tensor.view(-1, 3)

        flat_points = flat_points[flat_mask]
        flat_colors = flat_colors[flat_mask] / 255.0 

        if model is not None:
            feats_map, gs_map, patch_tokens = model.extract_features_from_image(
                torch.tensor(rgb_cv, device=device).unsqueeze(0), depth_tensor.unsqueeze(0), intrinsics=intrinsics
            ) 
            feats_flat = feats_map.squeeze().permute(1, 2, 0).view(-1, 16)[flat_mask]
            gs_flat = gs_map.squeeze().permute(1, 2, 0).view(-1, 8)[flat_mask]
            pose_np = frame_data['pose'] 
            gs_flat = apply_rotation_to_gs(gs_flat, pose_np, device)
            return flat_points, flat_colors, feats_flat, gs_flat, None
        
        return flat_points, flat_colors, None, None, None

    # -------------------------------------------------------------------------
    # ScanNet Perspective/Pinhole Image
    # -------------------------------------------------------------------------
    elif ftype == 'scannet_perspective':
        depth_cv = cv2.imread(frame_data['depth_path'], cv2.IMREAD_UNCHANGED)
        rgb_cv = cv2.imread(frame_data['rgb_path'])
        if depth_cv is None or rgb_cv is None: return None, None, None, None, None

        # Filter blurry frames
        gray = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 120.0:
            return None, None, None, None, None 

        rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)
        
        # Resize to 320x240
        target_w, target_h = 320, 240
        rgb_cv = cv2.resize(rgb_cv, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        depth_cv = cv2.resize(depth_cv, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # 640x480 to 320x240
        K = frame_data['intrinsics'].copy()
        K[0, 0] /= 2.0; K[1, 1] /= 2.0
        K[0, 2] /= 2.0; K[1, 2] /= 2.0

        pose = torch.tensor(frame_data['pose'], device=device, dtype=torch.float32)
        intrinsics = torch.tensor(K, device=device).unsqueeze(0)
        depth_tensor = torch.tensor(depth_cv, device=device, dtype=torch.float32) / 1000.0 
        rgb_tensor = torch.tensor(rgb_cv, device=device, dtype=torch.float32)

        i, j = torch.meshgrid(torch.linspace(0, target_w-1, target_w, device=device), 
                              torch.linspace(0, target_h-1, target_h, device=device), indexing='xy')
        
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        x = (i - cx) * depth_tensor / fx
        y = (j - cy) * depth_tensor / fy
        z = depth_tensor
        
        points_cam = torch.stack([x, y, z], dim=-1).view(-1, 3)
        # max_depth 4.0m
        valid_mask = (depth_tensor > 0.1) & (depth_tensor < 4.0)
        flat_mask = valid_mask.view(-1)
        
        R = pose[:3, :3]
        t = pose[:3, 3]
        flat_points = points_cam @ R.T + t
        flat_colors = rgb_tensor.view(-1, 3)[flat_mask] / 255.0 
        flat_points = flat_points[flat_mask]

        if model is not None:
            feats_map, gs_map, patch_tokens = model.extract_features_from_image(
                torch.tensor(rgb_cv, device=device).unsqueeze(0), depth_tensor.unsqueeze(0), intrinsics=intrinsics) 
            feats_flat = feats_map.squeeze().permute(1, 2, 0).view(-1, 16)[flat_mask]
            gs_flat = gs_map.squeeze().permute(1, 2, 0).view(-1, 8)[flat_mask]
            gs_flat = apply_rotation_to_gs(gs_flat, frame_data['pose'], device)
            return flat_points, flat_colors, feats_flat, gs_flat, patch_tokens
        
        return flat_points, flat_colors, None, None, None

    # -------------------------------------------------------------------------
    # 3RScan Perspective/Pinhole image
    # -------------------------------------------------------------------------
    elif ftype == '3rscan_perspective':
        depth_cv = cv2.imread(frame_data['depth_path'], cv2.IMREAD_UNCHANGED)
        rgb_cv = cv2.imread(frame_data['rgb_path'])
        if depth_cv is None or rgb_cv is None: return None, None, None, None, None

        gray = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 100.0:
            return None, None, None, None, None 

        rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)
        
        target_w, target_h = 224, 160
        rgb_cv = cv2.resize(rgb_cv, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        depth_cv = cv2.resize(depth_cv, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # The native intrinsics of 3RScan are strictly mapped to the 224x172 depth map!
        # NEVER use the high-resolution RGB dimensions to calculate the scale factors!
        K = frame_data['intrinsics'].copy()
        scale_y = 160.0 / 172.0 
        K[1, 1] *= scale_y  # fy
        K[1, 2] *= scale_y  # cy
        # K[0, 0] (fx) 和 K[0, 2] (cx) do not change

        pose = torch.tensor(frame_data['pose'], device=device, dtype=torch.float32)
        intrinsics = torch.tensor(K, device=device).unsqueeze(0)
        depth_tensor = torch.tensor(depth_cv, device=device, dtype=torch.float32) / 1000.0 
        rgb_tensor = torch.tensor(rgb_cv, device=device, dtype=torch.float32)

        i, j = torch.meshgrid(torch.linspace(0, target_w-1, target_w, device=device), 
                              torch.linspace(0, target_h-1, target_h, device=device), indexing='xy')
        
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        x = (i - cx) * depth_tensor / fx
        y = (j - cy) * depth_tensor / fy
        z = depth_tensor
        
        points_cam = torch.stack([x, y, z], dim=-1).view(-1, 3)
        valid_mask = (depth_tensor > 0.1) & (depth_tensor < 4.0)
        flat_mask = valid_mask.view(-1)
        
        R = pose[:3, :3]
        t = pose[:3, 3]
        # 3RScan pose: Cam-to-World
        flat_points = points_cam @ R.T + t
        flat_colors = rgb_tensor.view(-1, 3)[flat_mask] / 255.0 
        flat_points = flat_points[flat_mask]

        if model is not None:
            feats_map, gs_map, patch_tokens = model.extract_features_from_image(
                torch.tensor(rgb_cv, device=device).unsqueeze(0), depth_tensor.unsqueeze(0), intrinsics=intrinsics) 
            feats_flat = feats_map.squeeze().permute(1, 2, 0).view(-1, 16)[flat_mask]
            gs_flat = gs_map.squeeze().permute(1, 2, 0).view(-1, 8)[flat_mask]
            gs_flat = apply_rotation_to_gs(gs_flat, frame_data['pose'], device)
            return flat_points, flat_colors, feats_flat, gs_flat, patch_tokens
        
        return flat_points, flat_colors, None, None, None

    # -------------------------------------------------------------------------
    # ARKitScenes Perspective/Pinhole image
    # -------------------------------------------------------------------------
    elif ftype == 'arkitscenes_perspective':
        depth_cv = cv2.imread(frame_data['depth_path'], cv2.IMREAD_UNCHANGED)
        rgb_cv = cv2.imread(frame_data['rgb_path'])
        if depth_cv is None or rgb_cv is None: return None, None, None, None, None

        gray = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 80.0:
            return None, None, None, None, None 

        rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)
        
        K = frame_data['intrinsics'].copy()
        target_w, target_h = round(K[0,2])*2, round(K[1,2])*2

        if rgb_cv.shape[:2] != (target_h, target_w):
            rgb_cv = cv2.resize(rgb_cv, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            depth_cv = cv2.resize(depth_cv, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        pose = torch.tensor(frame_data['pose'], device=device, dtype=torch.float32)
        intrinsics = torch.tensor(K, device=device).unsqueeze(0)
        depth_tensor = torch.tensor(depth_cv, device=device, dtype=torch.float32) / 1000.0 
        rgb_tensor = torch.tensor(rgb_cv, device=device, dtype=torch.float32)

        i, j = torch.meshgrid(torch.linspace(0, target_w-1, target_w, device=device), 
                              torch.linspace(0, target_h-1, target_h, device=device), indexing='xy')
        
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        x = (i - cx) * depth_tensor / fx
        y = (j - cy) * depth_tensor / fy
        z = depth_tensor
        
        points_cam = torch.stack([x, y, z], dim=-1).view(-1, 3)
        # max_depth 3.0m (depth is high noisey)
        valid_mask = (depth_tensor > 0.1) & (depth_tensor < 3.0)
        flat_mask = valid_mask.view(-1)
        
        R = pose[:3, :3]
        t = pose[:3, 3]
        flat_points = points_cam @ R.T + t
        flat_colors = rgb_tensor.view(-1, 3)[flat_mask] / 255.0 
        flat_points = flat_points[flat_mask]

        if model is not None:
            feats_map, gs_map, patch_tokens = model.extract_features_from_image(
                torch.tensor(rgb_cv, device=device).unsqueeze(0), depth_tensor.unsqueeze(0), intrinsics=intrinsics) 
            feats_flat = feats_map.squeeze().permute(1, 2, 0).view(-1, 16)[flat_mask]
            gs_flat = gs_map.squeeze().permute(1, 2, 0).view(-1, 8)[flat_mask]
            gs_flat = apply_rotation_to_gs(gs_flat, frame_data['pose'], device)
            return flat_points, flat_colors, feats_flat, gs_flat, patch_tokens
        
        return flat_points, flat_colors, None, None, None
    

    elif ftype in ['matterport', 'hm3d', 'gibson']:
        rgb_img = cv2.imread(frame_data['rgb_path'])
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        h, w = height, width
        if rgb_img.shape[:2] != (h, w):
            rgb_img = cv2.resize(rgb_img, (w, h), interpolation=cv2.INTER_LINEAR)

        rgb_np = np.asarray(rgb_img) 
        depth_np = read_dpt(frame_data['depth_path'])
        pose_np = frame_data['pose']
        
        rgb_t = torch.tensor(rgb_np, device=device, dtype=torch.float32).view(-1,3)
        pose_t = torch.tensor(pose_np, device=device, dtype=torch.float32)

        depth_t = torch.tensor(depth_np, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        depth_t = F.interpolate(depth_t, size=(h,w), mode='nearest').view(h,w)

        depth_flat = depth_t.view(-1)
        mask = (depth_flat > 0.1) & (depth_flat < 10.0)

        if inpaint_depth:
            depth_t = model.inpaint_depth_2d(torch.tensor(rgb_np,device=device).unsqueeze(0), depth_t.unsqueeze(0))
            mask = (depth_flat > 0.1) & (depth_flat < 5.0)
        
        grid_v, grid_u = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        theta = (2 * np.pi * grid_u) / w
        phi = (np.pi * grid_v) / h
        sin_phi = torch.sin(phi)
        
        X = sin_phi * -torch.sin(theta)
        Y = torch.cos(phi)
        Z = sin_phi * torch.cos(theta)
        
        local_pts = torch.stack((X, Y, Z), dim=-1).view(-1, 3) 
        local_pts = local_pts * depth_t.view(-1, 1)
        
        local_pts = local_pts[mask]
        colors = rgb_t[mask] / 255.0 
        
        ones = torch.ones((local_pts.shape[0], 1), device=device)
        pts_homo = torch.cat([local_pts, ones], dim=1)
        world_pts = (pose_t @ pts_homo.T).T[:, :3]
        
        if model is not None:

            feats_map, gs_map, patch_tokens = model.extract_features_from_image(torch.tensor(rgb_np,device=device).unsqueeze(0), depth_t.unsqueeze(0)) 
            feats_flat = feats_map.squeeze().permute(1, 2, 0).view(-1,16)[mask]
            gs_flat = gs_map.squeeze().permute(1, 2, 0).view(-1,8)[mask]
            pose_np = frame_data['pose']
            gs_flat = apply_rotation_to_gs(gs_flat, pose_np, device)
            return world_pts, colors, feats_flat, gs_flat, patch_tokens
        
        return world_pts, colors, None, None, None

    elif ftype == 'realsee':
        depth_img = cv2.imread(frame_data['depth_path'], cv2.IMREAD_UNCHANGED)
        rgb_img = cv2.imread(frame_data['rgb_path'])
        if depth_img is None or rgb_img is None: return None, None, None, None, None
        
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        h, w = height, width
        if rgb_img.shape[:2] != (h, w):
            rgb_img = cv2.resize(rgb_img, (w, h))
        if depth_img.shape[:2] != (h, w):
            depth_img = cv2.resize(depth_img, (w, h), interpolation=cv2.INTER_NEAREST)
            
        scale = frame_data['depth_scale']
        ext_np = frame_data['pose'] 
        ext_t = torch.tensor(ext_np, device=device, dtype=torch.float32)

        depth_img = np.array(depth_img).astype(np.float32) / scale
        dist_t = torch.tensor(depth_img, device=device, dtype=torch.float32)
        mask = (dist_t > 0.1) & (dist_t < 10.0) 

        rgb_t = torch.tensor(rgb_img, device=device, dtype=torch.float32)

        if inpaint_depth:
            depth_img_for_mask = fill_sparse_lidar_depth(depth_img, kernel_size=8, iterations=4)
            dist_t_for_mask = torch.tensor(depth_img_for_mask, device=device, dtype=torch.float32)
            mask = (dist_t_for_mask > 0.1) & (dist_t_for_mask < 3.0)

            depth_img = fill_sparse_lidar_depth(depth_img, kernel_size=32, iterations=16)
            dist_t = torch.tensor(depth_img, device=device, dtype=torch.float32)
            dist_t = model.inpaint_depth_2d(rgb_t.unsqueeze(0), dist_t.unsqueeze(0))


        grid_v, grid_u = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        yaw = (grid_v.float() / h - 0.5) * np.pi
        pitch = (grid_u.float() / w - 0.5) * 2.0 * np.pi
        
        if not mask.any(): return None, None, None, None, None
        
        y_s = yaw[mask]
        p_s = pitch[mask]
        d_s = dist_t[mask]
        c_s = rgb_t[mask] / 255.0 
        
        cos_y = torch.cos(y_s)
        sin_y = torch.sin(y_s)
        cos_p = torch.cos(p_s)
        sin_p = torch.sin(p_s)
        
        x_local = d_s * cos_y * sin_p
        y_local = d_s * sin_y
        z_local = d_s * cos_y * cos_p
        
        pts_local = torch.stack([x_local, y_local, z_local, torch.ones_like(x_local)], dim=1) 
        pts_world = pts_local @ ext_t.T
        pts_world = pts_world[:, :3]
        
        pts_world_swapped = torch.stack([
            pts_world[:, 0],
            pts_world[:, 2],
            -pts_world[:, 1]
        ], dim=1)
        
        if model is not None:

            feats_map, gs_map, patch_tokens = model.extract_features_from_image(rgb_t.unsqueeze(0), dist_t.unsqueeze(0)) 
            feats_flat = feats_map.squeeze().permute(1, 2, 0)[mask]
            gs_flat = gs_map.squeeze().permute(1, 2, 0)[mask]

            R_swap = np.array([
                [1,  0,  0],
                [0,  0,  1],
                [0, -1,  0]
            ], dtype=np.float32)
            R_pose = ext_np[:3, :3]
            R_final = R_swap @ R_pose

            gs_flat = apply_rotation_to_gs(gs_flat, R_final, device)
            return pts_world_swapped, c_s, feats_flat, gs_flat, patch_tokens

        return pts_world_swapped, c_s, None, None, None

    return None, None, None, None, None

# ------------------------------------------------------------------------------
# 3.5 NVS (Novel View Synthesis) Dedicated Data Builder - Semi-decoupled Context Injection
# ------------------------------------------------------------------------------
def build_scene_nvs_pointcloud_data(scene_path, dataset_type='matterport', device='cuda', voxel_size=0.01, model=None, num_grad_src=1, num_nograd_src=2, num_targets=2):
    """
    Semi-decoupled 3DGS Data Builder (Priority-Scheduled Revision)
    Forces the Target to be a panorama, while prioritizing pinhole camera images 
    as gradient-enabled Encoder inputs. This compels the model to learn 3D geometric 
    features and multi-view fusion, rather than simply shortcutting into 2D copying 
    of the panorama.
    """
    if isinstance(device, str):
        device = torch.device(device)

    # Load extra frames here to ensure we have a sufficient pool of both pinhole 
    # and panoramic images for classification.
    # Especially for S3D (Structured3D), a single room might contain dozens of 
    # perspective views but only 1-2 panoramas.
    total_frames_needed = num_grad_src + num_nograd_src + num_targets

    if dataset_type == 'matterport':
        frames_data = load_matterport_data(scene_path, max_batch_size=total_frames_needed)
    elif dataset_type == 'realsee':
        frames_data = load_realsee_data(scene_path, max_batch_size=total_frames_needed)
    elif dataset_type == 'structured3d':
        # Structured3D Special Handling: Retrieve more frames (multiplied by 3) to facilitate the subsequent separation of panoramas and pinhole images
        frames_data = load_structured3d_data(scene_path, max_batch_size=total_frames_needed * 3)
    elif dataset_type == 'hm3d':
        frames_data = load_hm3d_data(scene_path, max_batch_size=total_frames_needed)
    elif dataset_type == 'gibson':
        frames_data = load_gibson_data(scene_path, max_batch_size=total_frames_needed)
    elif dataset_type == 'scannet':
        frames_data = load_scannet_data(scene_path, max_batch_size=total_frames_needed)
    elif dataset_type == '3rscan':
        frames_data = load_3rscan_data(scene_path, max_batch_size=total_frames_needed)
    elif dataset_type == 'arkitscenes':
        frames_data = load_arkitscenes_data(scene_path, max_batch_size=total_frames_needed)
    else:
        return None, None, None, None, []

    if not frames_data or len(frames_data) < 2:
        return None, None, None, None, []

    # =================================================================
    # [Classification & Priority Scheduling] Ensure pinhole images are prioritized as Encoder (Source) inputs
    # =================================================================
    pano_frames = [f for f in frames_data if 'perspective' not in f['type']]
    pinhole_frames = [f for f in frames_data if 'perspective' in f['type']]
    
    # CRITICAL: Must guarantee at least 1 panorama is available as the Target for supervised rendering
    if len(pano_frames) < 1:
        return None, None, None, None, []

    random.shuffle(pano_frames)
    random.shuffle(pinhole_frames)

    # 1. Extract Targets (MUST be entirely panoramas!)
    actual_targets_count = min(num_targets, len(pano_frames))
    targets = pano_frames[:actual_targets_count]
    remaining_panos = pano_frames[actual_targets_count:]

    # 2. Build Source Candidate Pool (Pinhole Prioritized)
    src_candidates = pinhole_frames + remaining_panos

    if len(src_candidates) == 0:
        return None, None, None, None, []

    # 3. Strictly slice and allocate Grad and NoGrad Sources based on priority
    actual_grad = min(num_grad_src, len(src_candidates))
    src_grad = src_candidates[:actual_grad]
    
    remaining_src = src_candidates[actual_grad:]
    actual_nograd = min(num_nograd_src, len(remaining_src))
    src_nograd = remaining_src[:actual_nograd]

    all_xyz, all_rgb, all_feats, all_gs = [], [], [], []

    # 4. Process Grad-enabled Sources (Generate computation graph, update DINO/Upsampler)
    for frame in src_grad:
        xyz, rgb, feats, gs, _ = process_frame_to_points_cuda(frame, device, model=model)
        if xyz is not None:
            all_xyz.append(xyz)
            all_rgb.append(rgb)
            all_feats.append(feats)
            all_gs.append(gs)

    # 5. Process NoGrad Context Sources (Purely for stitching 3D memory of vast physical spaces, zero gradient overhead)
    with torch.no_grad():
        for frame in src_nograd:
            xyz, rgb, feats, gs, _ = process_frame_to_points_cuda(frame, device, model=model)
            if xyz is not None:
                # detach() !!!
                all_xyz.append(xyz.detach())
                all_rgb.append(rgb.detach())
                all_feats.append(feats.detach())
                all_gs.append(gs.detach())

    if not all_xyz:
        return None, None, None, None, []

    # Concatenate all Source point clouds (both Grad and NoGrad enter _memory uniformly)
    all_xyz = torch.cat(all_xyz, dim=0)
    all_rgb = torch.cat(all_rgb, dim=0)
    all_feats = torch.cat(all_feats, dim=0)
    all_gs = torch.cat(all_gs, dim=0)

    # 6. Voxel Downsampling (Remains differentiable on gradient-enabled points, successfully propagating Loss back to the network)
    if voxel_size > 0:
        all_xyz, all_rgb, all_feats, all_gs = voxel_down_sample_torch_cuda(
            all_xyz, all_rgb, all_feats, all_gs, voxel_size
        )

    torch.cuda.empty_cache()
    
    # Return the constructed large-scale scene GS, along with the metadata of Target frames to be rendered (guaranteed to be panoramas)
    return all_xyz, all_rgb, all_feats, all_gs, targets

# ------------------------------------------------------------------------------
# Scene Data Preprocessing for Inference
# ------------------------------------------------------------------------------
def build_scene_pointcloud_data(scene_path, dataset_type='matterport', device='cuda', voxel_size=0.01, model=None, max_batch_size=1, inpaint_depth=False):
    if isinstance(device, str):
        device = torch.device(device)

    if dataset_type == 'matterport':
        frames_data = load_matterport_data(scene_path, max_batch_size)
    elif dataset_type == 'realsee':
        frames_data = load_realsee_data(scene_path, max_batch_size)
    elif dataset_type == 'structured3d':
        frames_data = load_structured3d_data(scene_path, max_batch_size)
    elif dataset_type == 'hm3d':
        frames_data = load_hm3d_data(scene_path, max_batch_size)
    elif dataset_type == 'gibson':
        frames_data = load_gibson_data(scene_path, max_batch_size)
    elif dataset_type == 'scannet':
        frames_data = load_scannet_data(scene_path, max_batch_size=max_batch_size)
    elif dataset_type == '3rscan':
        frames_data = load_3rscan_data(scene_path, max_batch_size=max_batch_size)
    elif dataset_type == 'arkitscenes':
        frames_data = load_arkitscenes_data(scene_path, max_batch_size=max_batch_size)
    else:
        return None, None, None, None, []

    if not frames_data:
        return None, None, None, None, []

    all_xyz, all_rgb, all_feats, all_gs = [], [], [], []

    for i, frame in enumerate(frames_data):
        xyz, rgb, feats, gs, _ = process_frame_to_points_cuda(frame, device, model=model, inpaint_depth=inpaint_depth)
        if xyz is not None:

            all_xyz.append(xyz)
            all_rgb.append(rgb)
            all_feats.append(feats)
            all_gs.append(gs)

            all_xyz = torch.cat(all_xyz, dim=0)
            all_rgb = torch.cat(all_rgb, dim=0)
            all_feats = torch.cat(all_feats, dim=0)
            all_gs = torch.cat(all_gs, dim=0)

            if voxel_size > 0:
                all_xyz, all_rgb, all_feats, all_gs = voxel_down_sample_torch_cuda(all_xyz, all_rgb, all_feats, all_gs, voxel_size)

            all_xyz = [all_xyz]
            all_rgb = [all_rgb]
            all_feats = [all_feats]
            all_gs = [all_gs]
    
    all_xyz = torch.cat(all_xyz, dim=0)
    all_rgb = torch.cat(all_rgb, dim=0)
    all_feats = torch.cat(all_feats, dim=0)
    all_gs = torch.cat(all_gs, dim=0)
    

    torch.cuda.empty_cache()
    
    return all_xyz, all_rgb, all_feats, all_gs, frames_data

# ==============================================================================
# 4. Tools
# ==============================================================================
def load_gt_depth_normalized(frame_data, target_h, target_w):
    """Read and normalize GT depth into meters"""
    depth_path = frame_data['depth_path']
    dtype = frame_data['type']
    
    depth_map = None
    
    if dtype in ['matterport', 'hm3d', 'gibson']:
        depth_map = read_dpt(depth_path) # already float meters
        
    elif 'structured3d' in dtype:
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is not None:
            depth_map = depth_img.astype(np.float32) / 1000.0
            
    elif dtype == 'realsee':
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is not None:
            scale = frame_data['depth_scale']
            depth_map = depth_img.astype(np.float32) / scale

    if depth_map is None:
        return np.zeros((target_h, target_w), dtype=np.float32)

    if depth_map.shape[:2] != (target_h, target_w):
        depth_map = cv2.resize(depth_map, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
    return depth_map




def load_checkpoint(model, load_path_or_dir, device="cuda"):

    load_path = load_path_or_dir
    
    # 1. If a directory is passed, automatically look for the file with the maximum iteration
    if os.path.isdir(load_path_or_dir):
        # Assume the filename format is image2sim_iter_{iteration}.pth
        pattern = os.path.join(load_path_or_dir, "image2sim_iter_*.pth")
        files = glob.glob(pattern)
        if not files:
            print(f"[Warning] No checkpoints found in {load_path_or_dir}. Starting from scratch.")
            return 0

        # Sort by iteration number and select the maximum
        load_path = max(files, key=lambda x: int(x.split("_iter_")[-1].split(".pth")[0]))
    
    if not os.path.isfile(load_path):
        print(f"[Warning] Checkpoint file {load_path} not found. Starting from scratch.")
        return 0
    
    # 2. Load data to the corresponding device
    checkpoint = torch.load(load_path, map_location=device)
    
    # 3. Load Model state_dict
    # Handle the 'module.' prefix issue caused by DDP
    # If the current model is a DDP instance (determined by hasattr(model, 'module')), 
    # but the checkpoint keys missing 'module.' (or vice versa), fix the mismatch

    ckpt_dict = checkpoint['model']
    
    is_model_ddp = hasattr(model, 'module')
    is_ckpt_ddp = list(ckpt_dict.keys())[0].startswith('module.')
    
    if is_model_ddp and not is_ckpt_ddp:
        # Model with DDP，but Checkpoint not -> add 'module.'
        ckpt_dict = {f'module.{k}': v for k, v in ckpt_dict.items()}
    elif not is_model_ddp and is_ckpt_ddp:
        # Model without DDP，but Checkpoint is -> remove 'module.'
        ckpt_dict = {k.replace('module.', ''): v for k, v in ckpt_dict.items()}
        
    # strict=True 
    try:
        model.load_state_dict(ckpt_dict, strict=True)
    except RuntimeError as e:
        print(f"[Error] Key mismatch during loading. Trying with strict=False. Details: {e}")
        model.load_state_dict(ckpt_dict, strict=False)

    if 'ema_teacher' in checkpoint:
        ema_ckpt_dict = checkpoint['ema_teacher']
        model.rendering_decoder.load_state_dict(ema_ckpt_dict, strict=True)

    return model