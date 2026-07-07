import habitat_sim
import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
import copy
import glob
import struct
from PIL import Image
from scipy.spatial.transform import Rotation as R
import cv2

# ==========================================
# 1. Tools
# ==========================================

def read_dpt(file_path):
    with open(file_path, 'rb') as fid:
        tag = struct.unpack('f', fid.read(4))[0]
        width = struct.unpack('i', fid.read(4))[0]
        height = struct.unpack('i', fid.read(4))[0]
        depth_data = np.fromfile(fid, np.float32)
        depth_data = depth_data.reshape(height, width)
    return depth_data

def write_dpt(depth_data, file_path):
    height, width = depth_data.shape
    with open(file_path, 'wb') as fid:
        fid.write(struct.pack('f', 3136.0)) 
        fid.write(struct.pack('i', width))
        fid.write(struct.pack('i', height))
        fid.write(depth_data.astype(np.float32).tobytes())

def get_pose_matrix(file_path):
    with open(file_path, 'r') as f:
        content = f.read().replace(',', ' ').strip()
        values = [float(x) for x in content.split()]
    
    t_vec = np.array(values[:3])
    quat = values[3:] 
    
    r = R.from_quat(quat)
    R_matrix = r.as_matrix()
    
    T_c2w = np.eye(4)
    T_c2w[:3, :3] = R_matrix
    T_c2w[:3, 3] = t_vec
    
    return T_c2w


def get_euclidean_correction_matrix(width, height):
    x_idx = np.arange(width)
    y_idx = np.arange(height)
    xx, yy = np.meshgrid(x_idx, y_idx)

    theta = (2 * np.pi * xx) / width
    phi = (np.pi * yy) / height

    X = np.sin(phi) * -np.sin(theta)
    Y = np.cos(phi)
    Z = np.sin(phi) * np.cos(theta)
    
    direction_vectors = np.stack((X, Y, Z), axis=-1)

    cos_alpha = np.max(np.abs(direction_vectors), axis=-1)
    
    return cos_alpha


def get_gibson_pose_from_habitat(sensor_state):

    pos_hab = np.array(sensor_state.position)
    rot_hab = sensor_state.rotation
    
    if hasattr(rot_hab, 'vector'):
        qx, qy, qz = rot_hab.vector
        qw = rot_hab.scalar
    else: 
        qw, qx, qy, qz = rot_hab.components
        
    R_hab_w_c = R.from_quat([qx, qy, qz, qw]).as_matrix()
    
    R_gibson_hab = np.array([
        [ 1,  0,  0],
        [ 0,  0, -1],
        [ 0,  1,  0]
    ])
    
    R_final = R_gibson_hab @ R_hab_w_c
    pos_final = R_gibson_hab @ pos_hab
    
    quat_final = R.from_matrix(R_final).as_quat() # [x, y, z, w]
    
    return pos_final, quat_final


def sphere_to_local_cartesian(depth, width, height, heading_offset_rad=0.0):
    x_idx = np.arange(width)
    y_idx = np.arange(height)
    xx, yy = np.meshgrid(x_idx, y_idx)

    theta = (2 * np.pi * xx) / width + heading_offset_rad
    phi = (np.pi * yy) / height

    X = np.sin(phi) * -np.sin(theta)
    Y = np.cos(phi)
    Z = np.sin(phi) * np.cos(theta)
    
    points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)
    points = points * depth.flatten()[:, np.newaxis]

    return points

# ==========================================
# 2. Habitat Simulator
# ==========================================

def make_habitat_sim_config(scene_path, camera_height=1.5):
    if not os.path.exists(scene_path):
        print(f"Warning: Scene file not found: {scene_path}")
        
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = True 
    
    agent_cfg = habitat_sim.AgentConfiguration()
    
    # --- (Equirectangular) ---
    rgb_sensor = habitat_sim.EquirectangularSensorSpec()
    rgb_sensor.uuid = "rgb_pano"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [512, 1024]

    rgb_sensor.position = np.array([0.0, camera_height, 0.0])
    
    depth_sensor = habitat_sim.EquirectangularSensorSpec()
    depth_sensor.uuid = "depth_pano"
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor.resolution = [512, 1024]

    depth_sensor.position = np.array([0.0, camera_height, 0.0])

    agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def get_navigable_pcd_from_habitat(sim: habitat_sim.Simulator, grid_resolution: float = 0.05):

    pathfinder = sim.pathfinder
    lower_bound, upper_bound = pathfinder.get_bounds()
    
    x_coords = np.arange(lower_bound[0], upper_bound[0], grid_resolution)
    y_coords = np.arange(lower_bound[1], upper_bound[1], 0.2) 
    z_coords = np.arange(lower_bound[2], upper_bound[2], grid_resolution)
    
    xv, yv, zv = np.meshgrid(x_coords, y_coords, z_coords)
    xyz_points = np.column_stack((xv.flatten(), yv.flatten(), zv.flatten()))
    
    valid_nav_points = []
    
    for i in tqdm(range(len(xyz_points)), desc="Snapping to NavMesh"):
        query_pt = xyz_points[i].astype(np.float32)
        snapped_pt = pathfinder.snap_point(query_pt)
        if not np.isnan(snapped_pt[0]) and pathfinder.is_navigable(snapped_pt):
            valid_nav_points.append(snapped_pt)

    valid_nav_points = np.unique(np.array(valid_nav_points), axis=0)
    

    habitat_nav_points = copy.deepcopy(valid_nav_points)

    valid_nav_points_gibson = np.zeros_like(valid_nav_points)
    valid_nav_points_gibson[:, 0] = valid_nav_points[:, 0]       
    valid_nav_points_gibson[:, 1] = -valid_nav_points[:, 2]      
    valid_nav_points_gibson[:, 2] = valid_nav_points[:, 1]       

    navigable_pcd = o3d.geometry.PointCloud()
    navigable_pcd.points = o3d.utility.Vector3dVector(valid_nav_points_gibson)
    navigable_pcd.paint_uniform_color([0.0, 1.0, 0.0]) 
    
    return navigable_pcd, habitat_nav_points


def sample_and_render_panos(sim, habitat_nav_points, output_dir, grid_resolution=0.05, sqm_per_view=2.0, min_views=10, max_views=500):

    os.makedirs(output_dir, exist_ok=True)

    point_area = grid_resolution * grid_resolution
    total_navigable_area = len(habitat_nav_points) * point_area
    
    calculated_views = int(total_navigable_area / sqm_per_view)
    
    num_views = max(min_views, min(calculated_views, max_views))
    
    print(f"[Area Auto-Scaling] navigable area: {total_navigable_area:.2f} m²")
    print(f"[Area Auto-Scaling] viewpoints: {calculated_views} -> {num_views}")

    if len(habitat_nav_points) > num_views:
        indices = np.random.choice(len(habitat_nav_points), num_views, replace=False)
        sampled_pts = habitat_nav_points[indices]
    else:
        sampled_pts = habitat_nav_points

    agent = sim.get_agent(0)
    print(f"Start rendering {len(sampled_pts)} Viewpoints...")

    height = 512
    width = 1024
    correction_matrix = get_euclidean_correction_matrix(width, height)

    for i, pt in enumerate(tqdm(sampled_pts, desc="Rendering Panos")):
        random_yaw = np.random.uniform(0, 2 * np.pi)
        rot = habitat_sim.utils.common.quat_from_angle_axis(random_yaw, np.array([0.0, 1.0, 0.0]))
        
        agent_state = habitat_sim.AgentState()
        agent_state.position = pt
        agent_state.rotation = rot
        agent.set_state(agent_state)
        
        obs = sim.get_sensor_observations()
        rgb = obs.get("rgb_pano")
        depth = obs.get("depth_pano")
        
        if rgb is None or depth is None:
            continue
            
        if rgb.shape[2] == 4:
            rgb = rgb[..., :3]
            
        euclidean_depth = depth / correction_matrix
        # ===================================================
        
        uuid = f"{i:05d}"
        
        Image.fromarray(rgb).save(os.path.join(output_dir, f"{uuid}_rgb.png"))
        
        depth_mm = (euclidean_depth * 1000.0).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(output_dir, f"{uuid}_depth.png"), depth_mm)
        
        sensor_state = agent.get_state().sensor_states['rgb_pano']
        t_gibson, q_gibson = get_gibson_pose_from_habitat(sensor_state)
        pose_str = f"{t_gibson[0]:.6f} {t_gibson[1]:.6f} {t_gibson[2]:.6f} " \
                   f"{q_gibson[0]:.6f} {q_gibson[1]:.6f} {q_gibson[2]:.6f} {q_gibson[3]:.6f}"
        
        with open(os.path.join(output_dir, f"{uuid}_pose.txt"), 'w') as f:
            f.write(pose_str)


def reconstruct_scene(folder_path, voxel_size=0.01):
    rgb_files = sorted(glob.glob(os.path.join(folder_path, "*_rgb.png")))
    global_pcd = o3d.geometry.PointCloud()
    
    print(f"\n {len(rgb_files)} viewpoints...")

    for i, rgb_path in enumerate(rgb_files):
        try:
            filename = os.path.basename(rgb_path)
            uuid = filename.split('_rgb')[0]
            
            depth_path = os.path.join(folder_path, f"{uuid}_depth.dpt")
            pose_path = os.path.join(folder_path, f"{uuid}_pose.txt")
                
            rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
            if os.path.exists(depth_path):
                depth = read_dpt(depth_path)
            else:
                depth_path = os.path.join(folder_path, f"{uuid}_depth.png")
                depth_mm = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                depth = depth_mm.astype(np.float32) / 1000.0

            pose_matrix = get_pose_matrix(pose_path)
            
            mask = (depth > 0.1) & (depth < 10.0) 
            
            h, w = depth.shape
            local_pts = sphere_to_local_cartesian(depth, w, h)
            colors = rgb.reshape(-1, 3) / 255.0
            
            valid_indices = mask.reshape(-1)
            local_pts = local_pts[valid_indices]
            colors = colors[valid_indices]
            
            if len(local_pts) == 0: continue
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(local_pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            pcd.transform(pose_matrix)
            
            if voxel_size > 0:
                pcd = pcd.voxel_down_sample(voxel_size)
            
            global_pcd += pcd
            
        except Exception as e:
            print(f"Error processing {uuid}: {e}")
            continue

    if voxel_size > 0:
        global_pcd = global_pcd.voxel_down_sample(voxel_size)
        
    return global_pcd


def save_pcd(nav_pcd, scene_pcd=None, scene_name="scene", save_path="gibson", visualization=False):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    
    geometries = [nav_pcd, coordinate_frame]

    if scene_pcd is not None:
        scene_vis = copy.deepcopy(scene_pcd)
        geometries.append(scene_vis)

    if save_path:
        o3d.io.write_point_cloud(f"{save_path}/{scene_name}_navigable.pcd", nav_pcd, write_ascii=False, compressed=True)
        if scene_pcd is not None:
            o3d.io.write_point_cloud(f"{save_path}/{scene_name}.pcd", scene_pcd, write_ascii=False, compressed=True)

    if visualization:
        o3d.visualization.draw_geometries(geometries, window_name="Navigable Area Debug", width=1024, height=768)


# ==========================================
# 3. Main
# ==========================================



def main():
    
    habitat_scene_path = "data/scene_datasets/mp3d" # data/scene_datasets/gibson, data/scene_datasets/hm3d
    output_path = "data/nav_map/mp3d" # data/nav_map/gibson, data/nav_map/hm3d
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    scenes = set(os.listdir(habitat_scene_path))
  
    print(f"Found {len(scenes)} unique scenes to process.")

    for idx, scene_name in enumerate(scenes):
        if 'glb' not in scene_name:
            continue
        scene_name = scene_name.split('.')[0]
        gibson_scene_path = habitat_scene_path+"/"+scene_name+".glb"
        
        print(f"\n################################################################")
        print(f"Processing Scene {idx+1}/{len(scenes)}: {scene_name}")
        print(f"################################################################")

        cfg = make_habitat_sim_config(gibson_scene_path)
        try:
            sim = habitat_sim.Simulator(cfg)
        except Exception as e:
            print(f"Failed to load scene {gibson_scene_path}: {e}")
            exit()
            
        sim.initialize_agent(0)
        
        grid_res = 0.05
        navigable_pcd, habitat_nav_points = get_navigable_pcd_from_habitat(sim, grid_res)
        
        sample_and_render_panos(
            sim, 
            habitat_nav_points, 
            output_path+"/"+scene_name, 
            grid_resolution=grid_res, 
            sqm_per_view=1.0,   
            min_views=10,       
            max_views=500       
        )
        sim.close()
        del sim

        save_pcd(navigable_pcd, None, scene_name, save_path=output_path+"/"+scene_name, visualization=False)

        #reconstructed_pcd = reconstruct_scene(output_path+"/"+scene_name, voxel_size=0.02)
        #save_pcd(navigable_pcd, reconstructed_pcd, scene_name, save_path=output_path+"/"+scene_name, visualization=True)

if __name__ == "__main__":
    main()