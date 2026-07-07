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


def read_dpt(file_path):
    with open(file_path, 'rb') as fid:
        tag = struct.unpack('f', fid.read(4))[0]
        width = struct.unpack('i', fid.read(4))[0]
        height = struct.unpack('i', fid.read(4))[0]
        depth_data = np.fromfile(fid, np.float32)
        depth_data = depth_data.reshape(height, width)
    return depth_data


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


def sphere_to_local_cartesian(depth, width, height):
    x_idx = np.arange(width)
    y_idx = np.arange(height)
    xx, yy = np.meshgrid(x_idx, y_idx)

    theta = (2 * np.pi * xx) / width
    phi = (np.pi * yy) / height

    X = np.sin(phi) * - np.sin(theta)
    Y = np.cos(phi)
    Z = np.sin(phi) * np.cos(theta)


    points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)
    
    points = points * depth.flatten()[:, np.newaxis]

    return points

def reconstruct_scene(folder_path, voxel_size=0.01):
    rgb_files = sorted(glob.glob(os.path.join(folder_path, "*_rgb.png")))
    global_pcd = o3d.geometry.PointCloud()
    
    print(f"{len(rgb_files)} viewpoints...")

    for i, rgb_path in enumerate(rgb_files):
        try:
            filename = os.path.basename(rgb_path)
            uuid = filename.split('_rgb')[0]
            
            depth_path = os.path.join(folder_path, f"{uuid}_depth.dpt")
            pose_path = os.path.join(folder_path, f"{uuid}_pose.txt")
            
            if not os.path.exists(depth_path) or not os.path.exists(pose_path):
                print(f"Skipping {uuid}: missing files.")
                continue
                
            rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
            depth = read_dpt(depth_path)
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
            if (i+1) % 5 == 0: 
                print(f"Processing {i+1}/{len(rgb_files)} | Points: {len(global_pcd.points)}")
            
        except Exception as e:
            print(f"Error processing {uuid}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if voxel_size > 0:
        global_pcd = global_pcd.voxel_down_sample(voxel_size)
        
    return global_pcd



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

    ################# Habitat coord (Y-up) to MP3D coord (Z-up) #################
    valid_nav_points_mp3d = np.zeros_like(valid_nav_points)
    valid_nav_points_mp3d[:, 0] = valid_nav_points[:, 0]       
    valid_nav_points_mp3d[:, 1] = -valid_nav_points[:, 2]      
    valid_nav_points_mp3d[:, 2] = valid_nav_points[:, 1]       
    
    valid_nav_points = valid_nav_points_mp3d


    navigable_pcd = o3d.geometry.PointCloud()
    navigable_pcd.points = o3d.utility.Vector3dVector(valid_nav_points)
    navigable_pcd.paint_uniform_color([0.0, 1.0, 0.0]) # Green
    
    return navigable_pcd


def make_habitat_sim_config(scene_path):
    if not os.path.exists(scene_path):
        print(f"Warning: Scene file not found: {scene_path}")
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = True 
    agent_cfg = habitat_sim.AgentConfiguration()
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def save_pcd(nav_pcd, scene_pcd=None, scene_name="scene", save_path="mp3d", visualization=False):

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


def main():
    
    habitat_scene_path = "data/scene_datasets/mp3d"
    pano_rgbd_path = "data/scene_datasets/Matterport3D_360/data"
    scenes = set(os.listdir(habitat_scene_path))
  

    print(f"Found {len(scenes)} unique scenes to process.")

    for idx, scene_name in enumerate(scenes):
        mp3d_scene_path = habitat_scene_path+"/"+scene_name+"/"+scene_name+".glb"
        rgbd_scene_path = pano_rgbd_path+"/"+scene_name
        
        print(f"\n################################################################")
        print(f"Processing Scene {idx+1}/{len(scenes)}: {scene_name}")
        print(f"################################################################")

        scene_pcd = reconstruct_scene(rgbd_scene_path, 0.02)

        cfg = make_habitat_sim_config(mp3d_scene_path)
        try:
            sim = habitat_sim.Simulator(cfg)
        except Exception as e:
            print(f"Failed to load scene {mp3d_scene_path}: {e}")
            exit()
            
        sim.initialize_agent(0)
        nav_pcd = get_navigable_pcd_from_habitat(sim, 0.05)
        save_pcd(nav_pcd, scene_pcd, scene_name, save_path="mp3d", visualization=False)
        

if __name__ == "__main__":
    main()