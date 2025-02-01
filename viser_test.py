import random
import time
from PIL import Image
from pathlib import Path
from typing import List
import open3d as o3d
import glob
from helpers import *
import re, os
import trimesh
import imageio.v3 as iio
import numpy as np
import tyro
from tqdm.auto import tqdm
from scipy.linalg import inv

import viser
import viser.transforms as tf
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)

def main(
    colmap_path: Path = Path(__file__).parent / "assets/colmap_garden/sparse/0",
    images_path: Path = Path(__file__).parent / "rgb",
    depth_path: Path = Path(__file__).parent / "depth",
    downsample_factor: int = 2,
) -> None:
    """Visualize COLMAP sparse reconstruction outputs.

    Args:
        colmap_path: Path to the COLMAP reconstruction directory.
        images_path: Path to the COLMAP images directory.
        downsample_factor: Downsample factor for the images.
    """
    server = viser.ViserServer()
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Load the colmap info.
    cameras = read_cameras_binary("cameras.bin")
    images = read_images_binary("images.bin")
    points3d = read_points3d_binary("points3D.bin")
    foundation_path = "newresults"
    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )
    # create dictionaries
    object_pose_camera = parse_pose_data("newresults")
    camera_pose_world = parse_poses('images.txt')
    world_pose_data = compute_world_poses(object_pose_camera, camera_pose_world)
    merged_world_poses = merge_object_variants(world_pose_data)
    final_obj_poses_world = fill_missing_poses(merged_world_poses)
    with open("outputy.txt", "w") as f:
        f.write(str(final_obj_poses_world))  # Convert dict to string and write
    # initialize some things
    handles = {}
    rgb_paths = sorted(glob.glob(os.path.join(images_path, "frame_*.png")))
    K = np.array([607.7611083984375, 0.0, 432.25628662109375,
        0.0, 606.5545043945312, 237.27389526367188,
        0.0, 0.0, 1.0]).reshape(3,3)
    P = np.eye(4) # C2W
    min_img = 1 # don't set to 0 or it will break - rip 1h :(
    max_img = 25
    visibility = False

    for rgb_img_path in rgb_paths:
        # get filename to get corresponding depth image and frame number
        filename = os.path.basename(rgb_img_path)
        depth_file = os.path.join(depth_path, filename)
        image_idx = int(re.search(r'\d+', filename).group()) 
        # only load a short sequence due to memory issues
        if min_img <= image_idx <= max_img:
            # get P from dict
            # Extract quaternion and translation for frame 
            quaternion = np.array(camera_pose_world[image_idx][0])  # [w, x, y, z]
            translation = np.array(camera_pose_world[image_idx][1])  # (t_x, t_y, t_z)
            # Convert quaternion to a 3x3 rotation matrix and calculate the transform
            rotation_matrix = Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
            P = np.eye(4) # C2W
            P[:3, :3] = rotation_matrix # Transpose to get C2W
            P[:3, 3] = translation  # Set translation
            T_world_camera = tf.SE3.from_matrix(P).inverse()
            # load the rgb image
            rgb_pil = Image.open(rgb_img_path)
            rgb_np = np.array(rgb_pil)
            rgb_image = o3d.geometry.Image(rgb_np)
            # load the depth image
            depth_pil = Image.open(depth_file)
            depth_np = np.array(depth_pil, dtype=np.uint16)  # Ensure it's 16-bit depth
            depth_image = o3d.geometry.Image(depth_np)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image, convert_rgb_to_intensity=False)
            # define K #TODO dimension 848x480?
            width, height = rgb_pil.size
            intrinsics = o3d.camera.PinholeCameraIntrinsic(
                        width,
                        height,
                        K[0,0], 
                        K[1,1],
                        K[0,2],
                        K[1,2])
            pcd_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
            pcd_points = np.array(pcd_o3d.points)  # @ c2w[:3,:3].T + c2w[:3,3]
            pcd_colors = (np.array(pcd_o3d.colors)*255.0).astype(np.uint8)
            image = np.array(rgb_image, dtype=np.uint8)  # not sure if necessary
            frame_handle = server.scene.add_frame(
                            f"/frame_{image_idx}",
                            wxyz=T_world_camera.rotation().wxyz,
                            position=T_world_camera.translation(),
                            axes_length=0.005,
                            axes_radius=0.001,
                            visible=visibility
                        )
            frustum = server.scene.add_camera_frustum(
                            f"/frame_{image_idx}/frustum",
                            fov=60,
                            aspect=width / height,
                            scale=0.1,
                            color=(255,0,0),
                            # wxyz=tf.SO3.from_z_radians(-np.pi/2).wxyz,
                            image=image,
                            visible=visibility
                        )
            pcd_handle = server.scene.add_point_cloud(f"/frame_{image_idx}/pcd",
                                    points=pcd_points,
                                    colors=pcd_colors,
                                    point_size=0.001,
                                    visible=visibility
                                    )
            handles[image_idx] = {
                'pcd': pcd_handle,
                'frustum': frustum,
                'frame': frame_handle
            } 
            print(f"added frame {image_idx}")

    server.scene.add_frame(
    name = "world_coordinate_system",
    wxyz=np.array([1,0,0,0]),
    position=np.array([0,0,0]),
    axes_length=0.1,
    axes_radius=0.005,
    )

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
            [0.0, -1.0, 0.0]
        )

    gui_points = server.gui.add_slider(
        "Max points",
        min=1,
        max=len(points3d),
        step=1,
        initial_value=min(len(points3d), 50_000),
    )

    gui_frames = server.gui.add_slider(
        "Max frames",
        min=1,
        max=len(images),
        step=1,
        initial_value=min(len(images), 1),
    )

    gui_point_size = server.gui.add_slider(
        "Point size", min=0.01, max=0.1, step=0.001, initial_value=0.05
    )

    points = np.array([points3d[p_id].xyz for p_id in points3d])
    colors = np.array([points3d[p_id].rgb for p_id in points3d])

    point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
    point_cloud = server.scene.add_point_cloud(
        name="/colmap/pcd",
        points=points[point_mask],
        colors=colors[point_mask],
        point_size=gui_point_size.value,
    )
    frames: List[viser.FrameHandle] = []

    def visualize_frames() -> None:
        """Send all COLMAP elements to viser for visualization. This could be optimized
        a ton!"""

        # Remove existing image frames.

        for frame in frames:
            frame.remove()
        frames.clear()

        # Interpret the images and cameras.
        img_ids = [im.id for im in images.values()]
        #random.shuffle(img_ids)
        img_ids = sorted(img_ids[: gui_frames.value])
        print(img_ids)
    
        def attach_callback(
            frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle
        ) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in tqdm(img_ids):
            img = images[img_id]
            cam = cameras[img.camera_id]

            # Skip images that don't exist.
            image_filename = images_path / img.name
            if not image_filename.exists():
                continue
            print(img.tvec)
            print(tf.SO3(img.qvec))
            T_world_camera = tf.SE3.from_rotation_and_translation(
                tf.SO3(img.qvec), img.tvec
            ).inverse()
            print(T_world_camera)
            frame = server.scene.add_frame(
                f"/colmap/frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.1,
                axes_radius=0.005,
            )
            frames.append(frame)

            # For pinhole cameras, cam.params will be (fx, fy, cx, cy).
            if cam.model != "PINHOLE":
                print(f"Expected pinhole camera, but got {cam.model}")

            H, W = cam.height, cam.width
            fy = cam.params[1]
            image = iio.imread(image_filename)
            image = image[::downsample_factor, ::downsample_factor]
            frustum = server.scene.add_camera_frustum(
                f"/colmap/frame_{img_id}/frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.15,
                image=image,
            )
            attach_callback(frustum, frame)  

    def transform_to_frame(final_obj_poses_world, camera_in_world, img_id):
        obj_at_frame = {}
        for obj, poses in final_obj_poses_world.items():
            print(obj)
            # Pose amtrix of obj in world
            quat =  poses[img_id][0]
            trans = poses[img_id][1]
            rot = Rotation.from_quat(quat, scalar_first=True).as_matrix()
            P = np.eye(4) # C2W
            P[:3, :3] = rot # Transpose to get C2W
            P[:3, 3] = trans  # Set translation
            P = inv(P)
            # Pose matrix of camera in world
            cam_quat_wxyz, cam_trans = camera_in_world[img_id]
            rot_cam = Rotation.from_quat(cam_quat_wxyz, scalar_first=True).as_matrix()
            T_cam_world = np.eye(4)
            T_cam_world[:3, :3] = rot_cam
            T_cam_world[:3, 3] = cam_trans
            # Perform Transformation
            T_cam_obj = P @ T_cam_world
            T_obj_cam = inv(T_cam_obj)
            obj_at_frame[obj] = T_obj_cam
        return obj_at_frame     
    need_update = True

    @gui_points.on_update
    def _(_) -> None:
        point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
        point_cloud.points = points[point_mask]
        point_cloud.colors = colors[point_mask]

    @gui_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud.point_size = gui_point_size.value

    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
                "Timestep",
                min=min_img,
                max= max_img, 
                step=1,
                initial_value=1,
                disabled=False,
            )            
        prev_timestep = gui_timestep.value
        @gui_timestep.on_update
        # TODO some frames are skipped when using the slider and i dont know why
        def _(_) -> None:
            nonlocal prev_timestep
            current_timestep = gui_timestep.value
            obj_at_frame = transform_to_frame(final_obj_poses_world, camera_pose_world, current_timestep)
            for obj, pose in obj_at_frame.items():
                print(pose)
                mesh = trimesh.load_mesh(os.path.join(f"resources/{obj}/mesh", f"{obj}.obj"), process=False)
                vertices = mesh.vertices
                faces = mesh.faces
                rot_mat = pose[:3,:3]
                quarternion = Rotation.from_matrix(rot_mat).as_quat(scalar_first=True)
                translation = pose[:3,3]
                server.scene.add_mesh_simple(
                            name=f"/world_coordinate_system/{obj}_{current_timestep}",
                            vertices=vertices,
                            faces=faces,
                            wxyz=quarternion,
                            position=translation,
                        )
                print("added")

            with server.atomic():
                for v in handles[prev_timestep].values():
                    v.visible = False
                for v in handles[current_timestep].values():
                    v.visible = True 
            prev_timestep = current_timestep
            server.flush()  # Optional! This will force the GUI to update immediately.

    while True:
        if need_update:
            need_update = False
            visualize_frames()


            # for obj, poses in final_obj_poses_world.items():
            #     mesh = trimesh.load_mesh(os.path.join(f"resources/{obj}/mesh", f"{obj}.obj"), process=False)
            #     assert isinstance(mesh, trimesh.Trimesh)
            #     vertices = mesh.vertices
            #     faces = mesh.faces
            #     # if obj == "milk":
            #     #     break
            #     for idx, pose in enumerate(poses):
            #         if idx == 1:
            #             print(f"added {obj} to frame {idx}")
            #             translation = pose[1]
            #             translation = np.array(translation)
            #             quarternion = pose[0]
            #             quarternion = np.array(quarternion)
            #             server.scene.add_mesh_simple(
            #                 name=f"/name/{obj}_{idx}",
            #                 vertices=vertices,
            #                 faces=faces,
            #                 wxyz=quarternion,
            #                 position=translation,
            #             )


        time.sleep(1e-3)



if __name__ == "__main__":
    tyro.cli(main)