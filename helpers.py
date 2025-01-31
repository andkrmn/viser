import os
import re
import numpy as np
import os
import re
import numpy as np
from scipy.spatial.transform import Rotation
import warnings
from collections import defaultdict

def parse_pose_data(root_dir):
    """
    Input: newresults directory (FoundationPose result directory)
    Returns: nested dictionary {obj_name: {frame_number: pose_matrix}}
    """
    # file pattern to look for when going through the results
    file_pattern = re.compile(r'^frame_(\d+)\.txt$')
    # dict to save result
    data = {}

    # walk through all folders in the rootdirectory and save all the file patterns that were found and the pose matrix
    for dirpath, _, filenames in os.walk(root_dir):
        # Check if we're in an "ob_in_cam" directory
        if os.path.basename(dirpath) == 'ob_in_cam':
            # Get object name from parent directory
            obj_name = os.path.basename(os.path.dirname(dirpath))
            # Add if obj not in data
            if obj_name not in data:
                data[obj_name] = {}

            # Process all frame files in this directory
            for filename in filenames:
                match = file_pattern.match(filename)
                if match:
                    # Extract frame number
                    frame_num = int(match.group(1))
                    
                    # Read and parse the pose matrix
                    file_path = os.path.join(dirpath, filename)
                    try:
                        matrix = np.loadtxt(file_path)
                        if matrix.shape == (4, 4):
                            data[obj_name][frame_num] = matrix
                        else:
                            print(f"Invalid matrix shape in {file_path}")
                    except Exception as e:
                        print(f"Error reading {file_path}: {str(e)}")
    return data

# object_pose_camera = parse_pose_data("newresults")
# print(object_pose_camera["bowl"][550])

def parse_poses(file_path):
    """
    Parses the given text file to extract quaternion and translation data for each frame.

    Args:
        file_path (str): Path to the image.txt file containing the pose data.
    Returns:
        dict: A dictionary where keys are frame numbers (integers) 
        and values the quaternions and translation of the camera in world coordinate frame as lists {frame_number: [[quaternionswxyz],[translation]]}
    """
    poses = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('#'):
            i += 1
            continue
        # Process the image line
        parts = line.split()
        # if its a points line or not the right length skip
        if len(parts) < 10:
            i += 1
            continue 
        
        # Extract quaternion and translation components
        try:
            qw = float(parts[1])
            qx = float(parts[2])
            qy = float(parts[3])
            qz = float(parts[4])
            tx = float(parts[5])
            ty = float(parts[6])
            tz = float(parts[7])
            filename = parts[9]
            # Extract frame number from filename (e.g., frame_0002.png -> 2)
            frame_number = int(filename.split('_')[1].split('.')[0])
            poses[frame_number] = [
                [qw, qx, qy, qz],
                [tx, ty, tz]
            ]
        except (IndexError, ValueError) as e:
            print(f"Error processing line {i+1}: {e}")
            i += 1
            continue
        
        # Skip the next line (POINTS2D data)
        i += 2

    return poses

# camera_pose_world = parse_poses('images.txt')
# print(camera_pose_world[2])

def compute_world_poses(object_pose_in_cam, camera_pose_in_world):
    """
    Computes object poses in world coordinates using:
    - object_pose_in_cam: {obj_name: {frame: 4x4_ndarray}} 
    - camera_pose_in_world: {frame: [quat_wxyz, translation]}
    
    Returns: {obj_name: [(frame, (quat_wxyz, translation))]}
    """
    world_poses = {}

    # Determine maximum frame number in camera poses
    max_frame = max(camera_pose_in_world.keys()) if camera_pose_in_world else 0

    # iterate objects
    for obj_name, obj_frames in object_pose_in_cam.items():
        # Initialize list with None for all possible frames
        obj_world = [None] * (max_frame + 1)
        # iterate poses of one object
        for frame, T_obj_cam in obj_frames.items():
            # handle possible mistakes
            if frame not in camera_pose_in_world:
                continue

            # Get camera pose components
            cam_quat_wxyz, cam_trans = camera_pose_in_world[frame]
            
            try:
                # Convert camera pose to 4x4 transformation matrix
                cam_rot = Rotation.from_quat([
                    cam_quat_wxyz[1],  # x
                    cam_quat_wxyz[2],  # y
                    cam_quat_wxyz[3],  # z
                    cam_quat_wxyz[0]   # w
                ]).as_matrix()
                
                T_cam_world = np.eye(4)
                T_cam_world[:3, :3] = cam_rot
                T_cam_world[:3, 3] = cam_trans

                # Compose transformations: T_obj_world = T_cam_world @ T_obj_cam
                T_obj_world = T_cam_world @ T_obj_cam

                # Extract rotation and translation
                world_rot = Rotation.from_matrix(T_obj_world[:3, :3])
                world_quat = world_rot.as_quat()  # [x, y, z, w]
                world_trans = T_obj_world[:3, 3].tolist()

                # Convert to wxyz format and store
                #TODO maybe [frame-1] as there is no frame 0 and we start with frame 1 but irrelevant currently as it only adds one additional at the start
                obj_world[frame] = (
                    [world_quat[3], world_quat[0], world_quat[1], world_quat[2]],
                    world_trans
                )

            except Exception as e:
                print(f"Error processing {obj_name} frame {frame}: {str(e)}")
                continue

        world_poses[obj_name] = obj_world
    
    return world_poses

# world_pose_data = compute_world_poses(object_pose_camera, camera_pose_world)
# print(len(world_pose_data["yogurt"]))
# for key, value in world_pose_data.items() :
#     print (key)

def merge_object_variants(world_pose_data):
    """
    Merges object variants (e.g., "bowl2", "bowl3") into base names ("bowl")
    assuming non-overlapping frames between variants.
    
    Args:
        world_pose_data: {obj_name: list of poses by frame}
        
    Returns:
        Merged dictionary with base object names
    """
    # Group objects by base nam        print(len(merged_data[final_name]))

    object_groups = defaultdict(list)
    
    # Split object names into base name and variant number
    for obj_name in world_pose_data.keys():
        # Split into base name and number (e.g., "bowl2" -> ("bowl", 2))
        base = obj_name.rstrip('0123456789')
        suffix = obj_name[len(base):]
        
        if suffix.isdigit():
            object_groups[base].append((int(suffix), obj_name))
        else:
            object_groups[obj_name].append((0, obj_name))  # Base object

    merged_data = {}
    
    for base_name, variants in object_groups.items():
        # Sort variants by suffix number (base first)
        variants.sort()

        
        # Create merged frame list
        max_frame = max(len(world_pose_data[v[1]]) for v in variants)
        merged_frames = [None] * max_frame

        # Check for frame overlaps and merge
        for suffix, variant_name in variants:
            variant_frames = world_pose_data[variant_name]
            
            for frame_idx, pose in enumerate(variant_frames):

                if frame_idx >= len(merged_frames):
                    continue  # Handle different length lists
                    
                if pose is not None:
                    if merged_frames[frame_idx] is not None:
                        warnings.warn(
                            f"Frame overlap detected in {base_name} variants "
                            f"at frame {frame_idx}! Data might be corrupted."
                        )
                    merged_frames[frame_idx] = pose
        
        # If base object exists, prefer its name
        final_name = base_name if (0, base_name) in variants else variants[0][1]
        merged_data[final_name] = merged_frames
    
    return merged_data

# merged_world_poses = merge_object_variants(world_pose_data)
# print(len(merged_world_poses["yogurt"]))
# for key, value in merged_world_poses.items() :
#     print (key)

def fill_missing_poses(merged_world_poses):
    """
    Fills missing poses using:
    - Leading Nones (before first detection) get first valid pose
    - Subsequent Nones carry forward last valid pose
    Modifies the input dictionary in place
    Returns: Final dictionary {obj_name:[(quat_wxyz, translation)] }
    """
    for obj_name, frame_poses in merged_world_poses.items():
        # Find first valid pose
        first_idx = None
        for idx, pose in enumerate(frame_poses):
            if pose is not None:
                first_idx = idx
                first_pose = pose
                break
        
        if first_idx is None:
            continue  # No poses to fill with
        
        # Fill leading frames (before first detection)
        for i in range(first_idx):
            frame_poses[i] = first_pose
        
        # Forward fill remaining frames
        last_valid_pose = first_pose
        for i in range(first_idx, len(frame_poses)):
            if frame_poses[i] is not None:
                last_valid_pose = frame_poses[i]
            else:
                frame_poses[i] = last_valid_pose
    
    return merged_world_poses

# filled_poses = fill_missing_poses(merged_world_poses)
# print(filled_poses["bowl"])


# Run them one after another
object_pose_camera = parse_pose_data("newresults")
camera_pose_world = parse_poses('images.txt')
world_pose_data = compute_world_poses(object_pose_camera, camera_pose_world)
merged_world_poses = merge_object_variants(world_pose_data)
final_obj_poses_world = fill_missing_poses(merged_world_poses)

# # Guide for final dict
# # this returns bowl poses of all frames (plus one additional frame pose at position 0)
# print(final_obj_poses_world["bowl"])
# # this returns bowl pose for frame 500
# print(final_obj_poses_world["bowl"][500])
# # this retruns the quaternions of bowl at frame 500 (for translation: final_obj_poses_world["bowl"][500][1])
# print(final_obj_poses_world["bowl"][500][0])
# # this retruns all keys (objects) of the dict
# for key, value in final_obj_poses_world.items():
#     print(key)



