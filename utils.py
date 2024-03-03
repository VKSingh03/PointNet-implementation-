import os
import torch
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
import imageio
from PIL import Image
import numpy as np

def save_checkpoint(epoch, model, args, best=False):
    if best:
        path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else:
        path = os.path.join(args.checkpoint_dir, 'model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def viz_seg (verts, path, device, labels = None, points=None, class_name=None, viz = "cls"): # added parameter of points to accept different no of points and viz for the classification/segmentation task
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)

    colors = [[1.0,1.0,1.0], [1.0,0.0,1.0], [0.0,1.0,1.0],[1.0,1.0,0.0],[0.0,0.0,1.0], [1.0,0.0,0.0]]
    class_color = {"vase": [1.0, 0.0, 0.0], "chair": [1.0, 1.0, 0.0], "lamp": [1.0, 0.0, 1.0]}

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.unsqueeze(0).repeat(30,1,1).to(torch.float)

    if(viz == "seg"): # add different colours for different segments in segmentation task. 
        sample_labels = labels.unsqueeze(0)
        sample_colors = torch.zeros((1,points,3))
        # Colorize points based on segmentation labels
        for i in range(6):
            sample_colors[sample_labels==i] = torch.tensor(colors[i])
        sample_colors = sample_colors.repeat(30,1,1).to(torch.float)
    else: 
        sample_colors = torch.tensor(class_color[class_name]).repeat(1,sample_verts.shape[1],1).repeat(30,1,1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3) 
    
    images = []
    for r in rend:
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    imageio.mimsave(path, images, duration=60, loop = 0)

def rotate_pc(data, rotation_angles):
    # Reference used for this code: https://www.brainm.com/software/pubs/math/Rotation_matrix.pdf
    
    # Convert angles from degrees to radians
    angles = np.radians(rotation_angles)

    # Define rotation matrices for x, y, z axes
    Rx = torch.tensor([[1, 0, 0],
                    [0, np.cos(angles[0]), -np.sin(angles[0])],
                    [0, np.sin(angles[0]), np.cos(angles[0])]], dtype=torch.float)

    Ry = torch.tensor([[np.cos(angles[1]), 0, np.sin(angles[1])],
                    [0, 1, 0],
                    [-np.sin(angles[1]), 0, np.cos(angles[1])]], dtype=torch.float)

    Rz = torch.tensor([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                    [np.sin(angles[2]), np.cos(angles[2]), 0],
                    [0, 0, 1]], dtype=torch.float)

    # Combine the rotation matrices
    rotation_matrix = Rz @ Ry @ Rx

    # Apply rotation to the point cloud
    rotated_data = torch.matmul(rotation_matrix, data.transpose(1, 2)).transpose(1, 2)

    return rotated_data

# function to write the experiment results to file. 
def write_experiment_results(file_path, exp_name, num_points, rotation_params, test_accuracy, class_name = None):
    with open(file_path, 'a') as file:
        file.write("------------------------------------------\n")
        file.write(f"Experiment {exp_name}\n")
        if(class_name == None):
            file.write(f"Number of Point {num_points} Rotation X:{rotation_params[0]} Y:{rotation_params[1]} Z:{rotation_params[2]}\n")
        else:
            file.write(f"Class {class_name} Number of Point {num_points} Rotation X:{rotation_params[0]} Y:{rotation_params[1]} Z:{rotation_params[2]}\n")
        file.write(f"Test Accuracy: {test_accuracy}.\n")