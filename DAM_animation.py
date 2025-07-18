# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 10:32:02 2025

@author: Philipp Schwarz
"""

import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA # Principal component analysis for the bounding box
import os 
import glob
import vtk

#--------SETTINGS--------------------------------------------------------------
# Video Settings
video_duration = 10     # Duration of the output video in seconds.
video_fps = 30          # Frames per second for the output video.
zoom_factor = 0.85      # Camera zoom. Values > 1 zoom in, values < 1 zoom out.
view = 'yx'             # Initial camera view. Use 'iso' for isometric or specify two axes (e.g., 'yx').
# For two-axis views, the first axis is horizontal, and the second is vertical.
# The axes correspond to the principal components of the particle:
# x-axis: longest dimension, y-axis: intermediate, z-axis: shortest.

# Model Settings
number_of_rotations = 1         # Number of full 360-degree rotations in the video.
norm_occupancy_threshold = 0.5  # Normalized occupancy threshold (0.0 to 1.0). Atoms below this value will be hidden.
sphere_size = 50                # Size of the spheres representing dummy atoms.
model_cmap = 'viridis'          # Colormap for dummy atom occupancy. Search "matplotlib colormaps" online for more options.
flip_on_head = False            # Flip the model upside down.
# Offset the model from the center (in Ångströms).
x_offset = 0  
y_offset = 0   
z_offset = 0   

# Bounding Box Settings
show_bounding_box = True # Display the bounding box around the model.
bb_label_font_size = 20  # Font size for the bounding box dimension labels.
bb_label_color = 'red'   # Color for the bounding box dimension labels.
bb_label_bold = True     # Make the bounding box dimension labels bold.

# Scalar Bar Settings
sb_title = 'Normalized Occupancy' # Title displayed above the scalar bar.
sb_title_bold = False   # Make the scalar bar title bold.
sb_title_y = 0.06       # Vertical position of the scalar bar title (as a fraction of viewport height).
sb_title_font_size = 20 # Font size for the scalar bar title.
sb_label_font_size = 16 # Font size for the scalar bar labels (the numbers).
sb_label_bold = False   # Make the scalar bar labels bold.
sb_text_color = 'black' # Color for all scalar bar text (title and labels).

sb_args = {
    # Position of the scalar bar as a fraction of the viewport.
    'width': 0.5,       # 50% of the viewport width.
    'height': 0.05,     # 5% of the viewport height.
    'position_y': 0.01, # Position near the bottom.
    'position_x': 0.25, # Center horizontally with: 0.5 - (width / 2)

    # Formatting for the numeric labels on the scalar bar.
    'fmt': '%.2f',      # Format labels to two decimal places.
    
    # It is not recommended to change the settings below.
    'title': '', 
    'label_font_size': sb_label_font_size,
    'color': sb_text_color,
    'bold': sb_label_bold,
    'vertical': False,  
}
#------------------------------------------------------------------------------

def find_files(directory: str, file_pattern: str = "*.dat") -> list:
    """
    Finds all files in the given directory matching the file pattern.
    """
    search_pattern = os.path.join(directory, file_pattern)
    files = glob.glob(search_pattern)
    return files

files = []

while True: # Loop until valid input is received
    fileinput = input('Enter .cif file, .pdb file or folder: ')
    rawfileinput = fr'{fileinput}'
    if os.path.exists(rawfileinput):
        if os.path.isfile(rawfileinput):
            if rawfileinput.lower().endswith('.cif') or rawfileinput.lower().endswith('.pdb'):
                print(f"Input is a file: {os.path.basename(rawfileinput)}")
                files.append(rawfileinput)
                break 
            else:
                print("Error: The input is a file but does not have a .cif or .pdb extension..")
        elif os.path.isdir(rawfileinput):
            print(f"Input is a folder: {os.path.basename(rawfileinput)}")
            cif_files = find_files(rawfileinput, '*.cif')
            files.extend(cif_files)
            pdb_files = find_files(rawfileinput, '*.pdb')
            files.extend(pdb_files)
            break 
        else:
            print("Error: The input exists but is neither a file nor a directory. Please enter a .cif file, .pdb file or a folder.")
    else:
        print("Error: The specified path does not exist. Please enter a valid path to a .cif file, .pdb file or a folder.")

for file_path in files:
    folder_path = os.path.dirname(file_path)
    base = os.path.basename(file_path)
    filename, file_ext = os.path.splitext(base)
    output_video = os.path.join(folder_path, filename + '.mp4')
    
    coords_list = []
    occupancy_list = []
    if file_path.lower().endswith('.cif'):
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    parts = line.split()
                    x, y, z = float(parts[9]), float(parts[10]), float(parts[11])
                    occupancy = float(parts[12])
                    coords_list.append([x, y, z])
                    occupancy_list.append(occupancy)
    
    if file_path.lower().endswith('.pdb'):
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    parts = line.split()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    occupancy = float(line[54:60])
                    coords_list.append([x, y, z])
                    occupancy_list.append(occupancy)
    
    coords = np.array(coords_list)
    occupancy_values = np.array(occupancy_list)

    if coords.shape[0] == 0:
        raise ValueError(f"No ATOM lines found in {filename+file_ext}. Parsing failed.")
    
    norm_occupancy = occupancy_values / occupancy_values.max() 
    mask = norm_occupancy >= norm_occupancy_threshold

    filtered_coords = coords[mask]
    filtered_occupancy = norm_occupancy[mask]

    pca = PCA(n_components=3, random_state=42)
    pca.fit(filtered_coords)
    transformed_coords = pca.transform(filtered_coords)
    
    # Calculate the geometric center of the oriented model
    min_b = transformed_coords.min(axis=0)
    max_b = transformed_coords.max(axis=0)
    bbox_center = (min_b + max_b) / 2
    
    # Re-center the model on its geometric center, then apply the manual offset
    offset_vector = np.array([x_offset, y_offset, z_offset])
    transformed_coords = transformed_coords - bbox_center + offset_vector
    
    min_extents = transformed_coords.min(axis=0)
    max_extents = transformed_coords.max(axis=0)
    side_lengths = max_extents - min_extents
    
    # For Debugging
    # plotter = pv.Plotter(off_screen=False)
    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = 'white'

    atom_cloud = pv.PolyData(transformed_coords)
    plotter.add_mesh(atom_cloud,
                     render_points_as_spheres=True,
                     scalars=filtered_occupancy,
                     cmap=model_cmap,
                     point_size=sphere_size*zoom_factor,
                     scalar_bar_args=sb_args)
    
    x_frac_center = 0.50  # 50% from left (horizontal center)
    y_frac_bottom = sb_title_y
    window_x, window_y = plotter.window_size
    pixel_x = int(window_x * x_frac_center)
    pixel_y = int(window_y * y_frac_bottom)
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput(sb_title)
    text_prop = text_actor.GetTextProperty()
    text_prop.SetFontSize(sb_title_font_size)
    text_prop.SetBold(sb_title_bold)
    text_prop.SetColor(*pv.Color(sb_text_color).float_rgb)
    text_prop.SetJustificationToCentered()
    coord = text_actor.GetPositionCoordinate()
    coord.SetCoordinateSystemToDisplay()
    coord.SetValue(pixel_x, pixel_y)
    plotter.add_actor(text_actor)
    
    if show_bounding_box:
        box = pv.Box(bounds=(min_extents[0], max_extents[0],
                       min_extents[1], max_extents[1],
                       min_extents[2], max_extents[2]))
        
        plotter.add_mesh(box, style='wireframe', color='black', line_width=2)
        box_center = (min_extents + max_extents) / 2
        principal_axes = np.identity(3)
        
        flip_sign = 1
        if flip_on_head:
            flip_sign = -1
        
        offset = 0.5
        edge_centers = [
        box_center+principal_axes[1]*side_lengths[1]*offset+principal_axes[2]*side_lengths[2]*offset,
        box_center+principal_axes[0]*side_lengths[0]*offset*flip_sign+principal_axes[2]*side_lengths[2]*offset,
        box_center+principal_axes[0]*side_lengths[0]*offset*flip_sign+principal_axes[1]*side_lengths[1]*offset
        ]
        
        for i in range(3):
            label = f"{side_lengths[i]:.0f} Å"
            plotter.add_point_labels(edge_centers[i], [label],
                                     font_size=bb_label_font_size, text_color=bb_label_color,
                                     shape=None, show_points=False, always_visible=True, bold=bb_label_bold)
    
    plotter.camera_position = view
    plotter.camera.focal_point = [0.0, 0.0, 0.0]
    plotter.reset_camera_clipping_range()
    plotter.camera.zoom(zoom_factor)
    if flip_on_head:
        current_up = plotter.camera.up
        plotter.camera.up = [-current_up[0], -current_up[1], -current_up[2]]
    
    print(f"Generating animation for {filename+file_ext}...")
    plotter.open_movie(output_video, framerate=video_fps)
    
    n_frames = int(video_duration * video_fps)
    angle_step = (360.0 * number_of_rotations) / n_frames
    # For Debugging
    # plotter.show(auto_close=False)
    
    for i in range(n_frames):
        plotter.camera.Azimuth(angle_step)
        plotter.write_frame()
    
    plotter.close()
    print(f"Video saved successfully to {output_video}")
