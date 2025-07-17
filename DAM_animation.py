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

#--------SETTINGS------------------------------
video_duration = 10 #in seconds
video_fps = 30
show_bounding_box = True
number_of_rotations = 1
norm_occupancy_threshold = 0.5
sphere_size = 50
zoom_factor = 1 #values >1 zoom in, values <1 zoom out
view = 'iso' #Either use 'iso' or specify 2 axes e.g. 'yx'
#When specifying 2 axis, the first axis (e.g. y) is horizontal and the second axis (e.g x) is vertical
# The x-axis corresponds to the direction of the longest particle dimension
# The z-axis corresponds to the direction of the shortest particle dimension
# May require changing the zoom factor
#-----------------------------------------------

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
                print("Error: The input is a file but does not have a .cif extension. Please enter a .cif file or a folder.")
        elif os.path.isdir(rawfileinput):
            print(f"Input is a folder: {os.path.basename(rawfileinput)}")
            cif_files = find_files(rawfileinput, '*.cif')
            files.extend(cif_files)
            pdb_files = find_files(rawfileinput, '*.pdb')
            files.extend(pdb_files)
            break 
        else:
            print("Error: The input exists but is neither a file nor a directory. Please enter a .cif file or a folder.")
    else:
        print("Error: The specified path does not exist. Please enter a valid path to a .cif file or a folder.")

for file_path in files:
    folder_path = os.path.dirname(file_path)
    base = os.path.basename(file_path)
    filename, file_ext = os.path.splitext(base)
    output_video = folder_path+'\\'+filename+'.mp4'
    
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

    pca = PCA(n_components=3)
    pca.fit(filtered_coords)
    transformed_coords = pca.transform(filtered_coords)
    
    min_extents = transformed_coords.min(axis=0)
    max_extents = transformed_coords.max(axis=0)
    side_lengths = max_extents - min_extents
    
    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = 'white'

    atom_cloud = pv.PolyData(transformed_coords)
    plotter.add_mesh(atom_cloud,
                     render_points_as_spheres=True,
                     scalars=filtered_occupancy,
                     cmap='viridis',
                     point_size=sphere_size*zoom_factor,
                     scalar_bar_args={'title': 'Normalized Occupancy'})
    
    if show_bounding_box:
        box = pv.Box(bounds=(min_extents[0], max_extents[0],
                       min_extents[1], max_extents[1],
                       min_extents[2], max_extents[2]))
        
        plotter.add_mesh(box, style='wireframe', color='black', line_width=2)
        box_center = (min_extents + max_extents) / 2
        principal_axes = np.identity(3)
        
        offset = 0.5
        edge_centers = [
        box_center+principal_axes[1]*side_lengths[1]*offset+principal_axes[2]*side_lengths[2]*offset,
        box_center+principal_axes[0]*side_lengths[0]*offset+principal_axes[2]*side_lengths[2]*offset,
        box_center+principal_axes[0]*side_lengths[0]*offset+principal_axes[1]*side_lengths[1]*offset
        ]
        
        for i in range(3):
            label = f"{side_lengths[i]:.0f} Å"
            plotter.add_point_labels(edge_centers[i], [label],
                                     font_size=20, text_color='red',
                                     shape=None, show_points=False, always_visible=True)
    
    plotter.camera_position = view
    plotter.reset_camera_clipping_range()
    plotter.camera.zoom(zoom_factor)
    print(f"Generating animation for {filename+file_ext}...")
    plotter.open_movie(output_video, framerate=video_fps)
    
    n_frames = int(video_duration * video_fps)
    angle_step = (360.0 * number_of_rotations) / n_frames
    
    plotter.show(auto_close=False)
    
    for i in range(n_frames):
        plotter.camera.Azimuth(angle_step)
        plotter.write_frame()
    
    plotter.close()
    print(f"Video saved successfully to {output_video}")
