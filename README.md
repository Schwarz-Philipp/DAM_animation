This script creates .mp4 videos of dummy atom models in .cif or .pdb format.  

This script requires Python 3 and the following packages:  
numpy, pyvista and scikit-learn  

They can be installed with: pip install numpy pyvista scikit-learn

To use this script simply run it and when prompted enter either a folder path or a file path to a .cif or .pdb file.

The following settings can changed by editing the DAM_animation.py file:  
video_duration...duration of the video in seconds (default: 10)  
video_fps...frames per second of the video (default: 30)  
show_bounding_box...toggle if the bounding box should be shown in the video (default: True)  
number_of_rotations...choose how many rotations the model should do in total (default: 1)  
norm_occupancy_threshold...set threshold for the normalized occupancy, dummy atoms with an occupancy below the threshold will be ignored (default: 0.5)  
sphere_size...set the size of the spheres representing the dummy atoms (default: 50)  
zoom_factor...choose if the camera should zoom further in or out (default: 1)  
