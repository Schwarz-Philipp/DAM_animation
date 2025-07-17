This script creates .mp4 videos of dummy atom models in .cif or .pdb format.  

This script requires Python 3 and the following packages:  
numpy, pyvista, imageio[ffmpeg] and scikit-learn  

They can be installed with: pip install numpy pyvista imageio[ffmpeg] scikit-learn

To use this script simply run it and when prompted enter either a folder path or a file path to a .cif or .pdb file.  
Example folder path: C:\Users\Username\DAM_models  
Example file path: C:\Users\Username\DAM_models\model.cif  

The following settings can changed by editing the DAM_animation.py file:  
video_duration  
video_fps  
show_bounding_box  
number_of_rotations...the amount of rotations the model does in the video   
norm_occupancy_threshold...threshold for the normalized occupancy, dummy atoms with an occupancy below the threshold will be ignored  
sphere_size...size of the spheres representing the dummy atoms  
zoom_factor...zoom the camera further in or out  
view = Change the view of the animation. Either use 'iso' or specify 2 axes e.g. 'yx'.  
When specifying 2 axes, the first axis (e.g. y) is horizontal and the second axis (e.g x) is vertical  
The x-axis corresponds to the direction of the longest particle dimension  
The z-axis corresponds to the direction of the shortest particle dimension  
May require changing the zoom factor