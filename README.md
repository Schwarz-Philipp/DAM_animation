This Python script generates animated MP4 videos of Dummy Atom Models (DAM) from .cif or .pdb files. The script orients the model along its principal axes and creates a 360-degree rotation video, saving it in the same directory as the input file.

REQUIREMENTS

The script requires Python 3 and the following packages:  
- numpy  
- pyvista  
- scikit-learn  
- imageio (with FFmpeg support)  

You can install all required packages with a single command:  
pip install numpy pyvista scikit-learn "imageio[ffmpeg]"

USAGE

1.  Run the script from your terminal.  
2.  When prompted, provide the full path to either a single .cif/.pdb file or a folder containing such files.  

    - Example file path: C:\Users\Username\DAM_models\model.cif  
    - Example folder path: C:\Users\Username\DAM_models

3.  The script will process the file(s) and save the resulting .mp4 video in the same directory.

CONFIGURATION

You can easily customize the video output and model appearance by modifying the variables in the SETTINGS block at the top of the DAM_animation.py script.

Key customizable options include:  
- Video duration, FPS, and zoom level.  
- Number of rotations.  
- Model properties like sphere size and colormap.  
- Toggling the visibility of the bounding box and its labels.  
- Adjusting the title and appearance of the scalar bar.  