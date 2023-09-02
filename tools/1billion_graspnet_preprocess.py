import os
import glob
import shutil

if __name__ == "__main__":
    
    input_dir = 'datasets/1billion_graspnet'
    
    # change folder structure to fit the COCO format
    scene_list = glob.glob(input_dir + '/scenes/*')