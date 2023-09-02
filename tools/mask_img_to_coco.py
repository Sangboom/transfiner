import os
import json
import cv2
import numpy as np
from PIL import Image
from sub_masks_annotations import create_sub_masks, create_sub_mask_annotation
import time
import random
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', help='train or test')
    parser.add_argument('--camera', type=str, default='kinect', help='kinect or realsense')
    args = parser.parse_args()
    
    input_dir = 'datasets/1billion_graspnet'
    
    # Generate the categories
    class_id_file = open(input_dir + '/object_id_list.txt')
    line = class_id_file.readline()
    count = 0
    category_id = 0
    categories = []
    while line:
        category_id += 1
        category = {'supercategory':line, 'id':category_id, 'name':line}
        categories.append(category)
        line = class_id_file.readline()
    class_id_file.close()
    
    # Read the names of the image to generate annotation
    image_names = []
    scene_list = os.listdir(input_dir + '/scenes')
    # remove .zip file from scene_list and remove .txt file from scene_list
    scene_list = [scene for scene in scene_list if scene.find('.zip') == -1 and scene.find('.txt') == -1]
    scene_list = sorted(scene_list)

    if args.split == 'train':
        scene_list = scene_list[0:100]
    elif args.split == 'test_seen':
        scene_list = scene_list[100:130]
    else:
        scene_list = scene_list[130:]

    for scene in scene_list:
        image_path = 'scenes/' + scene + '/' + args.camera + '/rgb'
        image_id_list = os.listdir(input_dir + '/' + image_path)
        image_id_list = sorted(image_id_list)

        for i in range(len(image_id_list)):
            image_names.append(image_path + '/' + image_id_list[i].split('.')[0])    
        
    num_of_images = len(image_names)
    # random.seed(0)
    # image_id_index = random.sample([i for i in range(0, num_of_images)], num_of_images)
    image_id_index = [i for i in range(0, num_of_images)]
 
    image_dir = input_dir + '/' + args.camera
    width = 1280
    height = 720
    iscrowd = 0
    annotation_id = 0
    annotations = []
    images = []
    image_count = -1
    count = 0
    
    for image_name in image_names:
        
        start_time = time.time()
        print('Procssing: ', image_name, '...')
        
        # Write information of each image
        file_name = image_name + '.png'
        image_count += 1
        image_id = image_id_index[image_count]

        image_item = {'file_name':file_name, 'height':height, 'id':image_id, 'width':width}
        images.append(image_item)
        
        # write information of each mask in the image
        # change rgb to label in file_name
        mask_name = file_name.replace('rgb', 'label')
        
        image = Image.open(input_dir + '/' + mask_name)

        sub_masks = create_sub_masks(image)

        # for key in sub_masks.keys():
        #     print(sub_masks[key])
        #     # sub_masks_image = Image.open(sub_masks[key])
        #     # sub_masks[key].show()
        #     print(sub_masks[key].size)
        # exit()
        
        count = count + len(sub_masks)
        
        for category_id, sub_mask in sub_masks.items():
            category_id = int(category_id[1:category_id.find(',')])
            annotation_id += 1
            cimg = np.array(sub_mask)
            opencvImage = np.stack((cimg, cimg, cimg), axis=2)
            instance = np.uint8(np.where(opencvImage == True, 0, 255))
            annotation_item = create_sub_mask_annotation(instance, image_id, category_id, annotation_id, iscrowd)
            annotations.append(annotation_item)
        
        print('Done! Time: ', time.time() - start_time)
        
    print('Test if all the instances are detected, the result is', count == annotation_id)
    
    # Combine categories, annotations and images to form a json file
    json_data = {'annotations':annotations, 'categories':categories, 'images':images}
    annotations_output_dir = input_dir + '/annotations/' + args.camera
    if not os.path.exists(annotations_output_dir):
                os.makedirs(annotations_output_dir)
    # train.json or test.json
    with open(annotations_output_dir + '/' + args.split + '.json', 'w') as f:
        json.dump(json_data, f)
