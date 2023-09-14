import os
import glob
import shutil
import json

if __name__ == "__main__":
    
    input_dir = 'datasets/1billion_graspnet'
    
    annotation_path = os.path.join(input_dir, 'annotations/realsense')
    annotation_file_names = os.listdir(annotation_path)
    for names in annotation_file_names:
        file_path = os.path.join(annotation_path, names)
        #remove _origin in the name
        new_name = new_name = names.replace('_origin', '')        

        with open(file_path, 'r') as f:
            json_data = json.load(f)
        
        annotation_info = json_data['annotations']
        for i in range(len(annotation_info)):
            annotation_info[i]['category_id'] = 1

        category_info = json_data['categories']
        category = {'supercategory':'1', 'id':1, 'name':'1'}
        json_data['categories'] = [category]
        # for i in range(len(category_info)):
        #     if i == 0:
        #         category_info[i]['supercategory'] = "1"
        #         category_info[i]['id'] = 1
        #         category_info[i]['name'] = "1"
        #     else:
                
        
        with open(annotation_path + '/' + new_name, 'w') as f:
            json.dump(json_data, f)