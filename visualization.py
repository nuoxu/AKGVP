import os
import cv2
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pdb

if __name__ == "__main__":
    vis_save_dir = "visualization_files"
    scene_dir = "datasets/Scene_Data"
    action_list = ['MoveAhead','RotateLeft','RotateRight','LookDown','LookUp','Done']

    json_name = 'AKGVPModel'
    with open(os.path.join(vis_save_dir, json_name+'.json'),'r') as load_f:
        load_dict_list = json.load(load_f)

    for navigation in load_dict_list:

        start_pos = navigation['states'][0].replace('|','#').replace('.','-')
        if_success = 'success' if navigation['success'] else 'fail'
        goal_name = navigation['target'][0].split('|')[0]
        scene_name = navigation['scene']

        video_name = scene_name+'_'+start_pos+'_'+goal_name+'_'+if_success+'.mp4'

        images_path = os.path.join(scene_dir, scene_name, "images.hdf5")
        video_path = os.path.join(vis_save_dir+'/'+json_name, video_name)
        images_file = h5py.File(images_path, 'r')

        videowrite = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (300, 480))
        fig = plt.figure()
        plt.figure(figsize=(3, 1.75))
        index_action = np.arange(len(action_list))

        for idx, img_id in enumerate(navigation['states']):
            img = np.array(images_file[img_id])
            img = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)

            det = navigation['detection_results'][idx]
            cv2.rectangle(img, (int(det[0]),int(det[1])), (int(det[2]),int(det[3])), (0,255,0), 2)

            action = navigation['action_outputs'][idx][0]
            plt.bar(index_action, height=action, width=0.3, color='b')
            plt.ylim(0, 1)
            plt.xticks(index_action, action_list, rotation=30)
            plt.savefig('temp.jpg', bbox_inches = 'tight')
            plt.clf()
            action_bar = cv2.imread('temp.jpg')
            action_bar = cv2.resize(action_bar, (300, 180))

            img = np.vstack((img, action_bar))

            videowrite.write(img)

        videowrite.release()
        images_file.close()
        plt.close(fig)

        print(video_path)