import pickle
import numpy as np
import sys
import os
import json
from tqdm import tqdm

driving_command = {
    0: "Turn Left",
    1: "Turn Right",
    2: "Go Straight"
}
def load_and_explore_pickle(args):
    """Load and explore the contents of a pickle file."""
    file_path = args.ego_pickle_file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    data_info_list = data['infos']
    training_data_list = []
    prev_images = []
    new_sequence = 0
    ego_status_dict = json.load(open(args.ego_status, 'r'))

    for info in tqdm(data_info_list):
        # load the current image path
        scene_token = info['token']
        ego_status = ego_status_dict.get(scene_token, None)
        assert ego_status is not None, f"Ego status for scene token {scene_token} not found."
        velocity = ego_status['velocity']
        acc = ego_status['accel']
        
        cam_path = info['cams']['CAM_FRONT']['data_path'].split('./')[1]
        lidar_path = info['lidar_path']
        lidar_bev_path = lidar_path.replace('LIDAR_TOP', 'LIDAR_BEV').replace('.pcd.bin', '.png')
        # obtain the target future trajectory
        gt_traj, gt_mask = info['gt_planning'], info['gt_planning_mask']
        gt_traj = np.round(gt_traj[:, :6, :2], 2)
        gt_traj = gt_traj.reshape(-1).tolist()
        gt_mask = gt_mask[:, :6]
        
        if info['prev'] == '':
            # the start of a new sequence
            print('start a new sequence')
            new_sequence += 1
            prev_images = [cam_path] * 3
        
        if not gt_mask.all():
            # make sure the historical image list is synchronized
            prev_images.pop(0)
            prev_images.append(cam_path)
            continue
        
        anno_token_id = info['token']
        anno_file_path = os.path.join(args.anno_path, anno_token_id+'.json')
        if not os.path.exists(anno_file_path) and args.mode == 'grpo':
            print(f"Warning: Annotation file {anno_file_path} does not exist, skipping...")
            # make sure the historical image list is synchronized
            prev_images.pop(0)
            prev_images.append(cam_path)
            continue
        if args.mode == 'grpo':
            anno_list = json.load(open(anno_file_path, 'r'))
        
        anno_tmp = {}
        anno_tmp["image"] =  prev_images + [cam_path] + [lidar_bev_path]
        anno_tmp["scene_token"] = scene_token
        assert len(anno_tmp["image"]) == 5
        navi_command = driving_command[int(info['gt_planning_command'])]
        if velocity is None:
            question = f"The first four images depict historical and current front views, and the last image is the current LiDAR bird's-eye view. Your navigation command is {navi_command}. Please plan a trajectory for the next three seconds."
        else:
            question = f"The first four images depict historical and current front views, and the last image is the current LiDAR bird's-eye view. Your navigation command is {navi_command}. Your current velocity is {velocity:.2f} m/s, and your acceleration speed is {acc:.2f} m/s^2. Please plan a trajectory for the next three seconds."
        gt_traj_answer = f"({gt_traj[0]}, {gt_traj[1]}), ({gt_traj[2]}, {gt_traj[3]}), ({gt_traj[4]}, {gt_traj[5]}), ({gt_traj[6]}, {gt_traj[7]}), ({gt_traj[8]}, {gt_traj[9]}), ({gt_traj[10]}, {gt_traj[11]})"
        if args.mode == 'grpo':
            for anno in anno_list:
                anno_action = anno
                if "What should be your next move and why" in anno['question']:
                    planned_action = anno_action['answer'].split('.')[0] + '.'
                    # thinking = anno_action['answer'].split(answer)[1]
                    cot_answer = f"<think> {planned_action} </think> \n <answer> {gt_traj_answer} </answer>"
                    anno_tmp['conversations'] = [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": cot_answer}
                    ]
                    training_data_list.append(anno_tmp)

        elif args.mode == 'sft':
            answer = f"{gt_traj_answer}"
            anno_tmp['conversations'] = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer}
            ]
            training_data_list.append(anno_tmp)
        # FIFO
        prev_images.pop(0)
        prev_images.append(cam_path)
            
    print('Total new sequences:', new_sequence)
    # saving the processed data
    json.dump(training_data_list, open(args.saving_file, 'w'), indent=4)
    print(f"Saved processed data to {args.saving_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and explore a pickle file")
    parser.add_argument("--ego_pickle_file", default='/home/spyder/data2/LLM-Datasets/OmniDrive/vqa_annos/data_nusc/nuscenes2d_ego_temporal_infos_train.pkl', type=str, help="Path to the pickle file")
    parser.add_argument("--anno_path", default='/home/spyder/data2/LLM-Datasets/OmniDrive/vqa_annos/data_nusc/vqa/train', help="Path to the annotation directory")
    parser.add_argument("--mode", type=str, choices=['grpo', 'sft'], default='sft')
    parser.add_argument("--ego_status", type=str, default='/home/spyder/data2/LLM-Datasets/OmniDrive/vqa_annos/ego_status.json')
    parser.add_argument("--saving_file", type=str, default='traj_train_sft_bev_ego_status_temporal_fixed.json', help="Path to the saving file")
    
    args = parser.parse_args()
    load_and_explore_pickle(args)