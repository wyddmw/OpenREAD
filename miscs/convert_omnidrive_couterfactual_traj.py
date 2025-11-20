import pickle
import numpy as np
import sys
import os
import json
import re
from tqdm import tqdm

driving_command = {
    0: "Turn Left",
    1: "Turn Right",
    2: "Go Straight"
}

def separate_first_sentence(text):
    """Separate the first sentence from the rest of the text."""
    # Find the first occurrence of '. ' (period followed by space)
    first_period_idx = text.find('. ')
    
    if first_period_idx != -1:
        first_sentence = text[:first_period_idx + 1]  # Include the period
        rest = text[first_period_idx + 2:]  # Skip the '. '
        return first_sentence, rest
    else:
        # If no '. ' found, return the whole string as first sentence
        return text, ""

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
        if not os.path.exists(anno_file_path):
            print(f"Warning: Annotation file {anno_file_path} does not exist, skipping...")
            # make sure the historical image list is synchronized
            prev_images.pop(0)
            prev_images.append(cam_path)
            continue

        anno_list = json.load(open(anno_file_path, 'r'))
        anno_tmp = {}
        anno_tmp["image"] =  prev_images + [cam_path] + [lidar_bev_path]
        anno_tmp["scene_token"] = scene_token
        assert len(anno_tmp["image"]) == 5
        navi_command = driving_command[int(info['gt_planning_command'])]

        if velocity is None:
            # make sure the historical image list is synchronized
            prev_images.pop(0)
            prev_images.append(cam_path)
            continue
        else:
            question = f"The first four images depict historical and current front views, and the last image is the current LiDAR bird's-eye view. Your navigation command is {navi_command}. Your current velocity is {velocity:.2f} m/s, and your acceleration speed is {acc:.2f} m/s^2. Please plan a trajectory for the next three seconds."
        if len(anno_list) != 4:
            # make sure the historical image list is synchronized
            prev_images.pop(0)
            prev_images.append(cam_path)
            continue
        pseudo_traj = anno_list[2]
        anno_action = pseudo_traj
        if "If you follow the trajectory" in anno_action['question']:
            question = anno_action['question']
            question  = re.sub(r'\[PT,\s*', '', question)
            question = re.sub(r'\]', '', question)
            question = question.replace(', what would happen?', ' in the next three seconds, what would happen?')
            answer = anno_action['answer']
            conclusion, reasoning = separate_first_sentence(answer)
            if args.mode == 'grpo':
                cot_answer = f"<think> {reasoning} </think> \n <answer> {conclusion} </answer>"
                anno_tmp['conversations'] = [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": cot_answer}
                ]
            else:
                answer = f"{conclusion}"
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
    parser.add_argument("--ego_pickle_file", default='/home/spyder/data2/LLM-Datasets/OmniDrive/vqa_annos/data_nusc/nuscenes2d_ego_temporal_infos_val.pkl', type=str, help="Path to the pickle file")
    parser.add_argument("--anno_path", default='/home/spyder/data2/LLM-Datasets/OmniDrive/vqa_annos/data_nusc/vqa/val', help="Path to the annotation directory")
    parser.add_argument("--mode", type=str, choices=['grpo', 'sft'], default='grpo')
    parser.add_argument("--ego_status", type=str, default='/home/spyder/data2/LLM-Datasets/OmniDrive/vqa_annos/ego_status.json')
    parser.add_argument("--saving_file", type=str, default='counterfactual_traj_val_ego_status.json', help="Path to the saving file")
    
    args = parser.parse_args()
    load_and_explore_pickle(args)