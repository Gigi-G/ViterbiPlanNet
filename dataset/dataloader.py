import numpy as np
import itertools
from torch.utils.data import Dataset
import torch
import json

class PlanningDataset(Dataset):
    def __init__(
        self, 
        video_list, 
        horizon = 3, 
        num_action = 133, 
        aug_range = 0, 
        M = 2, 
        mode = "train",
        PKG_labels = False
    ):
        super().__init__()
        self.aug_range = aug_range
        self.horizon = horizon
        self.video_list = video_list
        self.max_duration = 0
        self.mode = mode
        self.M = M
        self.num_action = num_action
        self.transition_matrix = np.zeros((num_action, num_action))
        self.PKG_labels = PKG_labels
        self.data = []
        self.load_data()
        if self.mode == "train":
            self.transition_matrix = self.cal_transition(self.transition_matrix)

    def cal_transition(self, matrix):
        ''' Cauculate transition matrix

        Args:
            matrix:     [num_action, num_action]

        Returns:
            transition: [num_action, num_action]
        '''
        for row in range(matrix.shape[0]):
            if np.sum(matrix[row, :]) > 0:
                matrix[row, :] = matrix[row, :] / np.sum(matrix[row, :])
        return matrix


    def load_data(self):
        with open(self.video_list, "r") as f:
            video_info_dict = json.load(f)
        
        for video_info in video_info_dict:
            if "vid" in video_info["id"]:
                video_id = video_info["id"]["vid"]
            else:
                video_id = video_info["id"]["feature"].split("/")[-1].split(".")[0]
            length_video = video_info["instruction_len"]
            video_anot = video_info["id"]["legal_range"]
            task_id = video_info["id"]["task_id"]
            saved_features = np.load(video_info["id"]["feature"], allow_pickle=True)["frames_features"]
            
            if self.PKG_labels:
                if "graph_action_path" in video_info["id"]:
                    PKG_labels = video_info["id"]["graph_action_path"]
                if "LLM_labels" in video_info:
                    LLM_labels = video_info["id"]["LLM_labels"]
                else:
                    LLM_labels = video_info["id"]["graph_action_path"]
                        
            ## update transition matrix
            if self.mode == "train":
                for i in range(len(video_anot)-1):
                    cur_action = video_anot[i][-1]
                    next_action = video_anot[i+1][-1]
                    self.transition_matrix[cur_action, next_action] += 1

            all_features = []
            all_action_ids = []
            for i in range(len(video_anot)):
                cur_action_id = video_anot[i][-1]
                
                features = []
                ## Using adjacent frames for data augmentation
                for frame_offset in range(-self.aug_range, self.aug_range+1):
                    s_time = video_anot[i][0] + frame_offset
                    e_time = video_anot[i][1] + frame_offset

                    if s_time < 0 or e_time >= saved_features.shape[0]:
                        continue
                    
                    s_offset_start = max(0, s_time-self.M//2)
                    s_offset_end = min(s_time+self.M//2+1,saved_features.shape[0])
                    e_offset_start = max(0, e_time-self.M//2)
                    e_offset_end = min(e_time+self.M//2+1,saved_features.shape[0])

                    start_feature = saved_features[s_offset_start:s_offset_end]
                    end_feature = saved_features[e_offset_start:e_offset_end]

                    while start_feature.shape[0] < self.M + 1:
                        # Replay the last frame if the video is too short
                        start_feature = np.concatenate([start_feature, start_feature[-1:]], axis = 0)
                    while end_feature.shape[0] < self.M + 1:
                        # Replay the last frame if the video is too short
                        end_feature = np.concatenate([end_feature, end_feature[-1:]], axis = 0)

                    features.append(np.stack((start_feature, end_feature)))

                if len(features) == 0:
                    # Add the previous feature if no empty features
                    if len(all_features) > 0:
                        features = all_features[-1]
                        cur_action_id = all_action_ids[-1]
                
                if len(features) > 0:
                    all_features.append(features)
                    all_action_ids.append(cur_action_id)

            if all_features == []:
                print(f"Video {video_id} has no valid features, skipping.")
                continue
            
            # Append the last action id if not already included
            while len(all_features) < self.horizon:
                all_features.append(all_features[-1])
                all_action_ids.append(all_action_ids[-1])
            
            ## permutation of augmented features, action ids and prompts
            aug_features = itertools.product(*all_features)
            
            if self.PKG_labels:
                self.data.extend([{"states": np.stack(f),
                                "actions": np.array(all_action_ids), 
                                "tasks": np.array(task_id),
                                "video_id": np.array(video_id),
                                "video_len": np.array(length_video),
                                "PKG_labels": np.array(PKG_labels),
                                "LLM_labels": np.array(LLM_labels)}
                                for f in aug_features])
            else:
                self.data.extend([{"states": np.stack(f),
                                "actions": np.array(all_action_ids), 
                                "tasks": np.array(task_id),
                                "video_id": np.array(video_id),
                                "video_len": np.array(length_video)}
                                for f in aug_features])
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, return_video_id = False):
        states = self.data[idx]["states"]
        actions = self.data[idx]["actions"]
        tasks = self.data[idx]["tasks"]
        # Convert the video_id to a tensor
        video_id = self.data[idx]["video_id"]
        video_len = self.data[idx]["video_len"]
        if return_video_id:
            return torch.as_tensor(states, dtype=torch.float32), torch.as_tensor(actions, dtype=torch.long), torch.as_tensor(tasks, dtype=torch.long), video_id, video_len
        elif self.PKG_labels:
            PKG_labels = self.data[idx]["PKG_labels"]
            LLM_labels = self.data[idx]["LLM_labels"]
            return torch.as_tensor(states, dtype=torch.float32), torch.as_tensor(actions, dtype=torch.long), torch.as_tensor(tasks, dtype=torch.long), torch.as_tensor(PKG_labels, dtype=torch.long), torch.as_tensor(LLM_labels, dtype=torch.long)
        else:
            return torch.as_tensor(states, dtype=torch.float32), torch.as_tensor(actions, dtype=torch.long), torch.as_tensor(tasks, dtype=torch.long)