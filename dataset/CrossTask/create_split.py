import click
import json
import os
import numpy as np

@click.command()
@click.option('--input_file', type=str, required=True)
@click.option('--horizon', type=int, required=True)
@click.option('--features_dir', type=str, default="./crosstask_features/processed_data/")
@click.option('--features_root', type=str, default="./dataset/CrossTask/crosstask_features/processed_data/")
def main(input_file, horizon, features_dir, features_root):
    # Check if the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The input file {input_file} does not exist.")
    # Load the JSON data
    with open(input_file, 'r') as file:
        data = json.load(file)
        
    # Create the output file name        
    output_file = input_file.replace('.json', f'_{horizon}_133.json')
    
    taxonomy = "crosstask_taxonomy_133.json"
    with open(taxonomy, 'r') as file:
        taxonomy_data = json.load(file)
    
    new_mapping_actions = {}    
    # Create mapping of task names to task IDs
    mapping_task = {}
    for task in taxonomy_data:
        task_split = task.split("_")
        task_id = task_split[0]
        task_name = task_split[1]
        mapping_task[int(task_name)] = int(task_id)
        new_mapping_actions[int(task_name)] = {}
        i = 0
        for action in taxonomy_data[task]:
            new_mapping_actions[int(task_name)][i] = action
            i += 1
    
    # IDs to actions mapping
    ids_to_actions_133 = {}
    task_ids_to_name_133 = {}
    for scenario in taxonomy_data:
        scenario_split = scenario.split("_")
        scenario_id = scenario_split[0]
        scenario_name = scenario_split[2]
        scenario_id = int(scenario_id)
        if scenario_id not in task_ids_to_name_133:
            task_ids_to_name_133[scenario_id] = scenario_name
        for action in taxonomy_data[scenario]:
            ids_to_actions_133[int(action)] = taxonomy_data[scenario][action]
    
    dataset = []
    for sample in data:
        vid = sample["vid"]
        task = sample["task"]
        instruct_len = sample["length"]
        feature_name = os.path.join(features_dir, task + "_" + vid + ".npy")
        # Check if the feature file exists
        if not os.path.exists(feature_name):
            print(f"Feature file {feature_name} does not exist. Skipping this sample.")
            continue
        # Check if the length of the sample is greater or equal to horizon
        if instruct_len >= horizon:
            # Load the video annotation
            video_annot = np.load(feature_name, allow_pickle=True)
            steps_ids = video_annot["steps_ids"]
            task_id = mapping_task[video_annot["cls"]]
            starts = video_annot["start"]
            ends = video_annot["end"]
            for i in range(instruct_len - horizon + 1):
                legal_range = []
                for j in range(horizon):
                    step = i + j
                    idx = int(steps_ids[step]) - 1
                    idx = int(new_mapping_actions[video_annot["cls"]][idx])
                    start_time = starts[step]
                    end_time = ends[step]
                    # Truncate the time
                    start_time = int(round(start_time))
                    end_time = int(round(end_time))
                    legal_range.append([start_time, end_time, idx])
                sample_data = {
                    "id": {
                        "vid": vid,
                        "task": str(video_annot["cls"]),
                        "feature": os.path.join(features_root, task + "_" + vid + ".npy"),
                        "legal_range": legal_range,
                        "task_id": task_id,
                        "task_name": task_ids_to_name_133[task_id],
                        "actions": [ids_to_actions_133[data[-1]] for data in legal_range]
                    },
                    "instruction_len": instruct_len  # Assuming instruction_len is always 0 as per the original code
                }
                dataset.append(sample_data)
    
    # Save the dataset to the output file
    with open(output_file, 'w') as file:
        json.dump(dataset, file, indent=4)
        
    taxonomy_133_json = json.load(open("crosstask_taxonomy_133.json", "r"))
    taxonomy_105_json = json.load(open("crosstask_taxonomy_105.json", "r"))

    taxonomy_105_name_to_id = {}
    for task in taxonomy_105_json:
        for action_id, action in taxonomy_105_json[task].items():
            taxonomy_105_name_to_id[action] = action_id
            
    taxonomy_133 = {}
    for task in taxonomy_133_json:
        for action_id, action in taxonomy_133_json[task].items():
            taxonomy_133[action_id] = action
            
    for sample in dataset:
        video = sample["id"]
        task_id = video["task_id"]
                
        for legal_range in video["legal_range"]:
            action_id = legal_range[-1]
            action = taxonomy_133[str(action_id)]
            legal_range[-1] = int(taxonomy_105_name_to_id[action])
        
    # Save the file with reduced actions
    output_file_rd = output_file.replace('_133.json', '_105.json')
    with open(output_file_rd, 'w') as file:
        json.dump(dataset, file, indent=4)
        

if __name__ == '__main__':
    main()
