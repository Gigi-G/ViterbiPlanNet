import click
import json
import os
import numpy as np

@click.command()
@click.option('--input_file', type=str, required=True)
@click.option('--taxonomy_file', type=str, default='niv_taxonomy.json')
@click.option('--horizon', type=int, required=True)
@click.option('--features_dir', type=str, default="./processed_data/")
@click.option('--features_root', type=str, default="./dataset/NIV/processed_data/")
def main(input_file, taxonomy_file, horizon, features_dir, features_root):
    # Check if the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The input file {input_file} does not exist.")
    # Load the JSON data
    with open(input_file, 'r') as file:
        data = json.load(file)
    # Load the taxonomy file
    with open(taxonomy_file, 'r') as file:
        taxonomy = json.load(file)
        
    # IDs to actions mapping
    ids_to_actions = {}
    task_ids_to_name = {}
    for scenario in taxonomy:
        scenario_id, scenario_name = scenario.split('_')
        scenario_id = int(scenario_id)
        if scenario_id not in task_ids_to_name:
            task_ids_to_name[scenario_id] = scenario_name
        for action in taxonomy[scenario]:
            ids_to_actions[int(action)] = taxonomy[scenario][action]
    
    # Create the output file name        
    output_file = input_file.replace('.json', f'_{horizon}.json')
    
    dataset = []
    for sample in data:
        task_id = int(sample["task_id"])
        file_name = sample["feature"]
        feature_name = os.path.join(features_dir, file_name)
        # Check if the feature file exists
        if not os.path.exists(feature_name):
            print(f"Feature file {feature_name} does not exist. Skipping this sample.")
            continue
        video_annot = np.load(feature_name, allow_pickle=True)
        instruct_len = int(video_annot["num_steps"])
        steps_ids = video_annot["steps_ids"]
        starts = video_annot["steps_starts"]
        ends = video_annot["steps_ends"]
        if instruct_len >= horizon:
            for i in range(instruct_len - horizon + 1):
                # Create a sample data structure
                legal_range = []
                for j in range(horizon):
                    step = i + j
                    idx = int(steps_ids[step])
                    start_time = starts[step]
                    end_time = ends[step]
                    # Truncate the time
                    start_time = int(round(start_time))
                    end_time = int(round(end_time))
                    legal_range.append([start_time, end_time, idx])
                sample_data = {
                    "id": {
                        "feature": os.path.join(features_root, file_name),
                        "legal_range": legal_range,
                        "task_id": task_id,
                        "task_name": task_ids_to_name[task_id],
                        "actions": [ids_to_actions[data[-1]] for data in legal_range]
                    },
                    "instruction_len": instruct_len  # Assuming instruction_len is always 0 as per the original code
                }
                dataset.append(sample_data)
    
    # Save the dataset to the output file
    with open(output_file, 'w') as file:
        json.dump(dataset, file, indent=4)
        

if __name__ == '__main__':
    main()
