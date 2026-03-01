import click
import json
import os

@click.command()
@click.option('--input_file', type=str, required=True)
@click.option('--taxonomy_file', type=str, default='coin_taxonomy.json')
@click.option('--horizon', type=int, required=True)
@click.option('--features_dir', type=str, default="./full_npy/")
@click.option('--features_root', type=str, default="./dataset/COIN/full_npy/")
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
        vid = list(sample.keys())[0]
        video = sample[vid]
        feature_name = os.path.join(features_dir, str(video["class"]) + "_" + str(video["recipe_type"]) + "_" + vid + ".npy")
        # Check if the feature file exists
        if not os.path.exists(feature_name):
            print(f"Feature file {feature_name} does not exist. Skipping this sample.")
            continue
        task_id = video["recipe_type"]
        # Check if the length of the sample is greater or equal to horizon
        if len(video["annotation"]) >= horizon:
            for i in range(len(video["annotation"]) - horizon + 1):
                legal_range = []
                for j in range(horizon):
                    step = i + j
                    step_annotation = video["annotation"][step]
                    idx = int(step_annotation["id"])
                    start_time = step_annotation["segment"][0]
                    end_time = step_annotation["segment"][1]
                    # Truncate the time
                    start_time = round(start_time)
                    end_time = round(end_time)
                    legal_range.append([start_time, end_time, idx-1])
                sample_data = {
                    "id": {
                        "vid": vid,
                        "feature": os.path.join(features_root, str(video["class"]) + "_" + str(video["recipe_type"]) + "_" + vid + ".npy"),
                        "legal_range": legal_range,
                        "task_id": task_id,
                        "task_name": task_ids_to_name[task_id],
                        "actions": [ids_to_actions[data[-1]] for data in legal_range]
                    },
                    "instruction_len": len(video["annotation"])  # Assuming instruction_len is the length of the annotation
                }
                dataset.append(sample_data)
        
    # Save the dataset to the output file
    with open(output_file, 'w') as file:
        json.dump(dataset, file, indent=4)

if __name__ == '__main__':
    main()
