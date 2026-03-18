import click
import json
import glob
import numpy as np
import random
import os

def bootstrap(data, n_samples, n_iterations=1000):
    """Perform bootstrap resampling on the data and return confidence intervals."""
    if not data:
        return 0, 0
    means = []
    for _ in range(n_iterations):
        # Resample with replacement
        bootstrap_sample = [random.choice(data) for _ in range(n_samples)]
        # Compute the mean for the bootstrap sample
        means.append(np.mean(bootstrap_sample))
    
    # Get the 5th and 95th percentiles for the 90% confidence interval
    lower_bound = np.percentile(means, 5)
    upper_bound = np.percentile(means, 95)
    return lower_bound, upper_bound

@click.command()
@click.option('--results', '-r', help='Path to the results directory (e.g. step_models/MLP_based/checkpoints)')
@click.option('--dataset', '-d', help='Name of the dataset (e.g. coin, crosstask, niv, egoper)')
@click.option('--horizon', '-h', help='Path to the horizon file/T value (e.g. 3, 4, 5, 6)')
@click.option('--no_subfolders', is_flag=True , help='If set, do not consider subfolders in the results directory', default=False)
def main(results: str, dataset: str, horizon: str, no_subfolders: bool):
    random.seed(42)
    if no_subfolders:
        folders = [results]
    else:
        folders = glob.glob(os.path.join(results, '*'))
        
    metrics_data = {
        "state_acc": [],
        "first_action_acc": [],
        "last_action_acc": [],
        "task_acc": []
    }

    json_to_consider = []
    for folder in folders:
        if dataset == 'crosstask':
            if dataset not in folder or 'crosstask_105' in folder:
                continue
        elif dataset not in folder:
            continue
        
        if no_subfolders:
            path_to_json = glob.glob(os.path.join(folder, '*.json'))
        else:
            path_to_json = glob.glob(os.path.join(folder, '*eval_results.json'))

        for file in path_to_json:
            if f"T{horizon}" in os.path.basename(file):
                json_to_consider.append(file)
                
    print(f"Found {len(json_to_consider)} files matching dataset={dataset} and T={horizon}")
    
    for file in json_to_consider:
        with open(file) as f:
            data = json.load(f)
            for k in metrics_data.keys():
                if k in data:
                    metrics_data[k].append(data[k])
            
    n = len(json_to_consider)
    if n == 0:
        print("No valid json files found to process. Exiting.")
        return

    results_metrics = {}
    for metric_name in metrics_data.keys():
        if len(metrics_data[metric_name]) == 0:
            continue
            
        data_array = np.array(metrics_data[metric_name])
        mean_val = np.mean(data_array)
        
        ci_bounds = bootstrap(metrics_data[metric_name], n)
        ci_val = ci_bounds[1] - ci_bounds[0]
        
        print(f"Mean 90% CI for {metric_name}: {mean_val:.2f}% ± {ci_val:.2f}%")
        
        results_metrics[metric_name] = {
            "mean": mean_val,
            "ci": ci_val
        }
        
    out_file = os.path.join(results, f"T{horizon}_{dataset}_eval_metrics.json")
    with open(out_file, 'w') as f:
        json.dump(results_metrics, f, indent=4)
        
    print(f"Results saved to {out_file}")

if __name__ == '__main__':
    main()