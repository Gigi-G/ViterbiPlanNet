import click
import json
import glob
import numpy as np
import random

def bootstrap(data, n_samples, n_iterations=1000):
    """Perform bootstrap resampling on the data and return confidence intervals."""
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
@click.option('--results', '-r', help='Path to the results file')
@click.option('--dataset', '-d', help='Name of the dataset')
@click.option('--horizon', '-h', help='Path to the horizon file')
@click.option('--type_of_model', '-t', default=None, help='Type of model used in the experiment (viterbi-DVL+VD, viterbi-DVL, None)')
@click.option('--no_subfolders', is_flag=True , help='If set, do not consider subfolders in the results directory', default=False)
def main(results: str, dataset: str, horizon: str, type_of_model: str, no_subfolders: bool):
    random.seed(42)
    if no_subfolders:
        folders = [results]
    else:
        folders = glob.glob(results + '/*')
    horizon_confidence_intervals = {
        "SR": [],
        "mAcc": [],
        "mIoU": []
    }

    json_to_consider = []
    for folder in folders:
        if dataset == 'crosstask':
            if dataset not in folder or 'crosstask_105' in folder:
                continue
        elif dataset not in folder:
            continue
        
        if no_subfolders:
            path_to_json = glob.glob(folder + f'/*.json')
        else:
            path_to_json = glob.glob(folder + f'/*metrics_viterbi.json')
            if len(path_to_json) == 0:
                path_to_json = glob.glob(folder + f'/*metrics.json')
        for file in path_to_json:
            if ("T" + str(horizon)) in file:
                json_to_consider.append(file)
    for file in json_to_consider:
        with open(file) as f:
            if type_of_model == 'viterbi-DVL':
                data = json.load(f)["viterbi-DVL"]
            elif type_of_model == 'viterbi-DVL+VD':
                data = json.load(f)["viterbi-DVL+VD"]
            else:
                data = json.load(f) # Default to viterbi if no specific type is provided
            horizon_confidence_intervals["SR"].append(data["SR"])
            try:
                horizon_confidence_intervals["mAcc"].append(data["mAcc"])
            except:
                horizon_confidence_intervals["mAcc"].append(data["acc"])
            horizon_confidence_intervals["mIoU"].append(data["mIoU"])
            
    # Convert to numpy arrays for easier manipulation
    sr_array = np.array(horizon_confidence_intervals["SR"])
    mAcc_array = np.array(horizon_confidence_intervals["mAcc"])
    mIoU_array = np.array(horizon_confidence_intervals["mIoU"])
    # Calculate mean for the horizon and store in final metrics
    sr_mean = np.mean(sr_array)
    mAcc_mean = np.mean(mAcc_array)
    mIoU_mean = np.mean(mIoU_array)
    # Number of samples in this folder (n)
    n = len(json_to_consider)
    sr_ci = bootstrap(horizon_confidence_intervals["SR"], n)
    sr_ci = sr_ci[1] - sr_ci[0]
    mAcc_ci = bootstrap(horizon_confidence_intervals["mAcc"], n)
    mAcc_ci = mAcc_ci[1] - mAcc_ci[0]
    mIoU_ci = bootstrap(horizon_confidence_intervals["mIoU"], n)
    mIoU_ci = mIoU_ci[1] - mIoU_ci[0]
    # Print the mean of the confidence intervals for each metric in percentages
    print(f"Mean 90% CI for SR: {sr_mean:.2f}% ± {sr_ci:.2f}%")
    print(f"Mean 90% CI for mAcc: {mAcc_mean:.2f}% ± {mAcc_ci:.2f}%")
    print(f"Mean 90% CI for mIoU: {mIoU_mean:.2f}% ± {mIoU_ci:.2f}%")
    
    # Save the results to a JSON file
    results_metrics = {
        "SR": {
            "mean": sr_mean,
            "ci": sr_ci
        },
        "mAcc": {
            "mean": mAcc_mean,
            "ci": mAcc_ci
        },
        "mIoU": {
            "mean": mIoU_mean,
            "ci": mIoU_ci
        }
    }
    with open(f"{results}/T{horizon}_results_{type_of_model}.json", 'w') as f:
        json.dump(results_metrics, f, indent=4)

if __name__ == '__main__':
    main()
