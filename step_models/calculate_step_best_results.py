import click
import json
import glob
import os
import re


METRICS = [
    "state_acc",
    "first_action_acc",
    "last_action_acc",
    "task_acc"
]


def extract_seed_from_path(path: str, dataset: str):
    """Extract seed from folder names like coin_7, niv_42, crosstask_123."""
    normalized_path = path.replace('\\\\', '/')
    match = re.search(rf'/(?:{re.escape(dataset)})_(\d+)(?:/|$)', normalized_path)
    if match:
        return int(match.group(1))

    match = re.search(r'_(\d+)(?:/|$)', normalized_path)
    if match:
        return int(match.group(1))
    return None


@click.command()
@click.option('--results', '-r', required=True, help='Path to the results directory (e.g. step_models/MLP_based/checkpoints)')
@click.option('--dataset', '-d', required=True, help='Name of the dataset (e.g. coin, crosstask, niv, egoper)')
@click.option('--horizon', '-h', required=True, help='Horizon value/T (e.g. 3, 4, 5, 6)')
@click.option('--rank_metric', '-m', default='state_acc', show_default=True,
              help='Metric used to select the best run')
@click.option('--no_subfolders', is_flag=True, default=False,
              help='If set, do not consider subfolders in the results directory')
def main(results: str, dataset: str, horizon: str, rank_metric: str, no_subfolders: bool):
    if rank_metric not in METRICS:
        raise click.BadParameter(
            f"rank_metric must be one of: {', '.join(METRICS)}",
            param_hint='rank_metric'
        )

    if no_subfolders:
        folders = [results]
    else:
        folders = glob.glob(os.path.join(results, '*'))

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

    if len(json_to_consider) == 0:
        print("No valid json files found to process. Exiting.")
        return

    run_results = []
    for file in sorted(json_to_consider):
        with open(file) as f:
            data = json.load(f)

        metrics = {metric: data.get(metric) for metric in METRICS}
        run_results.append({
            "file": file,
            "seed": extract_seed_from_path(file, dataset),
            "metrics": metrics
        })

    valid_runs = [run for run in run_results if run["metrics"].get(rank_metric) is not None]
    if len(valid_runs) == 0:
        print(f"No runs contain rank metric '{rank_metric}'. Exiting.")
        return

    best_run = max(valid_runs, key=lambda run: run["metrics"][rank_metric])

    print("Best run summary:")
    print(f"- rank metric: {rank_metric}")
    print(f"- best value: {best_run['metrics'][rank_metric]:.2f}%")
    print(f"- seed: {best_run['seed']}")
    print(f"- file: {best_run['file']}")
    for metric_name, metric_value in best_run["metrics"].items():
        if metric_value is not None:
            print(f"- {metric_name}: {metric_value:.2f}%")

    out_data = {
        "rank_metric": rank_metric,
        "best_run": best_run
    }

    out_file = os.path.join(results, f"T{horizon}_{dataset}_best_eval_metrics.json")
    with open(out_file, 'w') as f:
        json.dump(out_data, f, indent=4)

    print(f"Results saved to {out_file}")


if __name__ == '__main__':
    main()
