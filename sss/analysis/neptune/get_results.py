import neptune
from neptune import management
import os
import pandas as pd
import yaml
from typing import List, Dict, Any, Tuple, Union
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from collections import defaultdict
import numpy as np
from datetime import datetime

# Rich Console
console = Console()

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def init_neptune(config: Dict[str, Any]) -> neptune.Project:
    api_token = os.getenv('NEPTUNE_API_TOKEN')
    return neptune.init_project(
        project=config['neptune']['project'],
        api_token=api_token
    )

def get_query_columns(config: Dict[str, Any]) -> List[str]:
    columns = []
    columns.extend(config.get('filters', {}).keys())
    columns.extend(config.get('metrics', []))
    columns.extend(config.get('ablations', {}).get('params', []))
    columns.append('sys/id')  # Always include sys/id for run identification
    columns.append('parameters/exp/seed')  # Include seed for grouping
    columns.append('sys/creation_time')  # Include creation time
    return list(set(columns))  # Remove duplicates

def fetch_runs_table(project: neptune.Project, config: Dict[str, Any]) -> pd.DataFrame:
    columns = get_query_columns(config)
    deciding_metric = config.get('deciding_metric')
    sort_by = deciding_metric if deciding_metric else None

    runs_table_df = project.fetch_runs_table(
        columns=columns,
    ).to_pandas()

    return runs_table_df

def filter_dataframe(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    for key, value in filters.items():
        df = df[df[key] == value]
    return df

def format_combination(combination: Dict[str, Any]) -> str:
    return ", ".join(f"{key.split('/')[-1]}={value}" for key, value in combination.items())

def trash_objects(project: str, ids: Union[str, List[str]], workspace: str = None, api_token: str = None):
    try:
        management.trash_objects(project=project, ids=ids, workspace=workspace, api_token=api_token)
        if isinstance(ids, str):
            console.log(f"[green]Trashed object: {ids}[/green]")
        else:
            console.log(f"[green]Trashed {len(ids)} objects[/green]")
    except Exception as e:
        console.log(f"[red]Error trashing objects: {str(e)}[/red]")

def process_ablations(project: neptune.Project, config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    runs_table_df = fetch_runs_table(project, config)

    base_filters = config['filters']
    metrics = config['metrics']
    ablation_params = config['ablations']['params']
    seeds = config['ablations']['seeds']
    num_seeds = len(seeds)
    delete_duplicates = config.get('delete_duplicates', False)
    duplicate_ranking = config.get('duplicate_ranking', 'recent')
    deciding_metric = config.get('deciding_metric')
    ranking = config.get('ranking', 'max')

    filtered_df = filter_dataframe(runs_table_df, base_filters)

    console.log(f"[blue]Total runs fetched: {len(filtered_df)}[/blue]")
    console.log(f"[blue]Delete duplicates: {delete_duplicates}[/blue]")
    console.log(f"[blue]Duplicate ranking: {duplicate_ranking}[/blue]")
    console.log(f"[blue]General ranking: {ranking}[/blue]")

    for param in ablation_params:
        if param not in filtered_df.columns:
            console.log(f"[yellow]Warning: Parameter '{param}' not found in run data. Check your configuration.[/yellow]")

    # Group runs by their parameter combinations and seed
    grouped_runs = defaultdict(lambda: defaultdict(list))
    for _, run in filtered_df.iterrows():
        param_combination = tuple((param, run[param]) for param in ablation_params)
        seed = run.get('parameters/exp/seed', 'unknown')
        grouped_runs[param_combination][seed].append(run)

    console.log(f"[blue]Unique parameter combinations found: {len(grouped_runs)}[/blue]")

    results = {}
    skipped_combinations = []
    total_runs_to_trash = []

    for param_combination, seed_runs in grouped_runs.items():
        combination_dict = dict(param_combination)
        combination_str = format_combination(combination_dict)

        console.log(f"[cyan]Processing combination: {combination_str}[/cyan]")
        console.log(f"[cyan]Seeds found for this combination: {list(seed_runs.keys())}[/cyan]")

        valid_runs = []
        runs_to_trash = []

        for seed in seeds:
            if seed in seed_runs:
                seed_runs_list = seed_runs[seed]

                if duplicate_ranking == 'recent':
                    key_func = lambda x: x['sys/creation_time']
                    reverse = True
                elif duplicate_ranking in ['max', 'min']:
                    key_func = lambda x: x.get(deciding_metric, float('-inf') if duplicate_ranking == 'max' else float('inf'))
                    reverse = duplicate_ranking == 'max'
                else:
                    console.log(f"[red]Error: Invalid duplicate_ranking '{duplicate_ranking}'. Using 'recent' as default.[/red]")
                    key_func = lambda x: x['sys/creation_time']
                    reverse = True

                sorted_runs = sorted(seed_runs_list, key=key_func, reverse=reverse)

                valid_runs.append(sorted_runs[0])  # Keep the best run

                if delete_duplicates and len(sorted_runs) > 1:
                    runs_to_trash.extend([run['sys/id'] for run in sorted_runs[1:]])

        if len(valid_runs) < num_seeds:
            console.log(f"[yellow]Warning: Not enough runs for combination {combination_str}. Found {len(valid_runs)}, expected {num_seeds}. Skipping.[/yellow]")
            skipped_combinations.append(combination_str)
            continue

        # Add excess runs to trash if there are more valid runs than seeds
        if len(valid_runs) > num_seeds:
            console.log(f"[yellow]Warning: Too many runs for combination {combination_str}. Found {len(valid_runs)}, expected {num_seeds}. Taking the top {num_seeds} runs.[/yellow]")
            runs_to_trash.extend([run['sys/id'] for run in valid_runs[num_seeds:]])
            valid_runs = valid_runs[:num_seeds]

        total_runs_to_trash.extend(runs_to_trash)

        runs_df = pd.DataFrame(valid_runs)

        # Debug logging for NaN values
        for metric in metrics:
            if metric not in runs_df.columns:
                console.log(f"[red]Error: Metric '{metric}' not found in run data for combination {combination_str}[/red]")
            elif runs_df[metric].isnull().any():
                nan_count = runs_df[metric].isnull().sum()
                console.log(f"[yellow]Warning: {nan_count} NaN values found for metric '{metric}' in combination {combination_str}[/yellow]")

                # Additional debugging for NaN values
                nan_runs = runs_df[runs_df[metric].isnull()]
                for _, nan_run in nan_runs.iterrows():
                    console.log(f"[yellow]NaN run details: seed={nan_run.get('parameters/exp/seed', 'unknown')}, status={nan_run.get('sys/state', 'unknown')}, creation_time={nan_run.get('sys/creation_time', 'unknown')}[/yellow]")

        results[combination_str] = calculate_metrics(runs_df, metrics)

    # Trash marked runs
    if delete_duplicates and total_runs_to_trash:
        trash_objects(project=config['neptune']['project'], ids=total_runs_to_trash)

    console.log(f"[blue]Total valid combinations after processing: {len(results)}[/blue]")
    console.log(f"[blue]Total runs marked for deletion: {len(total_runs_to_trash)}[/blue]")

    if skipped_combinations:
        console.log(f"[yellow]Skipped {len(skipped_combinations)} combinations due to insufficient runs:[/yellow]")
        for combination in skipped_combinations:
            console.log(f"[yellow]  - {combination}[/yellow]")

    return results

def calculate_metrics(df: pd.DataFrame, metrics: List[str]) -> Dict[str, float]:
    results = {}
    for metric in metrics:
        if metric not in df.columns:
            results[metric] = np.nan
            console.log(f"[red]Error: Metric '{metric}' not found in DataFrame[/red]")
        else:
            metric_values = df[metric].dropna()
            if len(metric_values) == 0:
                results[metric] = np.nan
                console.log(f"[yellow]Warning: All values for metric '{metric}' are NaN[/yellow]")
            else:
                results[metric] = metric_values.mean()

    # Add creation time information
    results['earliest_run'] = df['sys/creation_time'].min()
    results['latest_run'] = df['sys/creation_time'].max()

    return results

def save_results_to_csv(results: Dict[str, Dict[str, float]], output_path: str, deciding_metric: str, ranking: str):
    # Create a DataFrame with all results
    all_results_df = pd.DataFrame.from_dict(results, orient='index')
    all_results_df.reset_index(inplace=True)
    all_results_df.rename(columns={'index': 'combination'}, inplace=True)

    # Convert datetime columns to string for better CSV compatibility
    all_results_df['earliest_run'] = all_results_df['earliest_run'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    all_results_df['latest_run'] = all_results_df['latest_run'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

    # Sort the DataFrame based on the deciding metric
    all_results_df = all_results_df.sort_values(by=deciding_metric, ascending=(ranking == 'min'))

    # Save the sorted DataFrame to CSV
    all_results_df.to_csv(output_path, index=False)
    console.print(f"[green]Results saved to {output_path}[/green]")

def find_top_5_experiments(results: Dict[str, Dict[str, float]], deciding_metric: str, ranking: str) -> List[Tuple[str, Dict[str, float]]]:
    sorted_results = sorted(results.items(), key=lambda x: x[1][deciding_metric], reverse=(ranking == 'max'))
    return sorted_results[:5]

def print_top_5_experiments(top_5: List[Tuple[str, Dict[str, float]]], deciding_metric: str):
    for i, (combination, metrics) in enumerate(top_5, 1):
        panel_content = "\n".join([
            f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}"
            for metric, value in metrics.items()
        ])
        panel = Panel(
            panel_content,
            title=f"Top {i}: {combination}",
            expand=False
        )
        console.print(panel)

def main(ablation_name: str):
    ablation_config_path = Path('ablations') / f"{ablation_name}.yaml"

    if not ablation_config_path.exists():
        console.log(f"[red]Ablation configuration file not found: {ablation_config_path}[/red]")
        return

    config = load_config(str(ablation_config_path))

    console.log(f"[blue]Configuration loaded: {len(config['ablations']['params'])} parameters, {len(config['ablations']['seeds'])} seeds[/blue]")

    project = init_neptune(config)

    results = process_ablations(project, config)

    deciding_metric = config.get('deciding_metric')
    ranking = config.get('ranking', 'max')
    if deciding_metric:
        top_5 = find_top_5_experiments(results, deciding_metric, ranking)
        if top_5:
            console.log(f"\n[bold blue]Top 5 experiments based on {deciding_metric} ({ranking}):[/bold blue]")
            print_top_5_experiments(top_5, deciding_metric)
        else:
            console.log("[yellow]No valid results found. All experiments may have failed or produced nan values.[/yellow]")

        output_path = f"./results/{ablation_name}.csv"
        save_results_to_csv(results, output_path, deciding_metric, ranking)
    else:
        console.log("\n[yellow]No deciding metric specified in the configuration. Cannot determine the top experiments.[/yellow]")

        output_path = f"./results/{ablation_name}.csv"
        save_results_to_csv(results, output_path, next(iter(results[next(iter(results))])), ranking)

    # Add a summary of nan values
    nan_count = sum(1 for result in results.values() if np.isnan(result.get(deciding_metric, np.nan)))
    total_count = len(results)
    console.log(f"\n[blue]Summary: {nan_count} out of {total_count} experiments produced nan values for the deciding metric.[/blue]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Neptune analysis with custom ablations.")
    parser.add_argument("ablation", help="Name of the ablation configuration (without .yaml extension)")
    args = parser.parse_args()

    main(args.ablation)
