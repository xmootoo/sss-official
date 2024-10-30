import os
from pathlib import Path
import torch
import neptune
from rich.console import Console
from sss.analysis.soz.heatmap import heatmap_analysis
from sss.config.config import load_config
from sss.utils.models import get_model
import yaml
from pydantic import BaseModel, Field

# Assume this is the structure of your YAML file
class Config(BaseModel):
    run_id: str = Field(default="SOZ-33", description="Neptune run ID")
    mode: str = Field(default="train", description="Options: 'train', 'val', 'test'")
    save: bool = Field(default=True, description="Save the heatmap")
    method_name: str = Field(default="SSS", description="Method name")
    patient_cluster: str = Field(default="umf", description="Patient cluster. Options: 'jh', 'umf', 'ummc', 'pt")
    avg_mode: str = Field(default="sma", description="Moving average mode. Options: 'sma', 'ema'")
    new_seed: bool = Field(default=False, description="Generate a new seed")
    window_stride: int = Field(default=1, description="Window stride")
    batch_size: int = Field(default=4096, description="Batch size")
    num_sampled_channels: int = Field(default=1, description="Number of sampled channels")
    pad_mode: str = Field(default="last", description="Padding mode. Options: 'last', 'none'")
    cmap: str = Field(default="viridis", description="Colormap")
    dpi: int = Field(default=300, description="DPI")
    calibrate: bool = Field(default=False, description="Calibrate the model")
    calibration_model: str = Field(default="isotonic_regression", description="Calibration model")
    relative: bool = Field(default=False, description="Relative heatmap")
    probability: bool = Field(default=True, description="Use probability. If False use logits")
    project_name: str = Field(default="neptune-project-name", description="Neptune project name")

# Load the arguments

def main():
    with open('plot.yaml', 'r') as file:
        yaml_args = yaml.safe_load(file)
    args = Config(**yaml_args)

    # Console and Neptune
    console = Console()
    api_token = os.getenv('NEPTUNE_API_TOKEN')

    # Find directories
    # model_folder = "../../../checkpoint/analysis"
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    model_folder = project_root / "checkpoint" / "analysis"
    model_folder.mkdir(parents=True, exist_ok=True) # Create the model folder if it does not exist

    # Load model weights
    model_path = os.path.join(model_folder, "local_weights.pth")
    run = neptune.init_run(
        project=args.project_name,
        api_token=api_token,
        with_id=args.run_id,
        mode="read-only"
    )
    run["model_checkpoints/PatchTSTBlind_sl"].download(destination=model_path)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    exp_args_path = project_root / "sss" / "jobs" / "exp" / "open_neuro" / "binary" / "sss" / "best" / "args.yaml"
    exp_args = load_config(exp_args_path)
    model = get_model(exp_args)

    # Adjust the state_dict keys if the model was trained with DDP
    if exp_args.ddp.ddp:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)

    # Run heatmap analysis
    heatmap_analysis(
        run=run,
        run_id=args.run_id,
        model=model,
        device=device,
        new_seed=args.new_seed,
        mode=args.mode,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        num_sampled_channels=args.num_sampled_channels,
        patient_cluster=args.patient_cluster,
        avg_mode=args.avg_mode,
        save=args.save,
        method_name=args.method_name,
        cmap=args.cmap,
        dpi=args.dpi,
        calibrate=args.calibrate,
        calibration_model=args.calibration_model,
        relative=args.relative,
        probability=args.probability
    )

if __name__ == "__main__":
    main()
