import warnings
warnings.filterwarnings("ignore")

import os
import yaml
import time
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Set, Tuple, List, Optional, Dict, Union

import random
import string

def generate_random_id(length=10):
    # Seed the random number generator with current time and os-specific random data
    random.seed(int(time.time() * 1000) ^ os.getpid())

    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

class Experiment(BaseModel):
    model_id: str = Field(default="PatchTSTBlind", description="Model ID. Options: 'PatchTSTOG', 'PatchTSTBlind', 'JEPA', 'DualJEPA'")
    backbone_id: str = Field(default="LSTM", description="Backbone for the RecurrentModel class. Options: 'LSTM', 'RNN', 'GRU', 'Mamba'.")
    seed_list: List[int] = Field(default=[2024], description="List of random seeds to run a single experiment on.")
    seed: int = Field(default=2024, description="Random seed")
    learning_type: str = Field(default="sl", description="Type of learning: 'sl', 'ssl'")
    id: str = Field(default_factory=generate_random_id, description="Experiment ID, randomly generated 10-character string")
    neptune: bool = Field(default=False, description="Whether to use Neptune for logging. If False, offline logging (JSON) will be used.")
    api_token: str = Field(default=os.environ.get('NEPTUNE_API_TOKEN', ''), description="Neptune API token")
    project_name: str = Field(default="neptune-project-name", description="Neptune project name")
    time: str = Field(default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), description="Neptune run ID")
    early_stopping: bool = Field(default=True, description="Whether to use early stopping")
    acc: bool = Field(default=False, description="Evaluate accuracy or not for classification tasks")
    ch_acc: bool = Field(default=False, description="Evaluate whole channel accuracy or not for classification tasks")
    thresh: float = Field(default=0.5, description="The threshold for binary classification")
    mae: bool = Field(default=False, description="Evaluate mean absolute error or not for forecasting")
    other_metrics: bool = Field(default=False, description="Whether to use other metrics for classification evaluation (e.g., F1-score")
    batchwise_metrics: bool = Field(default=False, description="Used for Stochastic Sparse Sampling (SSS). Computes the metrics scores using only windows in a given batch -> channel predictions, rather than aggregating all windows across all batches -> channel predictions.")
    best_model_metric: str = Field(default="loss", description="Metric to use for model saving and early stopping. Options: 'loss', 'acc', 'ch_acc'")
    tuning_metric: str = Field(default="loss", description="Metric to use for hyperparameter tuning. Options: 'ch_acc', 'ch_f1', 'ch_auroc', etc. See get_logger_mapping() under utils/classification.py for a full list of keys.")
    mps: bool = Field(default=False, description="Whether to use MPS for Apple silicon hardware acceleration")
    rank_verbose: bool = Field(default=False, description="Verbose logging in the console for each rank, e.g., prints the loss for each rank before reduction.")
    sklearn: bool = Field(default=False, description="Whether to use scikit-learn for model training and evaluation")
    sklearn_n_jobs: int = Field(default=-1, description="Number of CPU cores to use (-1 for all cores)")
    sklearn_verbose: int = Field(default=1, description="Verbosity level")
    grid_search: bool = Field(default=False, description="Whether to use grid search for hyperparameter tuning")
    task: str = Field(default="classification", description="Task type. Options: 'forecasting', 'classification'")
    gpu_id: int = Field(default=0, description="GPU ID to use for for single device training")
    u_weight: bool = Field(default=False, description="Whether to use uncertainty/confidence based weighting in the ChannelLoss for SSS")
    ablation_id: int = Field(default=1, description="Ablation ID for the base experiment")
    calibrate: bool = Field(default=False, description="Whether to calibrate the model for its window or channel probabilities")
    calibration_model: str = Field(default="isotonic_regression", description="Calibration model. Options: 'isotonic_regression', 'platt_scaling', 'ensemble', 'beta_calibration'.")
    calibration_type: str = Field(default="none", description="Calibration type. Options: 'none', 'window', 'channel'.")
    va_inductive: bool = Field(default=True, description=" True to run the Inductive (IVAP) or False for Cross (CVAP) Venn-ABERS calibtration")
    va_splits: int = Field(default=5, description="Number of splits for Cross (CVAP) Venn-ABERS calibration.")

class Data(BaseModel):
    dataset: str = Field(default="electricity", description="Name of the dataset. Options: 'electricity', 'traffic', 'weather', 'exchange_rate', 'illness', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'open_neuro', 'LongTerm17'.")
    dtype: str = Field(default="float32", description="Type of data. Options: 'float32', 'float64'")
    seq_len: int = Field(default=512, description="Sequence length of the input.")
    window_stride: int = Field(default=1, description="Window stride for generating windows.")
    pred_len: int = Field(default=96, description="Prediction length of the forecast window.")
    num_classes: int = Field(default=2, description="Number of classes for classification tasks.")
    num_channels: int = Field(default=321, description="Number of time series channels")
    drop_last: bool = Field(default=False, description="Whether to drop the last batch.")
    scale: bool = Field(default=True, description="Normalize data along each channel.")
    balance: bool = Field(default=True, description="Balance classes within dataset for classification.")
    train_split: float = Field(default=0.6, description="Portion of data to use for training")
    val_split: float = Field(default=0.2, description="Portion of data used for validation")
    num_workers: int = Field(default=4, description="Number of workers for the dataloader")
    pin_memory: bool = Field(default=True, description="Whether to pin memory for the dataloader")
    prefetch_factor: int = Field(default=2, description="Prefetch factor for the dataloader")
    shuffle_test: bool = Field(default=False, description="Whether to shuffle the test set")
    patching: bool = Field(default=False, description="Whether to use patching for the dataset (LSTM only)")
    patch_dim: int = Field(default=16, description="Patch dimension or patch length.")
    patch_stride: int = Field(default=8, description="Patch stride for generating patches.}")
    univariate: bool = Field(default=False, description="Whether to process a multivariate time series as univariate data (mixed together but separated by channel)")
    seq_load: bool = Field(default=False, description="Whether to use sequential dataloading. Loads train datasets first, and then test set at test time.")
    rank_seq_load: bool = Field(default=False, description="Whether to use sequential dataloading for each rank for large datasets, only for DDP.")
    pad_to_max: bool = Field(default=False, description="Whether to pad sequences to the maximum length in the dataset.")
    dataset_only: bool = Field(default=False, description="Whether to only load the dataset without training the model (for sickit-learn).")
    tslearn: bool = Field(default=False, description="Whether to use tslearn time series data for processing multiple time series.")
    numpy_data: bool = Field(default=False, description="Whether to use numpy data in dataloading (will not be converted to torch tensors).")
    rocket_transform: bool = Field(default=False, description="Whether to use the ROCKET transform for time series data.")
    full_channels: bool = Field(default=False, description="Whether to use the full channels for the OpenNeuro dataset or not.")
    resizing_mode: str = Field(default="None", description="Mode for resizing the time series data. Options: 'pad_trunc' or 'resizing' for linear interpolation.")
    median_seq_len: bool = Field(default=False, description="Whether to use the median sequence length from the training set as the context/window size.")
    target_channel: int = Field(default=-1, description="Target channel for univariate modelling.")
    time_indices: bool = Field(default=False, description="Whether to use relative time indices for sorting windows within the channel.")

class SparseContext(BaseModel):
    sparse_context: bool = Field(default=False, description="Whether to use sparse context encoding for the model.")
    d_model: int = Field(default=64, description="Dimension of the model.")
    num_enc_layers: int = Field(default=1, description="Number of encoder layers.")
    bidirectional: bool = Field(default=False, description="Whether to use bidirectional (LSTM, GRU, RNN).")
    backbone_id: str = Field(default="GRU", description="Backbone ID for the recurrent model.")
    dropout: float = Field(default=0.1, description="Dropout rate.")
    norm_mode: str = Field(default="layer", description="Normalization mode. Options: 'layer', 'batch', 'none'.")
    last_state: bool = Field(default=False, description="Whether to use the last state of the RNN as the context.")
    avg_state: bool = Field(default=False, description="Whether to use the average state of the RNN as the context.")
    patching: bool = Field(default=False, description="Whether to use patching for the dataset (LSTM only)")
    patch_dim: int = Field(default=16, description="Patch dimension or patch length.")
    patch_stride: int = Field(default=8, description="Patch stride for generating patches.")
    combine: str = Field(default="concat", description="Combination mode for the context and the input. Options: 'concat', 'add'.")
    final_norm: str = Field(default="layer", description="Normalization mode before the final layer. Options: 'layer', 'batch1d'.")

class SSL(BaseModel):
    optimizer: str = Field(default="adamw", description="Optimizer for SSL. Options: 'adamw' or 'adam'")
    scheduler: str = Field(default="cosine_warmup", description="Scheduler for SSL. Options: 'cosine_warmup', 'cosine', 'onecycle', ")
    criterion: str = Field(default="SmoothL1", description="Criterion for SSL. Options: 'SmoothL1', 'MSE', or 'CrossEntropy'")
    epochs: int = Field(default=200, description="Number of epochs for SSL.")
    batch_size: int = Field(default=128, description="Batch size for SSL.")
    lr: float = Field(default=1e-6, description="Learning rate for SSL.")
    revin: bool = Field(default=False, description="Use RevIN during SSL pretraining.")
    revout: bool = Field(default=False, description="Use RevOUT during SSL pretraining.")
    revin_affine: bool = Field(default=False, description="Use RevIN with affine parameters during SSL pretraining.")
    weight_decay: float = Field(default=1e-5, description="Weight decay for the optimizer")
    dataset_class: str = Field(default="forecasting", description="Task type: 'forecasting', 'forecasting_og', 'classification', 'JEPA', 'DualJEPA'")
    early_stopping: bool = Field(default=False, description="Early stopping for SSL.")

class Downstream(BaseModel):
    epochs: int = Field(default=5, description="Number of epochs for the model")
    batch_size: int = Field(default=32, description="Batch size for the model")
    model_id: str = Field(default="PatchTSTBlind", description="The model ID for the pretrained backbone used")
    ignore_pretrain: bool = Field(default=False, description="Ignore pretraining for only using running downstream training")
    eval: str = Field(default="finetune", description="Evaluation type: 'linear_probe' or 'finetune'")
    optimizer: str = Field(default="adam", description="Optimizer for task: 'adam' or 'adamw'")
    criterion: str = Field(default="MSE", description="Criterion for task: 'MSE' or 'SmoothL1'")
    scheduler: str = Field(default=None, description="Scheduler for task: 'cosine'")
    lr_encoder: float = Field(default=1e-4, description="Learning rate for encoder in SSL finetuning")
    lr_head: float = Field(default=1e-4, description="Learning rate for linear head in SSL finetuning")
    final_lr: float = Field(default=1e-6, description="Final learning rate for task")
    last_epoch: int = Field(default=-1, description="Last epoch for the scheduler")
    model_id: str = Field(default="PatchTSTBlind", description="Model ID for the downstream task")
    weight_decay: float = Field(default=1e-6, description="Weight decay for the optimizer")
    dataset_class: str = Field(default="forecasting", description="Task type: 'forecasting', 'forecasting_og', 'classification', 'JEPA', 'DualJEPA'")
    revin: bool = Field(default=True, description="Whether to use instance normalization with RevIN.")
    revout: bool = Field(default=True, description="Whether to use add mean and std back after forecast.")
    revin_affine: bool = Field(default=True, description="Whether to use learnable affine parameters for RevIN.")
    early_stopping: bool = Field(default=False, description="Early stopping for downstream.")

class SL(BaseModel):
    optimizer: str = Field(default="adam", description="Optimizer for supervised learning: 'adam' or 'adamw'")
    criterion: str = Field(default="MSE", description="Criterion for supervised learning: 'MSE', 'SmoothL1', 'CrossEntropy', 'BCE', 'ChannelLossBCE', 'ChannelLossCE'")
    num_enc_layers: int = Field(default=3, description="Number of encoder layers in the model")
    d_model: int = Field(default=128, description="Dimension of the model")
    d_ff: int = Field(default=256, description="Dimension of the feedforward network model")
    num_heads: int = Field(default=16, description="Number of heads in each MultiheadAttention block")
    dropout: float = Field(default=0.05, description="Dropout for some of the linears layers in PatchTSTOG")
    attn_dropout: float = Field(default=0.2, description="Dropout value for attention")
    ff_dropout: float = Field(default=0.2, description="Dropout value for feed forward")
    pred_dropout: float = Field(default=0.1, description="Dropout value for prediction")
    batch_first: bool = Field(default=True, description="Whether the first dimension is batch")
    norm_mode: str = Field(default="batch1d", description="Normalization mode: 'batch1d', 'batch2d', or 'layer'")
    batch_size: int = Field(default=64, description="Batch size")
    revin: bool = Field(default=True, description="Whether to use instance normalization with RevIN.")
    revout: bool = Field(default=True, description="Whether to use add mean and std back after forecast.")
    revin_affine: bool = Field(default=True, description="Whether to use learnable affine parameters for RevIN.")
    eps_revin: float = Field(default=1e-5, description="Epsilon value for reversible input")
    lr: float = Field(default=1e-4, description="Learning rate")
    epochs: int = Field(default=100, description="Number of epochs to train")
    scheduler: str = Field(default=None, description="Scheduler to use for learning rate annealing")
    weight_decay: float = Field(default=1e-6, description="Weight decay for the optimizer")
    dataset_class: str = Field(default="forecasting", description="Task type: 'forecasting', 'forecasting_og', 'classification', 'JEPA', 'DualJEPA'")
    early_stopping: bool = Field(default=False, description="Early stopping for supervised learning.")
    head_type: str = Field(default="linear", description="Head type for supervised learning: 'linear' or 'mlp'")
    num_kernels: int = Field(default=32, description="Number of convolutional kernels.")
    return_head: bool = Field(default=False, description="Whether to return the head of the model.")
    bidirectional: bool = Field(default=False, description="Whether to use bidirectional LSTM (typically for classification).")
    last_state: bool = Field(default=True, description="Whether to use the last state hidden of the LSTM (typically for classification).")
    avg_state: bool = Field(default=False, description="Whether to use the average the hidden states of the LSTM.")
    max_dilation: int = Field(default=32, description="Value is proportional to the max dilation size for the ROCKET model's kernels.")
    num_neighbours: int = Field(default=5, description="Number of neighbors to use")
    knn_weights: str = Field(default='uniform', description="Weight function used in prediction. Options: 'uniform', 'distance'")
    knn_metric: str = Field(default="dtw", description="Distance metric to use. Options: 'dtw', 'dtw_sakoe_chiba', 'softdtw'")
    knn_metric_params: Dict = Field(default={}, description="Additional keyword arguments for the metric function")

class MLPMixer(BaseModel):
    tok_mixer_dim: int = Field(default=128, description="Token mixer hidden dimension")
    cha_mixer_dim: int = Field(default=128, description="Channel mixer hidden dimension")
    pos_enc_type: str = Field(default="1d_sincos", description="Position encoding type, Options: 'learnable', '1d_sincos' or 'None'")
    dropout: float = Field(default=0.0, description="Dropout value for Token-Mixer MLP and Channel-Mixer MLP")

class DDP(BaseModel):
    ddp: bool = Field(default=False, description="Running distributive process on multiple nodes")
    master_addr: str = Field(default="localhost", description="Address of the master node")
    master_port: str = Field(default="12355", description="Port of the master node")
    shuffle: bool = Field(default=True, description="Shuffle the data for distributed training")
    slurm: bool = Field(default=False, description="Running on SLURM cluster")
    find_unused_parameters: bool = Field(default=False, description="Find unused parameters for distributed training")

class EarlyStopping(BaseModel):
    patience: int = Field(default=10, description="Patience for early stopping.")
    verbose: bool = Field(default=True, description="Verbose print for early stopping class.")
    delta: float = Field(default=0.0, description="Delta additive to the best scores for early stopping.")

class Scheduler(BaseModel):
    warmup_steps: int = Field(default=15, description="Number of warmup epochs for the scheduler for cosine_warmup")
    start_lr: float = Field(default=1e-4, description="Starting learning rate for the scheduler for warmup, for cosine_warmup")
    ref_lr: float = Field(default=1e-3, description="End learning rate for the scheduler after warmp, for cosine_warmup")
    final_lr: float = Field(default=1e-6, description="Final learning rate by the end of the schedule (starting from ref_lr) for cosine_warmup")
    T_max: int = Field(default=100, description="Maximum number of epochs for the scheduler for CosineAnnealingLR or cosine_warmup")
    last_epoch: int = Field(default=-1, description="Last epoch for the scheduler or CosineAnnealingLR or cosine_warmup")
    eta_min: float = Field(default=1e-6, description="Minimum learning rate for CosineAnnealingLR")
    pct_start: float = Field(default=0.3, description="Percentage of the cycle (in number of steps) spent increasing the learning rate for OneCycleLR")
    lradj: str = Field(default="type3", description="Learning rate adjustment type (ontop of scheduling). Options: 'type3', 'TST'")

class OpenNeuro(BaseModel):
    patient_cluster: str = Field(default="jh", description="Patient cluster for the OpenNeuro dataset. Options: 'jh', 'pt', 'umf', 'ummc'")
    kernel_size: int = Field(default=150, description="Kernel size for OpenNeuro dataset.")
    kernel_stride: int = Field(default=75, description="Kernel stride for OpenNeuro dataset.")
    pool_type: str = Field(default="avg", description="Pooling type for OpenNeuro dataset.")
    alpha: float = Field(default=1.0, description="Weight for the Normal Loss in the OpenNeuro dataset.")
    beta: float = Field(default=1.0, description="Weight for the Channel Loss in the OpenNeuro dataset.")
    gamma: float = Field(default=0.0, description="Weight for the variance of the channcel predictions in ChannelLOss.")
    chi: float = Field(default=0.0, description="Weight for the entropy loss of the uncertainty/confidence coefficients to encourage diversity (not focused on a single window).")
    ch_loss: bool = Field(default=False, description="Use channel-wise loss or not for classification tasks")
    ch_loss_type: str = Field(default="BCE", description="Channel loss type for the OpenNeuro dataset. Options: 'BCE', 'CE'")
    ch_var_loss: bool = Field(default=False, description="Whether to include the variance of the channel predictions in the loss for ChannelLoss.")
    task: str = Field(default="binary", description="Task type for the OpenNeuro dataset. Options: 'binary', 'multi'. Where 'multi' includes surgical outcomes. 'multi' is only valid for 'pt' and 'ummc' clusters.")
    ch_aggr: bool = Field(default=False, description="Whether to aggregate the channel latent representations before prediction.")
    ch_reduction: str = Field(default="mean", description="Channel reduction type for the OpenNeuro dataset. Options: 'mean', 'max', 'sum'")
    all_clusters: bool = Field(default=False, description="Whether to use all clusters for training + evaluation or not.")
    loocv: bool = Field(default=False, description="Leave-one-out cross-validation for the OpenNeuro dataset on the patient clusters.")
    train_clusters: List[str] = Field(default=["jh", "umf", "pt"], description="The patient clusters in Leave-one-out cross-validation used for training. Options, any subset of {'jh', 'pt', 'umf', 'ummc'}")
    test_clusters: List[str] = Field(default=["umf"], description="The patient clusters in Leave-one-out cross-validation used for testing. Options, any subset of {'jh', 'pt', 'umf', 'ummc'}")

class QWA(BaseModel):
    qwa: bool = Field(default=False, description="Whether to use the Quartile Weighted Aggregation model.")
    ch_loss_refined: bool = Field(default=True, description="Whether to use ChannelLoss for the refined probabilities.")
    ch_loss_coarse: bool = Field(default=True, description="Whether to use ChannelLoss for the coarse probabilities.")
    window_loss_refined: bool = Field(default=False, description="Whether to use window loss for the refined probabilities.")
    window_loss_coarse: bool = Field(default=False, description="Whether to use window loss for the coarse probabilities.")
    skew_loss: bool = Field(default=False, description="Whether to use skew loss for the QWA coefficients.")
    delta: float = Field(default=0.1, description="The target skew value for the skew loss.")
    coeffs: List[float] = Field(default=[1., 1., 1., 1., 1.], description="Coefficients for the QWAloss function. See module for meaning of each coefficient.")
    loss_type: str = Field(default="BCE", description="Loss type for the QWA model. Options: 'BCE', 'CE'.")
    num_networks: int = Field(default=3, description="Number of networks for the QWA model to process latents. Options: 1, 2, or 3.")
    network_type: str = Field(default="attn", description="Network type for the QWA model. Options: 'mlp', 'atnn', or 'None' to apply the identity function.")
    hidden_dim: int = Field(default=32, description="Only for network_type='mlp'. Hidden expansion factor for the QWA MLP networks.")
    mlp_dropout: float = Field(default=0.1, description="Dropout rate for the QWA MLP networks.")
    attn_dropout: float = Field(default=0., description="Dropout rate for the QWA Attention networks.")
    ff_dropout: float = Field(default=0., description="Dropout rate for the QWA FeedForward networks.")
    norm_mode: str = Field(default="layer", description="Normalization mode for the QWA model. Options: 'batch1d', 'layer', or 'None'.")
    num_heads: int = Field(default=4, description="Only for network_type='attn'. Number of heads for the QWA Attention networks.")
    num_enc_layers: int = Field(default=2, description="Only for network_type='attn'. Number of encoder layers for the QWA model.")
    upper_quantile: float = Field(default=0.9, description="Upper quantile for the quartile separation mechanism.")
    lower_quantile: float = Field(default=0.1, description="Lower quantile for the quartile separation mechanism.")

class PatchTST(BaseModel):
    num_enc_layers: int = Field(default=3, description="Number of encoder layers for the PatchTST model.")
    d_model: int = Field(default=16, description="Model dimension for the PatchTST model.")
    d_ff: int = Field(default=128, description="FeedForward dimension for the PatchTST model.")
    num_heads: int = Field(default=4, description="Number of heads for the PatchTST model.")
    attn_dropout: float = Field(default=0.3, description="Dropout rate for attention mechanism in the PatchTST model.")
    ff_dropout: float = Field(default=0.3, description="Dropout rate for feedforward mechanism in the PatchTST model.")
    norm_mode: str = Field(default="batch1d", description="Normalization mode for the PatchTST model.")

class DLinear(BaseModel):
    moving_avg: int = Field(default=25, description="Moving average window for the DLinear model.")
    individual: bool = Field(default=False, description="Whether to use model channels together or separately.")

class TimesNet(BaseModel):
    num_enc_layers: int = Field(default=2, description="Number of encoder layers for the TimesNet model.")
    d_model: int = Field(default=16, description="Model dimension for the TimesNet model.")
    d_ff: int = Field(default=128, description="FeedForward dimension for the TimesNet model.")
    num_kernels: int = Field(default=6, description="Number of kernels for the TimesNet model.")
    c_out: int = Field(default=1, description="Output channels for the TimesNet model for forecasting.")
    top_k: int = Field(default=3, description="Top k amplitudes used for the periodic slicing block in TimesNet.")
    dropout: float = Field(default=0.3, description="Dropout rate for the TimesNet model.")

class ModernTCN(BaseModel):
    num_enc_layers: List[int] = Field(default=[2], description="Choose from {1, 2, 3} and can make it a list (in str format a,b,c,...) for multistaging with 5 possible stages [a,b,c,d,e] with each element from {1, 2, 3}. For example [1, 1] or [2, 2, 3].")
    d_model: List[int] = Field(default=[16], description="The model dimension (i.e. Conv1D channel dimension) for each stage. Choose from {32, 64, 128, 256, 512}. Make a list (in str format a,b,c,...)  for multistaging, length equal to number of stages.")
    ffn_ratio: int = Field(default=1, description="The expansion factor for the feed-forward networks in each block, d_ffn = d_model*ffn_ratio. Choose from {1, 2, 4, 8}")
    dropout: float = Field(default=0.0, description="Dropout rate for the ModernTCN model.")
    class_dropout: float = Field(default=0.0, description="Dropout rate for the classification head.")
    large_size: List[int] = Field(default=[9], description="Size of the large kernel. Choose from {13, 31, 51, 71}. Make a list (in str format a,b,c,...)  for multistaging, length equal to number of stages.")
    small_size: List[int] = Field(default=[5], description="Size of the small kernel Set to 5 for all experiments. Make a list (in str format a,b,c,...) for multistaging, length equal to number of stages.")
    dw_dims: List[int] = Field(default=[256], description="Depthwise dimension for each stage. Set to 256 for all stages. Make a list (in str format a,b,c,...) for multistaging, length equal to number of stages.")

class ChannelLatentMixing(BaseModel):
    num_enc_layers: int = Field(default=1, description="Number of layers to use for the channel latent representations mixing method's transformer.")
    clm: bool = Field(default=False, description="Whether to use the channel latent representations mixing method.")
    combo: str = Field(default="concat_patch_dim", description="Channel combination type for the OpenNeuro dataset. Options: 'concat_patch_dim' and 'concat_embed_dim'.")

class MonteCarloDropout(BaseModel):
    mcd: bool = Field(default=False, description="Whether to use Monte Carlo Dropout for uncertainty estimation in SSS.")
    stats: List[str] = Field(default=["mean", "std"], description="Statistics to compute for Monte Carlo Dropout. Options: 'mean', 'var', 'std', 'cv', 'q25', 'q75', 'iqr', 'entropy', 'mutual_info', 'kl_div', 'js_div', 'ci_lower', 'ci_upper'.")
    num_samples: int = Field(default=100, description="Number of samples to draw for Monte Carlo Dropout for each example in the batch.")
    mcd_prob: float = Field(default=0.1, description="Dropout rate for Monte Carlo Dropout.")

class Global(BaseModel):
    exp: Experiment = Experiment()
    data: Data = Data()
    ssl: SSL = SSL()
    downstream: Downstream = Downstream()
    sl: SL = SL()
    ddp: DDP = DDP()
    early_stopping: EarlyStopping = EarlyStopping()
    scheduler: Scheduler = Scheduler()
    open_neuro: OpenNeuro = OpenNeuro()
    mlp_mixer: MLPMixer = MLPMixer()
    qwa: QWA = QWA()
    clm: ChannelLatentMixing = ChannelLatentMixing()
    patchtst: PatchTST = PatchTST()
    dlinear: DLinear = DLinear()
    timesnet: TimesNet = TimesNet()
    moderntcn: ModernTCN = ModernTCN()
    mcd: MonteCarloDropout = MonteCarloDropout()
    sparse_context: SparseContext = SparseContext()

def load_config(file_path: str) -> Global:
    print(f"Received file_path in load_config: {file_path}")
    print(f"Absolute file_path in load_config: {os.path.abspath(file_path)}")
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return Global(**config_data)
