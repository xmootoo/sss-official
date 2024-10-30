# Stochastic Sparse Sampling
A framework for variable-length time series classification (VSTC) with local interpretability, tailored for medical time series.

## Installation
To install the required packages, run the following commands:
```bash
pip install -r requirements.txt
pip install -e .
```

## Data
To download the data, run the following commands:
```bash
chmod u+x download_data.sh
./download_data.sh
```

## Experiments
To reproduce the experiments, run the following commands:
```python
python main.py <model>
```
where `<model>` is one of the following:
- `sss`: Stochastic Sparse Sampling
- `finite-context/dlinear`: DLinear.
- `finite-context/patchtst`: PatchTST.
- `finite-context/timesnet`: TimesNet.
- `finite-context/moderntcn`: ModernTCN.
- `infinite-context/mamba`: Mamba.
- `infinite-context/gru`: GRUs.
- `infinite-context/lstm`: LSTMs.
- `infinite-context/rocket`: ROCKET.

The results will be saved in the `logs` folder. To enable Distributed Data Parallel (DDP) or change any other configurations, you must edit `sss/jobs/exp/<model>/args.yaml`.
