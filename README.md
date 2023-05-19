# Learning to Sequence, Blend and Refine Robotics Skills with Reinforcement Learning 

## Installation
To install the library, clone the repository and install the related packages with
```
pip install -r requirements.txt
```
However, install the following from their respective sources:
* PyTorch and TorchVision from their [official website](https://pytorch.org/). 
* `calvin_env` dependencies from [their repo](https://github.com/mees/calvin_env).

## CALVINExperiment

### Step 0: Download CALVIN dataset
Download the [CALVIN dataset](https://github.com/mees/calvin) and place it inside [data/](./examples/CALVINExperiment/data/). 

### Step 1: Extract skill demos from the CALVIN dataset
Configure [config/demos_extract.yaml](./examples/CALVINExperiment/config/demos_extract.yaml).
```
> cd examples/CALVINExperiment/
> python demos_extract/extract.py
```

### Step 2: Train and evaluate skill libraries (Dynamical Systems) with ManifoldGMM 
Configure [config/train_ds.yaml](./examples/CALVINExperiment/config/train_ds.yaml).
```
> cd examples/CALVINExperiment/
> python dynsys/train_skills.py
```
Configure [config/eval_ds.yaml](./examples/CALVINExperiment/config/eval_ds.yaml).
```
> cd examples/CALVINExperiment/
> python dynsys/eval_skills.py
```

### Step 3: Train RL Agent
Configure [config/seqblend_rl.yaml](./examples/CALVINExperiment/config/seqblend_rl.yaml).
```
> cd examples/CALVINExperiment/
> python seqblend/seqblend_rl.py
```

## Reference
If you found this code useful for you work, we are delighted! Please consider citing the following reference:
```
@article{ABC,
  author={ABC},
  title={Learning to Sequence, Blend and Refine Robot Skills with Reinforcement Learning},
  year={2023},
  journal = {ABC}
}
```
