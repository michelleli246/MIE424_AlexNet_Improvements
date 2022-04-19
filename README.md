# MIE424_AlexNet_Improvements

## Installation

This section will cover the requirements to set up the training environment.

1. Install PyTorch Version 1.11.0 with CUDA 11.3

```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

2. Install the required packages specified in requirements.txt

```
pip install -r requirements.txt
```

## Experiments

### Weight Regularization

To reproduce the weight regularization experiment results, run the following script

```
python main_weight_regularization.py
```
The model `.pt` files will be saved in the `./models` directory, and the plots will be saved in the `./images` directory.


### Activation Function

To reproduce the activation function experiment results, run the following script

```
python main_activation.py
```
The model `.pt` files will be saved in the `./models` directory, and the plots will be saved in the `./images` directory.

### Learning Rate

To repoduce the learning rate experiment results, run the following script
```
python main_learning_rate.py
```
The model `.pt` files will be saved in the `./models` directory, and the plots will be saved in the `./images` directory.

### Optimizer

To repoduce the optimizer experiment results, run the following jupyter notebook
```
main_optimizer.ipynb
```
By default, the model `.pt` files will be saved in the `MIE424Project/models` directory on Google Drive, and the output training loss, training accuracy, and validation accuracy numbers will be saved in the `MIE424Project/Results` directory on Google Drive.
To run the experiments locally, edit the `model_save_path` and `result_save_path` variables to a local directory.
