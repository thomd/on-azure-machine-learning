# Azure ML Experiments

These are experiments inspired by [Azure Machine Learning Python SDK notebooks](https://github.com/Azure/MachineLearningNotebooks).

## Setup

    conda install jupyterlab
    conda install watermark
    conda install nbdime
    PIP_REQUIRE_VIRTUALENV=false pip install azureml-core
    conda install scikit-learn
    conda install conda
    conda install matplotlib

## Start Jupyter notebooks

    jupyter lab

## Experiments

* [train-model-local/train-local.ipynb](https://github.com/thomd/on-azure-machine-learning/blob/main/train-model-local/train-local.ipynb): train a model using local computer as compute target
* [train-model-aml-compute/train-aml-compute.ipynb](https://github.com/thomd/on-azure-machine-learning/blob/main/train-model-aml-compute/train-aml-compute.ipynb): train a model using Azure ML as compute target
