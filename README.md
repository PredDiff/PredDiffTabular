# *PredDiff*: Explanations and Interactions from Conditional Expectations


This repository provides all resources to reproduce the paper:
*PredDiff*: Explanations and Interactions from Conditional Expectations

In particular, we provide ready-to-run jupyter notebooks, which apply *PredDiff* on different datasets
* **synthetic regression** (`synthetic_dataset.ipynb`): (Interaction) relevances for a regressor on the synthetic dataset discussed in the paper
* **(tabular) MNIST** (`mnist.ipynb`): (Interaction) relevances for a classifier trained on MNIST seen as a tabular dataset
* **NHANES** (`nhanes.ipynb`): (Interaction) relevances for a classifier trained on the NHANES (mortality regression) dataset

In an additional notebook (`uci_datasets.ipynb`), we invite the user to try *PredDiff*  to obtain (interaction) relevances for models trained on the following additional datasets (not discussed in the paper):
* the **10 UCI regression datasets** investigated by Gal et al 2015 in `https://github.com/yaringal/DropoutUncertaintyExps` (using data as provided in this repository)
* the **7 synthetic classification datasets** proposed by Sikonja et al 2008
* the **UCI Adult** (census income) classification dataset as obtained from `https://archive.ics.uci.edu/ml/datasets/adult`
* the **UCI Bike Sharing** regression dataset as obtained from `https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset`


# Requirements
Install dependencies from `pred_diff.yml` by running `conda env create -f pred_diff.yml` and activate the environment via `conda activate pred_diff`

