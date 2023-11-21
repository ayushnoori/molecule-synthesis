# Synthesis of BBB-Permeable CNS Target Binders

## Project Summary

Inspired by Stokes et al. (forthcoming in *Nature Machine Intelligence*, see paper [here](https://edstem.org/us/courses/42364/discussion/3715178)), we will develop a Monte Carlo Tree Search (MCTS) method to synthesize small molecules that satisfy three constraints imposed by the property predictor:

**First, blood-brain-barrier (BBB) permeability.** We will extend prior work on predicting BBB permeability (see [ayushnoori/graph-bbb](https://github.com/ayushnoori/graph-bbb) on GitHub) to develop our property predictor. We will also leverage functionality from the [Therapeutics Data Commons](https://tdcommons.ai/) project. To train the property predictor, we will use the Blood-Brain-Barrier Dataset (B3DB), published in *Nature Scientific Data* in 2021: 

Meng, F., Xi, Y., Huang, J. & Ayers, P. W. [A curated diverse molecular database of blood-brain barrier permeability with chemical descriptors.](https://www.nature.com/articles/s41597-021-01069-5) *Sci Data* **8**, 289 (2021).

Please also see [theochem/B3DB](https://github.com/theochem/B3DB) and [Issue #174 of mims-harvard/TDC](https://github.com/mims-harvard/TDC/issues/174) on GitHub. 

**Second, binding affinity to a central nervous system (CNS) target,** such as the dopamine type 2 receptor (DRD2) (a task which stochastic iterative target augmentation, presented [here](https://edstem.org/us/courses/42364/discussion/3715178), was also benchmarked against in the ICML 2020 paper). To train the DRD2 binding affinity predictor, we could leverage the dataset from [Olivecrona *et al.* (2017)](https://arxiv.org/abs/1704.07555), which they describe as:

> "The dopamine type 2 receptor DRD2 was chosen as the target, and corresponding bioactivity data was extracted from ExCAPE-DB [33]. In this dataset there are 7218 actives (pIC50 > 5) and 343204 inactives (pIC50 < 5). A subset of 100 000 inactive compounds was randomly selected."

**Third, safety and toxicity,** which could be prioritized (just like solubility) during the molecular generation process (*e.g.*, via the Lipinski rule of 5).

Finally, we will also evaluate the performance of our method at a program synthesis task – *e.g.*, Karel programs, as in Bunel *et al.* (2018) – where instead of combining molecular fragments from the Enamine REAL space, we will successively combine primitives from a program bank to form higher-order programs (see, for example, [this demonstration](https://huggingface.co/spaces/ayushnoori/program-synthesis)).

## Installation

To install the code, please clone this repository with the following:

```bash
git clone git@github.com:ayushnoori/molecule-synthesis.git
cd molecule-synthesis
```

Create a virtual environment.

```
conda deactivate
pip install virtualenv
virtualenv synthesis_env
source synthesis_env/bin/activate
```

Install necessary packages specified in `requirements.txt`.

```bash
pip install -r requirements.txt
```

To save the specific versions of each package required, run the following:

```bash
pip freeze > requirements-frozen.txt
```

## Dependencies

To run the code, please install:
* The [R](https://www.r-project.org/) programming language and statistical computing environment (as well as the [RStudio](https://rstudio.com/) integrated development environment).
* The [Python](https://www.python.org/) programming language.

Individual dependencies are also specified in each script. Along with data manipulation and visualization packages, these include:
* The [PyTorch](https://pytorch.org/) open source machine learning framework for Python.
* The [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) library for geometric deep learning on graphs and manifolds.
* The [PyTorch Lightning](https://www.pytorchlightning.ai/) lightweight PyTorch wrapper for high-performance AI research.

Activate the `synthesis_env` virtual environment with the following:

```
source setup.sh
```
If desired, a Jupyter kernel can be created with the following:

```
source setup_jupyter.sh
```
