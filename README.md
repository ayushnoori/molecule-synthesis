# Multi-objective generative AI for designing novel brain-targeting small molecules

## Project Summary

The strict selectivity of the blood-brain barrier (BBB) represents one of the most formidable challenges to successful central nervous system (CNS) drug delivery, preventing the diagnosis and treatment of CNS disorders. Computational methods to generate BBB permeable lead compounds _in silico_ may be valuable tools in the CNS drug design pipeline. However, in real-world applications, BBB penetration alone is insufficient; rather, after transiting the BBB, molecules must perform some desired function – such as binding to a specific target or receptor in the brain – and must also be safe and non-toxic for use in human patients.

To discover small molecules that concurrently satisfy these constraints, we use multi‑objective generative AI to synthesize drug-like blood-brain-barrier permeable small molecules that also have high predicted binding affinity to a disease-relevant CNS target. Specifically, we computationally synthesize molecules with predicted bioactivity against dopamine receptor $D_2$, the primary target for almost all clinically effective antipsychotic drugs. After training several graph neural network-based property predictors, we adapt [SyntheMol](https://github.com/swansonk14/SyntheMol), a recently developed Monte Carlo Tree Search-based algorithm for antibiotic design, to perform a multi‑objective guided traversal over an easily synthesizable molecular space.

We design a library of 26,581 novel and diverse small molecules containing hits with high predicted BBB permeability and favorable predicted safety and toxicity profiles, and that could readily be synthesized for experimental validation in the wet lab. We also validate top scoring molecules with molecular docking simulation against the $D_2$ receptor and demonstrate predicted binding affinity on par with risperidone, a clinically prescribed $D_2$-targeting antipsychotic. In the future, the SyntheMol-based computational approach described here may enable the discovery of novel neurotherapeutics for currently intractable disorders of the CNS.

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
* The [PyTorch](https://pytorch.org/) open source machine learning framework for Python.

Individual dependencies are also specified in each script.

Activate the `synthesis_env` virtual environment with the following:

```
source setup.sh
```
If desired, a Jupyter kernel can be created with the following:

```
source setup_jupyter.sh
```
