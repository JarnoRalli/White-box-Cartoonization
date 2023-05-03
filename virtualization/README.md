# Conda

In order to create a virtual environment to execute the model, you need to install [miniconda](https://docs.conda.io/en/latest/miniconda.html)

## Libmamba Solver

Conda's own solver is very slow, so I recommend using `Libmamba`. To use the new solver, first update conda in the base environment:

```bash
conda update -n base conda
```

Then install and activate `Libmamba` as the solver:

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

## Create Virtual Environment

You can create a new virtual environment as follows:

```bash
conda env create -f whiteboxcartoon.yml
```

You can activatet the environment as follows:

```bash
conda activate whiteboxcartoon
```

