# influenza-USA

An age- and space stratified Influenza model for the USA.

## Installation

1. Install conda environment

```bash
conda env create -f influenza_USA_env.yml
```

or alternatively, to update the environment,

```bash
conda env update -f influenza_USA_env.yml --prune
```

2. Install model package inside conda environment

```bash
conda activate INFLUENZA-USA
pip install -e .
```