<div align="center">
  <img src="docs/_static/images/moss.jpg" width="65%">
</div>

# Moss: A Python library for Reinforcement Learning

[![PyPI Python Version](https://img.shields.io/pypi/pyversions/moss-rl)](https://pypi.org/project/moss-rl/)
[![PyPI](https://img.shields.io/pypi/v/moss-rl)](https://pypi.org/project/moss-rl/)
[![GitHub license](https://img.shields.io/github/license/hilanzy/moss)](https://github.com/hilanzy/moss/blob/master/LICENSE)

**Moss** is a Python library for Reinforcement Learning based on [jax](https://github.com/google/jax).

## Installation

To get up and running quickly just follow the steps below:

  **Installing from PyPI**: Moss is currently hosted on [PyPI](https://pypi.org/project/moss-rl/),
  you can simply install Moss from PyPI with the following command:

  ```bash
  pip install moss-rl
  ```

  **Installing from github**: If you are interested in running Moss as a developer,
  you can do so by cloning the Moss GitHub repository and then executing following command
  from the main directory (where `setup.py` is located):

  ```bash
  pip install -e ".[dev]"
  ```

After installation, open your python console and type

  ```python
  import moss
  print(moss.__version__)
  ```

If no error occurs, you have successfully installed Moss.

**Work on GPU or TPU**

If you want to run Moss with NVIDIA GPU, please run the steps below:

  ```bash
  pip install --upgrade pip

  pip uninstall jax jaxlib -y

  # CUDA 12 installation
  # Note: wheels only available on linux.
  pip install --upgrade "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

  # CUDA 11 installation
  # Note: wheels only available on linux.
  pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  ```

Or if you want to run with Google Cloud TPU:

  ```bash
  pip install "jax[tpu]==0.4.20" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  ```

For more details, please see the JAX installation instructions [here](https://github.com/google/jax/tree/jax-v0.4.9#installation).

## Quick Start

This is an example of Impala to train Atari game(use [envpool](https://github.com/sail-sg/envpool)).
  ```bash
  python impala.py --task_id Pong-v5 --learning_rate 1e-3
  ```
