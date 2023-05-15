<img src="docs/_static/images/moss.jpg" width="80%">

# Moss: A Python library for Reinforcement Learning

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
  pip install .["dev"]
  ```

After installation, open your python console and type

  ```python
  import moss
  print(moss.__version__)
  ```

If no error occurs, you have successfully installed Moss.

## Quick Start

This is an example of Impala to train Atari game(use [envpool](https://github.com/sail-sg/envpool)).
  ```bash
  python examples/atari_impala.py --task_id Pong-v5
  ```
