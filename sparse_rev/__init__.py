from jax import config
config.update("jax_enable_x64", True)

from importlib import import_module
from pathlib import Path
import os

_parent_dir = Path(__file__).parents[1]

_tmp_dir = os.path.join(_parent_dir, 'tmp')
os.makedirs(_tmp_dir, exist_ok=True)