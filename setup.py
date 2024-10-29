# Copyright 2020 DeepMind Technologies Limited and Google LLC
# Copyright 2023 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Setup for pip package."""

import unittest
from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    "jax[cuda12_pip]==0.4.23",
    "nvidia_cudnn_cu12<9.0",
    "absl-py==1.4.0",
    "attrs==21.2.0",
    "h5py==3.8.0",
    # "chex==0.1.5",
    "chex",
    # "kfac_jax==0.0.3",
    "kfac_jax==0.0.5",
    "ml-collections==0.1.1",
    # "optax==0.1.4",
    "optax==0.2.1",
    # "flax==0.6.1",
    "flax==0.8.0",
    # "numpy==1.21.5",
    "numpy==1.25.1",
    # "pyscf==2.1.1",
    "pyscf>=2.1.1",
    "pyblock==0.6",
    # "scipy==1.7.3",
    "scipy<=1.12.0",
    # "tables==3.7.0",
    "tables==3.10.0",
    "pandas==1.3.5",
    # "typing_extensions==4.5.0",
    "typing_extensions==4.7.1",
    # "dm-haiku==0.0.9",
    "dm-haiku==0.0.10",
]


setup(
    name="lapnet",
    version="0.0",
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    platforms=["any"],
    license="Apache 2.0",
)
