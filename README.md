# Code for TADRED: TAsk-DRiven Experimental Design in imaging

TADRED identifies the most informative channel-subset whilst simultaneously training a network to execute the task given the subset.

TADRED is a novel method for TAsk-DRiven experimental design in imaging.  TADRED couples feature scoring and task execution in consecutive networks.  The scoring and subsampling procedure enables efficient identification of subsets of complementarily informative channels jointly with training a high-performing network for the task.  TADRED also gradually reduces the full set of samples stepwise to obtain the subsamples, which improves optimization.


## Citation

Please consider citing our paper:

@article{<br>
&nbsp; &nbsp; title={Experimental Design for Multi-Channel Imaging via Task-Driven Feature Selection},<br>
&nbsp; &nbsp; author={Stefano B. Blumberg and Paddy J. Slator and Daniel C. Alexander},<br>
&nbsp; &nbsp; journal={In: International Conference on Learning Representations (ICLR)},<br>
&nbsp; &nbsp; year={2024}<br>
}

## Contact

stefano.blumberg.17@ucl.ac.uk

## Installation Part 1: Environment

First create an environment and enter it, we use Python v3.10.4.  We provide two examples either using Pyenv or Conda:

## Pyenv

```bash
# Pyenv documentation is [link](https://github.com/pyenv), where <INSTALL_DIR> is the directory the virtual environment is installed in.
python3.10 -m venv <INSTALL_DIR>/TADRED_env # Use compatible Python version e.g. 3.10.4
. <INSTALL_DIR>/TADRED_env/bin/activate
```

## Conda

```bash
# Conda documentation is [link](https://docs.conda.io/en/latest/), where <INSTALL_DIR> is the directory the virtual environment is installed in.
conda create -n tadred python=3.10.4
conda activate tadred
```

## Installation Part 2: Packages and Code

Code requires: pytorch, numpy, pyyaml, hydra.

Code is tested using PyTorch v2.0.0, cuda 11.7 on the GPU.

We provide examples of installing packages, using pip,

### Python Package from Source

```bash
pip install git+https://github.com/sbb-gh/tadred.git@main
```

### Using pip

```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 # Install PyTorch 2.0
pip install pyyaml hydra-core==1.3 # Install PyYAML and Hydra
```
Then clone this repository.

## Arguments/Options and Running from the Command Line

Please see config in [types.py](./tadred_code/types.py) for base arguments and descriptions.

To run from the command line:

```bash
python train_and_eval.py --cfg <YAML_CONFIG_PATH>
```

where <YAML_CONFIG_PATH> is a path to a config file.

## License

This project is licensed under the terms of the Apache 2.0 license. For more details, see the LICENSE file in the root of this repository.

## Acknowledgments

Many thanks to David Perez-Suarez, Stefan Piatek, Tom Young, who provided valuable feedback on the code.
