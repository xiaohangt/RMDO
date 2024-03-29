# Regret-Minimizing Double Oracle for Extensive-Form Games

This repository is the official implementation of the paper "Regret-Minimizing Double Oracle for Extensive-Form Games", ICML 2023. [[paper](https://arxiv.org/abs/2304.10498)]

## Before running
### Clone the repo with git submodules
```commandline
git clone --recursive https://github.com/xiaohangt/RMDO.git
cd RMDO
git submodule update --init --recursive
```

### Set up environments:
```commandline
conda env create -f environment.yaml
conda activate xodo
```
or
```commandline
pip3 install -r requirements.txt
```

### Install dependency(OpenSpiel)
```commandline
# Starting at the repo root
cd dependencies
export BUILD_WITH_ACPC=ON # to compile with the optional universal poker game variant
./install.sh
pip3 install -e . # This will start a compilation process. Will take a few minutes.
```

## Running Experiments
### Exploitability
To run Extensive-Form Online Double Oracle (with CFR+ as meta solver):
```commandline
python rmdo.py --game kuhn_poker --algorithm XODO --meta_solver cfr_plus
```

To run Extensive-Form Double Oracle (with CFR+ as meta solver) and other baselines.
```commandline
python baselines.py --game kuhn_poker --algorithm dxdo
python baselines.py --game kuhn_poker --algorithm cfr
python baselines.py --game kuhn_poker --algorithm xfp
```

To run Periodic Double Oracle (with CFR+ as meta solver):
```commandline
python rmdo.py --game kuhn_poker --algorithm PDO --meta_solver cfr_plus --meta_iterations 50
```

Results will be saved in the folder `results` as list.
