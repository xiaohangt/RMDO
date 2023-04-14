# Regret-Minimizing Double Oracle for Extensive-Form Games

This repository is the implementation of the paper Regret-Minimizing Double Oracle for Extensive-Form Games. 

## Before running
### Clone the repo with git submodules
```
git clone --recursive https://github.com/xiaohangt/ODO.git
cd ODO
git submodule update --init --recursive
```

### Set up environments:
```
conda env create -f environment.yml
conda activate xodo
```

### Install dependency(OpenSpiel)
```
# Starting at the repo root
cd dependencies/open_spiel
export BUILD_WITH_ACPC=ON # to compile with the optional universal poker game variant
./install.sh
pip install -e . # This will start a compilation process. Will take a few minutes.
```

## Running Experiments
### Exploitability
To run Extensive-Form Online Double Oracle(with Linear CFR+ as meta solver) and other baselines on Kuhn Poker:
```
python experiments.py -g kuhn_poker -a xodo -m lcfr_plus
python experiments.py -g kuhn_poker -a xdo -m lcfr_plus
python experiments.py -g kuhn_poker -a lcfr_plus
python experiments.py -g kuhn_poker -a lcfr
python experiments.py -g kuhn_poker -a xfp
python experiments.py -g kuhn_poker -a psro
```
Results will be saved in the folder `results` as list.
