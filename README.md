# MOP-truss-MARL

## Notice
The document will be updated soon.

## References
- Kupwiwat, C. T., Hayashi, K., & Ohsaki, M. (2024). Multi-objective optimization of truss structure using multi-agent reinforcement learning and graph representation. Engineering Applications of Artificial Intelligence, 129, 107594. [https://doi.org/10.1016/j.engappai.2023.107594]
  
## Installation
1. Clone the repository
```shell
git clone https://github.com/kupc25648/MOP-truss-MARL.git
cd MOP-truss-MARL
```
2. Install Python dependencies
```shell
python -m venv MOPtrussMARL
source MOPtrussMARL/bin/activate  
cd environments
pip install -r requirements.txt
```
## Train the models
1. Run
```shell
source MOPtrussMARL/bin/activate  
cd train/code
python -m master_DDPG_truss2D_MO
```
2. Process result: to process results into .csv files use ```train/code/cal_success.py``` <br>

## Test the models
1. copy ```model/2000pickle_base``` to ```test/[subdirectory]```<br>
2. Run<br>
```shell
source MOPtrussMARL/bin/activate  
cd test/[subdirectory]/code
python -m master_DDPG_truss2D_MO
```

## Results
Result files are zipped and named as result.zip in each folder.
