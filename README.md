# MOP-truss-MARL

## Notice
The codes and data will be updated soon. Please stay tuned.

## References
- [To be added]
  
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
3. Train the models
3.1 Run
```shell
source MOPtrussMARL/bin/activate  
cd train/code
python -m master_DDPG_truss2D_MO
```
3.2 Process result
use ```train/code/cal_success.py``` to process results to .csv files
4. Test the models
4.1 Run
copy ```model/2000pickle_base``` to ```test/[subdirectory]```
run
```shell
source MOPtrussMARL/bin/activate  
cd test/[subdirectory]/code
python -m master_DDPG_truss2D_MO
```
