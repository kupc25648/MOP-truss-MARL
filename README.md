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
3. Train the models<br>
3.1 Run<br>
```shell
source MOPtrussMARL/bin/activate  
cd train/code
python -m master_DDPG_truss2D_MO
```
3.2 Process result: to process results into .csv files use ```train/code/cal_success.py``` <br>

4. Test the models<br>
copy ```model/2000pickle_base``` to ```test/[subdirectory]```<br>
4.1 Run<br>
```shell
source MOPtrussMARL/bin/activate  
cd test/[subdirectory]/code
python -m master_DDPG_truss2D_MO
```
