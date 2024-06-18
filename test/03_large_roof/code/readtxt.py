my_file = open("fulltrack.txt", "r")
data = my_file.read()
my_file.close()
data_into_list = data.split("\n")
print(data_into_list[1].split(" "))


all_game = [['GAME','HV','N_sol','R_s','R_t','U']]
count = 1
for i in range(len(data_into_list)):
    if data_into_list[i][:8] == 'Step 100':
        data_into_list[i] = data_into_list[i].split(" ")
        hypervolume = float(data_into_list[i][4])
        number = int(data_into_list[i][6])
        R_s = float(data_into_list[i][9])
        R_t = float(data_into_list[i][11])
        U_r = float(data_into_list[i][13])
        all_game.append([int(count),hypervolume,number,R_s,R_t,U_r])
        count += 1
import numpy as np
np.savetxt("GFG.csv",
           all_game,
           delimiter =", ",
           fmt ='% s')

