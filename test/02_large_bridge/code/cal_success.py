'''
==================================================================
This file export result into csv file
このファイルはCSVファイルに結果をエクスポートします
==================================================================
Read filename in the 'Log_strain energy_aaa' folder and create csv file from the name
'''
#--------------------------------
# Import part / インポート部
#--------------------------------
import os, sys
import re
import math
import numpy as np
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt

#--------------------------------
# Parameters /  パラメーター
#--------------------------------
path = ['LogHV_Hyper-Step_2023-04-10_19_29_09'] # folder path / folder path
see = None # for debug [0:Game No ,1:Initial value ,2:Final value ,3:Reward] / デバッグ用[0：ゲーム番号、1：初期値、2：最終値、3：報酬]
name = '00_result' # filename of the output / 出b力のファイル名

#--------------------------------
# Function to export game-initial-final-reward to csv file / game-initial-final-rewardをcsvファイルにエクスポートする関数
#--------------------------------
def cal_success(path,see=None):
    dirs = os.listdir( path )
    # Lists to contain data in each category / 各カテゴリのデータを含むリスト
    rewardlist = []
    for_table = []
    result_game =[] # Game number / ゲーム番号
    result_finH =[] # Initial value / 初期値
    result_numH =[] # Initial value / 初期値
    result_R0 =[] # Final value / Final value
    result_R1 =[] # Reward / 報酬
    result_R2 =[] # Reward / 報酬
    result_Gr =[] # Reward / 報酬

    # Read each filename and put data into lists / 各ファイル名を読み取り、データをリストに入れます
    for file in dirs:



        for_table.append([
            int(file.split('_')[1])
            ,float(file.split('_')[3])
            ,float(file.split('_')[5])
            ,float(file.split('_')[7])
            ,float(file.split('_')[9])
            ,float(file.split('_')[11])
            ,float(file.split('_')[13][:-4])])
        result_game.append(int(file.split('_')[1]))
        result_finH.append(float(file.split('_')[3]))
        result_numH.append(float(file.split('_')[5]))
        result_R0.append(float(file.split('_')[7]))
        result_R1.append(float(file.split('_')[9]))
        result_R2.append(float(file.split('_')[11]))
        result_Gr.append(float(file.split('_')[13][:-4]))
        if for_table[-1][2] < for_table[-1][1]:
            for_table[-1].append(1) # success
        else:
            for_table[-1].append(0) # fail


    # Export and Save .cvs file /.csvファイルのエクスポートと保存
    result = {
    'Game':result_game,
    'finH':result_finH,
    'numH':result_numH,
    'R0':result_R0,
    'R1':result_R1,
    'R2':result_R2,
    'Gr':result_Gr
    }
    dataframe = pd.DataFrame(result)
    dataframe.to_csv('{}.csv'.format(name))

#--------------------------------
# Main function / 主な機能
#--------------------------------
for i in range(len(path)):
    cal_success(path[i],see)
