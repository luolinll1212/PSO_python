# -*- coding:utf-8 -*-
import numpy as np

def fitness(x, cityDist, particlesize, cityNum):
    total_Dist = np.zeros(particlesize)
    for i in range(particlesize):
        for j in range(cityNum-1):
            total_Dist[i] += cityDist[int(x[i,j]), int(x[i,j+1])]
        total_Dist[i] += cityDist[int(x[i,cityNum-1]), int(x[i,0])]
    return total_Dist

cityCoor = np.array([
    [1304, 2312],
    [3639, 1315],
    [4177, 2244],
    [3712, 1399],
    [3488, 1535],
    [3326, 1556],
    [3238, 1229],
    [4196, 1044],
    [4312, 790],
    [4386, 570],
    [3007, 1970],
    [2562, 1756],
    [2788, 1491],
    [2381, 1676],
    [1332, 695],
    [3715, 1678],
    [3918, 2179],
    [4061, 2370],
    [3780, 2212],
    [3676, 2578],
    [4029, 2838],
    [4263, 2931],
    [3429, 1908],
    [3507, 2376],
    [3394, 2643],
    [3439, 3201],
    [2935, 3240],
    [3140, 3550],
    [2545, 2357],
    [2778, 2826],
    [2370, 2975]
])

cityNum = cityCoor.shape[0]
cityDist = np.zeros((cityNum, cityNum))
for i in range(cityNum):
    for j in range(cityNum):
        if i != j:
            cityDist[i, j] = np.sqrt((cityCoor[i, 0] - cityCoor[j, 0]) ** 2 + (cityCoor[j, 1] - cityCoor[j, 1]) ** 2)
        else:
            cityDist[i, j] = cityDist[j, i]

MaxNum = 200        
particlesize = 1000 
personal_p = np.zeros((particlesize,cityNum))
for i in range(particlesize):
    personal_p[i,:] = np.random.permutation(cityNum) 
personal_fitness = fitness(personal_p, cityDist, particlesize, cityNum) 
global_best_p = personal_p[np.argmin(personal_fitness)] 
global_best_fitness = np.min(personal_fitness)          
record_global_best_p = np.zeros((MaxNum,cityNum))
record_global_best_fitness = np.zeros(MaxNum)
