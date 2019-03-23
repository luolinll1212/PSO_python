# -*- coding:utf-8 -*-
import copy
import numpy as np


def fitness(x, C, W, Bag_Vmax, c):
    total_W = np.sum(x * W)
    total_C = np.sum(x * C)
    if total_C > Bag_Vmax:
        total_W = total_W - alpha * (total_C - Bag_Vmax)
        # total_W = 0
    return total_W


C = open('c.txt').readlines().strip()
W = open('w.txt').readlines().strip()

particlesize = 100
dim = 10
MaxNum = 200
v_bound = [-10, 10]
w_bound = [0.2, 0.8]
c1 = c2 = 0.5
Bag_Vmax = 1000
alpha = 3

x = np.random.randint(0, 2, (particlesize, dim))
v = np.random.uniform(v_bound[0], v_bound[1], (particlesize, dim))

fit = np.zeros(particlesize)
for i in np.arange(particlesize):
    fit[i] = fitness(x[i, :], C, W, Bag_Vmax, alpha)
personal_p = copy.deepcopy(x)
personal_fitness = copy.deepcopy(fit)
global_best_p = personal_p[np.argmax(personal_fitness)]
global_best_fitness = np.max(personal_fitness)

# train
for step in np.arange(MaxNum):
    vx = np.zeros(dim)
    for i in np.arange(particlesize):
        r1 = np.random.rand()
        r2 = np.random.rand()

        w = w_bound[1] - (w_bound[1] - w_bound[0]) * step / MaxNum  
        v[i, :] = w * v[i, :] + c1 * r1 * (personal_p[i, :] - x[i, :]) + c2 * r2 * (global_best_p - x[i, :])
        for j in np.arange(dim):
            if v[i, j] > np.max(v_bound) or v[i, j] < np.min(v_bound):
                v[i, j] = np.random.uniform(v_bound[0], v_bound[1])

        vx = 1 / (1 + np.exp(-v[i, :]))  # sigmoid function
        for j in np.arange(dim):
            if vx[j] > np.random.rand():
                x[i, j] = 1
            else:
                x[i, j] = 0

        current_fitness = fitness(x[i, :], C, W, Bag_Vmax, alpha)
        if current_fitness > personal_fitness[i]:
            personal_p[i, :] = x[i, :]
            personal_fitness[i] = current_fitness
        if personal_fitness[i] > global_best_fitness:
            global_best_p = personal_p[i, :]
            global_best_fitness = personal_fitness[i]
