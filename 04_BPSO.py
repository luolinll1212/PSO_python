# -*- coding:utf-8 -*-
import numpy as np


def fitness(x, C, W, Bag_Vmax, c):
    total_W = np.sum(x * W)
    total_C = np.sum(x * C)
    if total_C > Bag_Vmax:
        total_W = total_W - alpha * (total_C - Bag_Vmax)
    return total_W

np.random.seed(0)

C = open('c.txt').readlines().strip()
W = open('w.txt').readlines().strip()

particlesize = 50
dim = 10
MaxNum = 100
v_bound = [-10, 10]
w_bound = [0.4, 0.8]
c1 = c2 = 0.5
Bag_Vmax = 300
alpha = 3

x = np.random.randint(0, 2, (particlesize, dim))
v = np.random.uniform(v_bound[0], v_bound[1], (particlesize, dim))

fit = np.zeros(particlesize)
for i in np.arange(particlesize):
    fit[i] = fitness(x[i, :], C, W, Bag_Vmax, alpha)
personal_p = x
personal_fitness = fit
global_best_p = personal_p[np.argmax(personal_fitness)]
global_best_fitness = np.max(personal_fitness)

# 迭代
for step in np.arange(MaxNum):
    vx = np.zeros(dim)
    for i in np.arange(particlesize):
        r1 = np.random.rand()
        r2 = np.random.rand()

        w = w_bound[1] - (w_bound[1] - w_bound[0]) * step / MaxNum  # 惯性权重动态衰减
        v[i, :] = w * v[i, :] + c1 * r1 * (personal_p[i, :] - x[i, :]) + c2 * r2 * (global_best_p - x[i, :])
        for j in np.arange(dim):
            if v[i, j] > np.max(v_bound) or v[i, j] < np.min(v_bound):
                v[i, j] = np.random.uniform(v_bound[0], v_bound[1])

        vx = 1 / (1 + np.exp(-v[i, :]))  # sigmoid函数
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
