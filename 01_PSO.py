# -*- coding:utf-8 -*-
import numpy as np

def fitness(x):
    return -x**2 + 3

EO = 1e-6
particlesize = 50
dim = 1
MaxNum = 50
x_bound = [0, 20]
v_bound = [-1, 1]
w = 0.8
c1 = c2 = 0.5
x = np.random.uniform(x_bound[0], x_bound[1], (particlesize, dim))
v = np.random.rand(particlesize, dim)
fit = fitness(x)
personal_p = x
personal_fitness = fit
global_best_p = x[np.argmax(fit)]
global_best_fitness = np.max(fit)

# train
record = np.zeros(shape=(MaxNum))
for step in range(MaxNum):
    r1 = np.random.rand(particlesize, dim)
    r2 = np.random.rand(particlesize, dim)
    v = w * v + c1 * r1 * (personal_p - x) + c2 * r2 * (global_best_p - x)
    v[v > np.max(v_bound)] = np.max(v_bound)
    v[v < np.min(v_bound)] = np.min(v_bound)

    x = x + v
    x[x > np.max(x_bound)] = np.max(x_bound)
    x[x < np.min(x_bound)] = np.min(x_bound)

    fit = fitness(x)
    personal_greater_id = np.greater(fit, personal_fitness)
    personal_p[personal_greater_id] = x[personal_greater_id]
    personal_fitness[personal_greater_id] = fit[personal_greater_id]
    if np.max(personal_fitness) > global_best_fitness:
        global_best_p = personal_p[np.argmax(personal_fitness)]
        global_best_fitness = np.max(personal_fitness)

    record[step] = global_best_fitness
    # if step ==1 and abs(record[step]) < EO:
    #     break
    # if np.abs(record[step] - record[step-1]) < EO:
    #     break

print(record)

