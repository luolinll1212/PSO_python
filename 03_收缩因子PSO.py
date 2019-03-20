# -*- coding:utf-8 -*-
import numpy as np

def fitness(x):
    pass


EO = 1e-6
particlesize = 50
dim = 1
MaxNum = 50
x_bound = [0, 20]
v_bount = [-1, 1]
t1 = 2
t2 = 4
theta = t1+t2
K = 2/np.abs(2-theta-np.sqrt(np.square(theta) - 4*theta))           # t1+t2 > 4
x = np.random.uniform(x_bound[0], x_bound[1], (particlesize, dim))
v = np.random.rand(particlesize, dim)
fit = fitness(x)
personal_p = x
personal_fitness = fit
global_best_p = x[np.argmax(fit)]
global_best_fitness = np.max(fit)

# train
for step in range(MaxNum):
    r1 = np.random.rand(particlesize, dim)
    r2 = np.random.rand(particlesize, dim)
    v = K*(v + t1 * r1 * (personal_p - x) + t2 * r2 * (global_best_p - x))
    v[v > np.max(v_bount)] = np.max(v_bount)
    v[v < np.min(v_bount)] = np.min(v_bount)

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

    # if step ==1 and abs(record[step]) < EO:
    #     break
    # if np.abs(record[step] - record[step-1]) < EO:
    #     break

