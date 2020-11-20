import numpy as np
import casadi as ca 

# DM - is purely floating point matrices 
# MX and SX - contain symbolic variables

# X will be the number of tickets
# X = [x1 x2] : x1 - first class tickets; x2 - second class tickets
lb_x1 = 20
lb_x2 = 35

profit_x1 = 2000
profit_x2 = 2000

max_passengers = 150

# x1 + x2 <= 130 -> [1 1] @ [x1; x2] <= 130
A = ca.DM.ones(1,2)
ub_a = max_passengers
lb_a = lb_x1 + lb_x2 

# Initializing the bounds for our problem 
ub_x = np.inf*np.ones(2)
lb_x = -np.inf*np.ones(2)
lb_x[0] = lb_x1
lb_x[1] = lb_x2

H = ca.DM.zeros(2,2)

g = np.zeros((2,1))
g[0,0] = profit_x1
g[1,0] = profit_x2
g = -1*g

qp = {'h': H.sparsity(), 'a': A.sparsity() }
S = ca.conic('S','osqp',qp)
r = S(h=H, g=g, a=A, lbx=lb_x, ubx=ub_x, lba=lb_a, uba=ub_a)

print("r[x]", r['x'])

print("Optimal sol is selling", int(round(float(r['x'][0]))), "first class and", int(round(float(r['x'][1]))), "second class tickets")
