import numpy as np
import casadi as ca 

N = 20
m = 7
ki = 1000
gc = 9.81

# x = [y1 z1 y2 z2 .. yn zn]
#        1     2    

H = ca.DM.zeros(2*N, 2*N)
for i in range(0,2*N-2):
    H[i, i+2] = -1.0
    H[i+2, i] = -1.0

for i in range(0,2*N):
    H[i, i] = 2.0

# Replace first and last terms of the diagonal with 1.0
H[0,0] = H[1,1] = H[-2,-2] = H[-1, -1] = 1.0
H = ki*H

# Array indexing
# x = [ 1 2 3 4 5 6 7 8 ... N-2  N-1  N]
#       0 1 2                -3  -2  -1
print("H:\n", H)

g = ca.DM.zeros(2*N)

# Iterate through given positions in the array with the syntax 
#                  start:stop:step
# g = [ 0  gc*m  0  gc*m  0 .... 0 gc*m]
g[1::2] = gc*m

print ("G: \n", g)

# Before: g[1::2] = [0, 39.24, 0, 39.24, 0, 39.24, 0, 39.24]
# After:  g[1:-2:2] = [0, 39.24, 0, 39.24, 0,   0,   0,   0]

# Initialize lbx and ubx:
lbx = -np.inf*np.ones(2*N)
ubx = np.inf*np.ones(2*N)

# Fix the last and first mass elements
lbx[0] = ubx[0] = -2
lbx[1] = ubx[1] = 1
lbx[-2] = ubx[-2] = 2
lbx[-1] = ubx[-1] = 1

# Add the ground constraint to our system
# lbx[3:-2:2] = 0.5

# Tilted ground constraint:
A = ca.DM.zeros(N, 2*N)
for k in range(N):
    A[k, 2*k] = -0.1
    A[k, 2*k+1] = 1.0

lba = 0.5*ca.DM.ones(N,1)
uba = np.inf*ca.DM.ones(N,1)

print("Bounds for our system:\nLBx: ", lbx, "\nUBx: ", ubx)

#### Just with slanted ground
# qp = {'h': H.sparsity()}
# S = ca.conic('hc', 'qpoases', qp)
# sol = S(h=H, g=g, lbx=lbx, ubx=ubx)
# print("Sol: \n", sol['x'])
# x_opt = sol['x']

# With A matrix
qp = {'h': H.sparsity(), 'a': A.sparsity()}
S = ca.conic('hc', 'qpoases', qp)
sol = S(h=H, g=g, a=A, lbx=lbx, ubx=ubx, lba=lba, uba=uba)
print("Sol: \n", sol['x'])
x_opt = sol['x']

Y0 = x_opt[0::2]
Z0 = x_opt[1::2]

import matplotlib.pyplot as plt
plt.plot(Y0,Z0,'o-')
ys = ca.linspace(-2.,2.,100)
# zs = 0.5*ca.DM.ones(100,1) #+ 0.1*ys
zs = 0.5+0.1*ys
plt.plot(ys,zs,'--')
plt.xlabel('y [m]')
plt.ylabel('z [m]')
plt.title('hanging chain QP')
plt.grid(True)
plt.legend(['Chain'],loc=9)
plt.legend(['Chain','z - 0.1y >= 0.5'],loc=9)
plt.show()
