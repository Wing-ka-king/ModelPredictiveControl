from model import Quadcopter
from simulation import EmbeddedSimEnvironment
import numpy as np
import scipy
from mpc import MPC
import casadi as ca


# #================================================================================================================
# ## QUADROTOR DYNAMICS - NO DISTURBANCE NO INTEGRAL ACTION

# # Create Quadcopter and controller objects

ref    = np.array([0.0,0.0,0.0,0,0,0,0,0,0,0,0,0])       # 12 states
ref_ag = np.array([0.0,0.0,0.0,0,0,0,0,0,0,0,0,0,0,0,0]) # 12 states + 3 error states
h = 0.1

Quadcopter_orginal = Quadcopter(h,ref,ref_ag,disturbance = False)

# Get the system discrete-time dynamics
A, B, C = Quadcopter_orginal.get_discrete_system_matrices_at_eq()

# Solve the ARE for our system to extract the terminal weight matrix P
Q = np.diag([1,1,1,1,1,1,1,1,1,1,1,1])
R = np.diag([1,1,1,1])
P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

# Instantiate controller
max_fz  = 4 * Quadcopter_orginal.m * Quadcopter_orginal.g
max_tau = (max_fz/2)*(17.5/100)
tau_z = 0.01

ctl = MPC(model=Quadcopter_orginal, 
          dynamics=Quadcopter_orginal.discrete_time_dynamics,    # Linear MPC
          Q = Q , R = R, P = P,
          horizon=0.5,
          ulb=[-max_fz,-max_tau,-max_tau,-tau_z], 
          uub=[max_fz,max_tau,max_tau,tau_z], 
          xlb=[-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.pi/2,-np.pi/2,-np.pi/2,-np.inf,-np.inf,-np.inf], 
          xub=[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.pi/2,np.pi/2,np.pi/2,np.inf,np.inf,np.inf],
          integrator=False) 

ctl.set_reference(x_sp=np.array([0,0,0.0,0,0,0,0,0,0,0,0,0]))

# Q2-1
ctl.set_constant_control([0, 0, 0, 0])
sim_env = EmbeddedSimEnvironment(model=Quadcopter_orginal, 
                                dynamics=Quadcopter_orginal.discrete_time_dynamics,
                                controller=ctl.constant_control,
                                time = 15)
#t, y, u = sim_env.run(x0 = [0,0,0,0,0,0,0,0,0,0,0,0])

# Q2-2
ctl.set_constant_control([0.1, 0, 0, 0])
sim_env = EmbeddedSimEnvironment(model=Quadcopter_orginal, 
                                dynamics=Quadcopter_orginal.discrete_time_dynamics,
                                controller=ctl.constant_control,
                                time = 15)
#t, y, u = sim_env.run(x0 = [0,0,0,0,0,0,0,0,0,0,0,0])

# Q2-3
ctl.set_constant_control([0.1, -0.001, 0, 0])
sim_env = EmbeddedSimEnvironment(model=Quadcopter_orginal, 
                                dynamics=Quadcopter_orginal.discrete_time_dynamics,
                                controller=ctl.constant_control,
                                time = 15)
#t, y, u = sim_env.run(x0 = [0,0,0,0,0,0,0,0,0,0,0,0])

# Q2-4
ctl.set_constant_control([0.1, 0, 0.001, 0])
sim_env = EmbeddedSimEnvironment(model=Quadcopter_orginal, 
                                dynamics=Quadcopter_orginal.discrete_time_dynamics,
                                controller=ctl.constant_control,
                                time = 15)
#t, y, u = sim_env.run(x0 = [0,0,0,0,0,0,0,0,0,0,0,0])

# #================================================================================================================
# ## MPC FOR A QUADROTOR - NO DISTURBANCE NO INTEGRAL ACTION

# Q3-1
ctl.set_reference(x_sp=np.array([0.0,0.0,0.5,0,0,0,0,0,0,0,0,0]))

sim_env = EmbeddedSimEnvironment(model=Quadcopter_orginal, 
                                dynamics=Quadcopter_orginal.discrete_time_dynamics,
                                controller=ctl.mpc_controller,
                                time = 15)
#t, y, u = sim_env.run(x0 = [0,0,0,0,0,0,0,0,0,0,0,0])

# Q3-2
ctl.set_reference(x_sp=np.array([0.0,0.5,0.5,0,0,0,0,0,0,0,0,0]))

sim_env = EmbeddedSimEnvironment(model=Quadcopter_orginal, 
                                dynamics=Quadcopter_orginal.discrete_time_dynamics,
                                controller=ctl.mpc_controller,
                                time = 15)
#t, y, u = sim_env.run(x0 = [0,0,0,0,0,0,0,0,0,0,0,0])

# Q3-3
ctl.set_reference(x_sp=np.array([0.5,0.5,0.5,0,0,0,0,0,0,0,0,0]))

sim_env = EmbeddedSimEnvironment(model=Quadcopter_orginal, 
                                dynamics=Quadcopter_orginal.discrete_time_dynamics,
                                controller=ctl.mpc_controller,
                                time = 15)
t, y, u = sim_env.run(x0 = [0,0,0,0,0,0,0,0,0,0,0,0])

# Q3-Non Linear
ctl.set_reference(x_sp=np.array([0.5,0.5,0.5,0,0,0,0,0,0,0,0,0]))

sim_env = EmbeddedSimEnvironment(model=Quadcopter_orginal, 
                                dynamics=Quadcopter_orginal.continuous_time_nonlinear_dynamics,
                                controller=ctl.mpc_controller,
                                time = 15)

t, y, u = sim_env.run(x0 = [0,0,0,0,0,0,0,0,0,0,0,0])

 # ================================================================================================================
# # INTEGRATOR DYNAMICS I

# # With disturbance and NO integral action (NORMAL SYSTEM)
# Create Quadcopter and controller objects

ref    = np.array([0.5,0.5,0.5,0,0,0,0,0,0,0,0,0])       # 12 states
ref_ag = np.array([0.5,0.5,0.5,0,0,0,0,0,0,0,0,0,0,0,0]) # 12 states + 3 error states

Quadcopter_disturb = Quadcopter(h,ref,ref_ag,disturbance = True)    # If disturbance = True --> then the system dynamics will be 
                                                            # having different mass compared to MPC controller
                                                            # If disturbance = False --> system and MPC have same mass
                                                            #
                                                            # Note: 
                                                            # The mass disturbance predominently affects the 
                                                            # Z axis displacement/velocity

# Get the system discrete-time dynamics
A, B, C = Quadcopter_disturb.get_discrete_system_matrices_at_eq()

# Instantiate controller
max_fz  = 4 * Quadcopter_disturb.m  * Quadcopter_orginal.g
max_tau = (max_fz/2)*(17.5/100)
tau_z = 0.01       

# Solve the ARE for our system to extract the terminal weight matrix P
Q = np.diag([1,1,1,1,1,1,1,1,1,1,1,1])
R = np.diag([1/(max_fz)**2, 1/(max_tau)**2,
             1/(max_tau)**2, 1/0.01**2])
P = np.matrix(scipy.linalg.solve_discrete_are(A,B, Q, R))

ctl = MPC(model=Quadcopter_disturb, 
          dynamics=Quadcopter_disturb.discrete_time_dynamics, 
          Q = Q , R = R, P = P,
          horizon=0.5,
          ulb=[-max_fz,-max_tau,-max_tau,-tau_z], 
          uub=[max_fz,max_tau,max_tau,tau_z], 
          xlb=[-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.pi/2,-np.pi/2,-np.pi/2,-np.inf,-np.inf,-np.inf], 
          xub=[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.pi/2,np.pi/2,np.pi/2,np.inf,np.inf,np.inf],
          integrator=False
          ) 

ctl.set_reference(x_sp=ref)

sim_env = EmbeddedSimEnvironment(model=Quadcopter_disturb, 
                                dynamics=Quadcopter_disturb.continuous_time_nonlinear_dynamics,
                                controller=ctl.mpc_controller,
                                time = 15)

t, y, u = sim_env.run(x0 = [0,0,0,0,0,0,0,0,0,0,0,0])

# # With disturbance and With integral action (AUGMENTED SYSTEM)

# Get the system augmented discrete-time dynamics
Ai, Bi = Quadcopter_disturb.get_augmented_discrete_system()
          
# Solve the ARE for our system to extract the terminal weight matrix P
Q = np.diag([1,1,1,1,1,1,1,1,1,1,1,1,  1,1,1])              
R = np.diag([1/(max_fz)**2, 1/(max_tau)**2,
              1/(max_tau)**2, 1/0.01**2])
P = np.matrix(scipy.linalg.solve_discrete_are(Ai,Bi,Q,R))

ulb=[-max_fz,-max_tau,-max_tau,-tau_z]
uub=[max_fz,max_tau,max_tau,tau_z]
xlb=[-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.pi/2,-np.pi/2,-np.pi/2,-np.inf,-np.inf,-np.inf,   -1,-1,-1 ]
xub=[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.pi/2,np.pi/2,np.pi/2,np.inf,np.inf,np.inf,    1,1,1]

ctl = MPC(model=Quadcopter_disturb, 
          dynamics=Quadcopter_disturb.Discrete_augmented_linear_dynamics, 
          Q = Q , R = R, P = P,
          horizon=1,ulb=ulb,uub=uub,xlb=xlb,xub=xub,
          integrator=True
          ) 

ctl.set_reference(x_sp=ref_ag)

sim_env = EmbeddedSimEnvironment(model=Quadcopter_disturb, 
                                dynamics=Quadcopter_disturb.Continuous_augmented_nonlinear_dynamics,
                                controller=ctl.mpc_controller,
                                time = 15)

t, y, u = sim_env.run(x0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
