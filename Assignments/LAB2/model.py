import casadi as ca
import numpy as np

# (KGi) - Comments made by Gobi Kumarasamy

class Quadcopter(object):
    def __init__(self,h=0.1,ref=None,ref_ag=None,disturbance=False):
        """
        Quadcopter model class. 
        
        Describes the movement of a Quadcopter with mass 'm'
        All methods should return casadi.MX or casadi.DM variable 
        types.

        :param h: sampling time, defaults to 0.1
        :type h: float, optional
        """

        # Model, gravity and sampling time parameters
        self.disturbance = disturbance
        print("Disturbace enabled ? = ", self.disturbance)

        self.model       = self.Quadcopter_linear_dynamics
        self.model_nl    = self.Quadcopter_nonlinear_dynamics
        self.model_nl_ag = self.Quadcopter_augmented_nonlinear_dynamics

        self.g = 9.81
        self.dt = h

        # System reference and disturbance (w)
        self.x_d    = ref        # 12 states
        self.x_d_ag = ref_ag     # 12 states + 3 error states
        self.w      = 0.0
        print("x_d",self.x_d)
        print("x_d_ag",self.x_d_ag)

        
        # Quadcopter Parameters
        self.m         = 1.4
        self.m_disturb = 0.5

        self.M  = np.diag([0.001,0.001,0.005])
        self.fz = self.m * self.g  # basic_thurst

        # Linearize system around equilibrium with some input
        self.x_eq = [0,0,0,0,0,0,0,0,0,0,0,0]  # 12 states
        self.u_eq = [self.fz,0,0,0]            # 4 inputs

        self.Integrator_lin = None  # Linear
        self.Integrator = None      # Non linear
        self.Integrator_ag = None   # Integrator dynamics

        self.set_integrators()
        self.set_discrete_time_system()

        self.set_augmented_discrete_system()
        self.set_integrators_augmented()


        print("Quadcopter class initialized")

    def set_integrators(self):
        """
        Generate continuous time high-precision integrators.
        """
        
        # Set CasADi variables
        x = ca.MX.sym('x', 12)   # Symbolic variables (KGi)
        u = ca.MX.sym('u', 4)

        # Integration method - integrator options can be adjusted
        options = {"abstol" : 1e-5, "reltol" : 1e-9, "max_num_steps": 100, 
                   "tf" : self.dt}

        # Create linear dynamics integrator
        dae = {'x': x, 'ode': self.model(x,u), 'p':ca.vertcat(u)}
        self.Integrator_lin = ca.integrator('integrator', 'cvodes', dae, options)
        
        #Create nonlinear dynamics integrator
        dae_nl = {'x': x, 'ode': self.model_nl(x,u), 'p':ca.vertcat(u)}
        self.Integrator = ca.integrator('integrator', 'cvodes', dae_nl, options)
       
        print("Integraters are set perfectly !")


    def set_integrators_augmented(self):
        """
        Generate continuous time high-precision integrators.
        """
        
        # Set CasADi variables
        u = ca.MX.sym('u', 4)
        options = {"abstol" : 1e-5, "reltol" : 1e-9, "max_num_steps": 100, 
                   "tf" : self.dt}

        # Create augmented system dynamics integrator
        x_ag = ca.MX.sym('x', 15)
        dae = {'x': x_ag, 'ode': self.model_nl_ag(x_ag,u), 'p':ca.vertcat(u)}
        self.Integrator_nl_ag = ca.integrator('integrator', 'cvodes', dae, options)
        
        print("Augmented integraters are set perfectly !")


    def set_discrete_time_system(self):
        """
        Set discrete-time system matrices from linear continuous dynamics.
        """
        
        # Check for integrator definition
        if self.Integrator_lin is None:
            print("Integrator_lin not defined. Set integrators first.")
            exit()

        # Set CasADi variables
        x = ca.MX.sym('x', 12)
        u = ca.MX.sym('u', 4)
    
        # Jacobian of exact discretization                                 #### Syntax ??? #### 
        self.Ad = ca.Function('jac_x_Ad', [x, u], [ca.jacobian(
                            self.Integrator_lin(x0=x, p=u)['xf'], x)])
        self.Bd = ca.Function('jac_u_Bd', [x, u], [ca.jacobian(
                            self.Integrator_lin(x0=x, p=u)['xf'], u)])


    def set_augmented_discrete_system(self):
        """
        Quadrocopter dynamics with integral action.

        :param x: state
        :type x: casadi.DM
        :param u: control input
        :type u: casadi.DM
        """

        # Grab equilibrium dynamics
        Ad_eq = self.Ad(self.x_eq, self.u_eq)
        Bd_eq = self.Bd(self.x_eq, self.u_eq)

        # Instantiate augmented system
        self.Ad_i = ca.DM.zeros(15,15)
        self.Bd_i = ca.DM.zeros(15,4)

        Ad_i_1 = ca.DM.zeros(12,3) 
        Ad_i_2 = ca.horzcat(Ad_eq,Ad_i_1)

        Bd_i_1 = ca.DM.zeros(3,4)

        Cd_i_1 = -self.dt*ca.DM.eye(3)
        Cd_i_2 = ca.DM.zeros(3,9)
        Cd_i_3 = ca.DM.eye(3)
        Cd_i_4 = ca.horzcat(Cd_i_1,Cd_i_2,Cd_i_3)

        self.R_i = ca.DM.zeros(15,15)
        self.R_i[12,12]=self.R_i[13,13]=self.R_i[14,14] = self.dt


        # Populate matrices
        self.Ad_i = ca.vertcat(Ad_i_2,Cd_i_4)
        self.Bd_i = ca.vertcat(Bd_eq,Bd_i_1)



    def Quadcopter_linear_dynamics(self, x, u, *_):  
        """ 
        Quadcopter continuous-time linearized dynamics.

        :param x: state
        :type x: MX variable, 12x1
        :param u: control input
        :type u: MX variable, 4x1
        :return: dot(x)
        :rtype: MX variable, 12x1
        """

        Ac = ca.MX.zeros(12,12)
        Bc = ca.MX.zeros(12,4)

        theta = 0
        phi = 0
        psi = 0
        f_z = self.fz
        w_x = 0
        w_y = 0
        w_z = 0
        m = self.m
        M_x = self.M[0,0]
        M_y = self.M[1,1]
        M_z = self.M[2,2]

        Ac = ca.DM([[ 0, 0, 0, 1, 0, 0,                                                                               0,                                                     0,                                                            0,                                                0,                                               0,                                                0],
                  [ 0, 0, 0, 0, 1, 0,                                                                               0,                                                     0,                                                            0,                                                0,                                               0,                                                0],
                  [ 0, 0, 0, 0, 0, 1,                                                                               0,                                                     0,                                                            0,                                                0,                                               0,                                                0],
                  [ 0, 0, 0, 0, 0, 0,                    (f_z*(np.cos(theta)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)))/m,                  (f_z*np.cos(phi)*np.cos(psi)*np.cos(theta))/m, (f_z*(np.cos(psi)*np.sin(theta) - np.cos(theta)*np.sin(phi)*np.sin(psi)))/m,                                                0,                                               0,                                                0],
                  [ 0, 0, 0, 0, 0, 0,                   -(f_z*(np.cos(psi)*np.cos(theta) + np.sin(phi)*np.sin(psi)*np.sin(theta)))/m,                  (f_z*np.cos(phi)*np.cos(theta)*np.sin(psi))/m, (f_z*(np.sin(psi)*np.sin(theta) + np.cos(psi)*np.cos(theta)*np.sin(phi)))/m,                                                0,                                               0,                                                0],
                  [ 0, 0, 0, 0, 0, 0,                                                    -(f_z*np.cos(phi)*np.sin(theta))/m,                          -(f_z*np.cos(theta)*np.sin(phi))/m,                                                            0,                                                0,                                               0,                                                0],
                  [ 0, 0, 0, 0, 0, 0,               w_z*np.cos(phi)*(np.tan(theta)**2 + 1) + w_y*np.sin(phi)*(np.tan(theta)**2 + 1),     w_y*np.cos(phi)*np.tan(theta) - w_z*np.sin(phi)*np.tan(theta),                                                            0,                                                1,                             np.sin(phi)*np.tan(theta),                              np.cos(phi)*np.tan(theta)],
                  [ 0, 0, 0, 0, 0, 0,                                                                               0,                         - w_z*np.cos(phi) - w_y*np.sin(phi),                                                            0,                                                0,                                        np.cos(phi),                                        -np.sin(phi)],
                  [ 0, 0, 0, 0, 0, 0, (w_z*np.cos(phi)*np.sin(theta))/np.cos(theta)**2 + (w_y*np.sin(phi)*np.sin(theta))/np.cos(theta)**2, (w_y*np.cos(phi))/np.cos(theta) - (w_z*np.sin(phi))/np.cos(theta),                                                            0,                                                0,                             np.sin(phi)/np.cos(theta),                              np.cos(phi)/np.cos(theta)],
                  [ 0, 0, 0, 0, 0, 0,                                                                               0,                                                     0,                                                            0,                                                0, ((M_y)*(w_z) - (M_z)*(w_z))/M_x,  ((M_y)*(w_y) - (M_z)*(w_y))/M_x],
                  [ 0, 0, 0, 0, 0, 0,                                                                               0,                                                     0,                                                            0, -((M_x)*(w_z) - (M_z)*(w_z))/M_y,                                               0, -((M_x)*(w_x) - (M_z)*(w_x))/M_y],
                  [ 0, 0, 0, 0, 0, 0,                                                                               0,                                                     0,                                                            0,  ((M_x)*(w_y) - (M_y)*(w_y))/M_z, ((M_x)*(w_x) - (M_y)*(w_x))/M_z,                                                0]
                  ])
        
        Bc = ca.DM([[                                                       0,     0,     0,     0],
                    [                                                       0,     0,     0,     0],
                    [                                                       0,     0,     0,     0],
                    [  (np.sin(psi)*np.sin(theta) + np.cos(psi)*np.cos(theta)*np.sin(phi))/m,     0,     0,     0],
                    [ -(np.cos(psi)*np.sin(theta) - np.cos(theta)*np.sin(phi)*np.sin(psi))/m,     0,     0,     0],
                    [                                 (np.cos(phi)*np.cos(theta))/m,     0,     0,     0],
                    [                                                       0,     0,     0,     0],
                    [                                                       0,     0,     0,     0],
                    [                                                       0,     0,     0,     0],
                    [                                                       0, 1/M_x,     0,     0],
                    [                                                       0,     0, 1/M_y,     0],
                    [                                                       0,     0,     0, 1/M_z]
                    ])
   

        ### Store matrices as class variables
        self.Ac = Ac
        self.Bc = Bc 

        return Ac @ x + Bc @ u  

    def Quadcopter_nonlinear_dynamics(self, x, u, *_):
        """
        Quadcopter nonlinear dynamics.

        :param x: state
        :type x: casadi.DM or casadi.MX
        :param u: control input
        :type u: casadi.DM or casadi.MX
        :return: state time derivative
        :rtype: casadi.DM or casadi.MX, depending on inputs
        """

        if self.disturbance == True:
            m_d = self.m_disturb 
        else:
            m_d = self.m_disturb - self.m_disturb  # = 0


        theta = x[6]
        phi   = x[7]
        psi   = x[8]

        w_x = x[9]
        w_y = x[10]
        w_z = x[11]
    
        M_x = self.M[0,0]
        M_y = self.M[1,1]
        M_z = self.M[2,2]

        m   = (self.m)
        
        f_z   = u[0] + (self.fz)
        Tau_x = u[1]
        Tau_y = u[2]
        Tau_z = u[3]

        f1  = x[3]
        f2  = x[4]
        f3  = x[5]
        f4  = (np.sin(psi) * np.sin(theta) + np.cos(psi)* np.sin(phi) * np.cos(theta)) * (f_z / (self.m+m_d))
        f5  = (-np.cos(phi) * np.sin(theta) + np.sin(psi)* np.sin(phi) * np.cos(theta)) * (f_z / (self.m+m_d))
        f6  = (np.cos(phi) * np.cos(theta)) * (f_z / (self.m+m_d)) - self.g
        f7  = w_x + np.sin(phi) * np.tan(theta) * w_y + np.cos(phi) * np.tan(theta) * w_z
        f8  = np.cos(phi) * w_y - np.sin(phi) * w_z  
        f9  =  (np.sin(phi) / np.cos(theta)) * w_y + (np.cos(phi) / np.cos(theta)) * w_z  
        f10 = (1/M_x) * (Tau_x - w_y * M_z * w_z + w_z * M_y * w_y ) 
        f11 = (1/M_y) * (Tau_y - w_z * M_x * w_x + w_x * M_z * w_z )
        f12 = (1/M_z) * (Tau_z - w_x * M_y * w_y + w_y * M_x * w_x )
        
        dxdt = [ f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12 ]
      
        return ca.vertcat(*dxdt)
  

        
    def get_discrete_system_matrices_at_eq(self):
        """
        Evaluate the discretized matrices at the equilibrium point

        :return: A,B,C matrices for equilibrium point
        :rtype: casadi.DM 
        """
        A_eq = self.Ad(self.x_eq, self.u_eq)
        B_eq = self.Bd(self.x_eq, self.u_eq)
        
        # Populate a full observation matrix
        C_eq = ca.DM.eye(12)

        return A_eq, B_eq, C_eq

    # ====================================================================
    def continuous_time_linear_dynamics(self, x0, u):

        out = self.Integrator_lin(x0=x0, p=u)
        return out["xf"]

    def continuous_time_nonlinear_dynamics(self, x0, u):
        out = self.Integrator(x0=x0, p=u)
        return out["xf"]

    def discrete_time_dynamics(self,x0,u):

        return self.Ad(self.x_eq, self.u_eq) @ x0 + \
                self.Bd(self.x_eq, self.u_eq) @ u

    # =======================================================================
    
    def Discrete_augmented_linear_dynamics(self, x, u):   # Mainly for MPC controller
        """Augmented Quadcopter system dynamics
        """
        r = ca.MX.zeros(len(self.x_d_ag))
        r[12,0]= self.x_d[0]
        r[13,0]= self.x_d[1]
        r[14,0]= self.x_d[2]     

        print("r =",r)
        
        A = self.Ad_i
        B = self.Bd_i
        R = self.R_i

        return A @ x + B @ u + R @ r
    # =========================================================================

    def Quadcopter_augmented_nonlinear_dynamics(self, x, u, *_):   # Mainly for system dynamics

        theta = x[6]
        phi   = x[7]
        psi   = x[8]

        w_x = x[9]
        w_y = x[10]
        w_z = x[11]
    
        M_x = self.M[0,0]
        M_y = self.M[1,1]
        M_z = self.M[2,2]

        m   = (self.m)
        
        f_z   = u[0] + (self.fz)
        Tau_x = u[1]
        Tau_y = u[2]
        Tau_z = u[3]

        f1  = x[3]
        f2  = x[4]
        f3  = x[5]
        f4  = (np.sin(psi) * np.sin(theta) + np.cos(psi)* np.sin(phi) * np.cos(theta)) * (f_z / (self.m+self.m_disturb))
        f5  = (-np.cos(phi) * np.sin(theta) + np.sin(psi)* np.sin(phi) * np.cos(theta)) * (f_z / (self.m+self.m_disturb))
        f6  = (np.cos(phi) * np.cos(theta)) * (f_z / (self.m+self.m_disturb)) - self.g
        f7  = w_x + np.sin(phi) * np.tan(theta) * w_y + np.cos(phi) * np.tan(theta) * w_z
        f8  = np.cos(phi) * w_y - np.sin(phi) * w_z  
        f9  =  (np.sin(phi) / np.cos(theta)) * w_y + (np.cos(phi) / np.cos(theta)) * w_z  
        f10 = (1/M_x) * (Tau_x - w_y * M_z * w_z + w_z * M_y * w_y ) 
        f11 = (1/M_y) * (Tau_y - w_z * M_x * w_x + w_x * M_z * w_z )
        f12 = (1/M_z) * (Tau_z - w_x * M_y * w_y + w_y * M_x * w_x )

        f13 = -1*x[0] + 1*self.x_d[0] 
        f14 = -1*x[1]  + 1*self.x_d[1] 
        f15 = -1*x[2]  + 1*self.x_d[2] 


        dxdt = [ f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15 ]
      
        return ca.vertcat(*dxdt)

        # ===================================================================================  
         
    def Continuous_augmented_nonlinear_dynamics(self, x0, u):   
        out = self.Integrator_nl_ag(x0=x0, p=u)
        return out["xf"]  
        # ===================================================================================  


    def get_augmented_discrete_system(self):
        
        return self.Ad_i, self.Bd_i


    pass