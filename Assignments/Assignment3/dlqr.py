import casadi as ca
import numpy as np
import scipy.linalg
from control.matlab import lqr

class DLQR(object):

    def __init__(self, A, B, C,
                       Q=ca.DM.eye(4), R=ca.DM.ones(1,1)):
        """
        Discrete-time LQR class.
        """

        # System matrices
        self.A = A 
        self.B = B
        self.C = C
                
        self.Q = Q
        self.R = R 

        self.K = None
        self.P = None

        self.int = None

        print(self)                             # You can comment this line

    def __str__(self):
        return """
            Linear-Quadratic Regulator class for discrete-time systems.
            Implements the following controllers:
            u_t = - K @ x                       - method: feedback(x)
            u_t = - K @ x + l_r @ r             - method: feedfwd_feedback(x,r)
            u_t = - K @ x - l_i @ i + l_r @ r   - method: lqr_ff_fb_integrator(x,i,r)
            where:
              x   - system state
              r   - system position reference
              i   - integral state
              K   - LQR feedback gain matrix
              l_r - feed-forward gain
              l_i - integral gain 
        """

    def set_system(self, A, B, C):
        """
        Set system matrices.

        :param A: state space A matrix
        :type A: casadi.DM
        :param B: state space B matrix
        :type B: casadi.DM
        :param C: state space C matrix
        :type C: casadi.DM
        """
        self.A = A
        self.B = B
        self.C = C


    def get_lqr_gain(self, Q=None, R=None):
        """
        Get LQR feedback gain.

        :return: LQR feedback gain K
        :rtype: casadi.DM
        :return: LQR infinite-horizon weight P
        :rtype: casadi.DM
        """
        A_np = np.asarray(self.A)
        B_np = np.asarray(self.B)
        
        if Q is None:
            Q_np = np.asarray(self.Q)
        else:
            Q_np = Q

        if R is None:
            R_np = np.asarray(self.R)
        else:
            R_np = R

        P_np = np.matrix(scipy.linalg.solve_discrete_are(A_np, 
                              B_np, Q_np, R_np))
        K_np = np.matrix(scipy.linalg.inv(B_np.T @ P_np @ B_np
                              + R_np)@(B_np.T @ P_np @ A_np))
        
        self.P = ca.DM.zeros(4,4).full()        
        self.P = P_np

        self.K = ca.DM.zeros(1,4).full()
        self.K = K_np
        
        return self.K, self.P

    def get_feedforward_gain(self, K=None):
        """
        Get the feedforward gain lr.

        :param L: close loop gain, defaults to None
        :type L: list, optional
        """

        if K is None and self.K is not None:
            K = self.K
        elif K is None:  
            print("Please provide an LQR gain K.")
            exit()       

        self.lr = ca.inv(self.C @ ca.inv(ca.DM.eye(self.A.size1())-(self.A-self.B @ K)) @ self.B)
        return self.lr

    def feedback(self, x, K=None):
        """
        State feedback LQR.

        :param x: state
        :type x: casadi.DM
        :param K: LQR feedback gain
        :type K: casadi.DM, 1x4
        """

        if K is None and self.K is not None:
            K = self.K

        u = - K @ x 
        return u

    def feedfwd_feedback(self, x, r=10.0, K=None, lr=None):
        """
        State feedback LQR with feedforward gain.

        :param x: state
        :type x: casadi.DM
        :param r: cart position reference
        :type r: float
        :param K: LQR feedback gain
        :type K: casadi.DM, 1x4
        """

        if K is None and self.K is not None:
            K = self.K
        if lr is None and self.lr is not None:
            lr = self.lr

        u = - K @ x + lr @ r
        return u

    def set_li(self, li):
        """Set integral gain function.

        :param li: integral gain
        :type li: float
        """
        self.l_i = li
    
    def set_lr(self,lr):
        """
        Set lr gain.

        :param lr: feedforward gain
        :type lr: scalar
        """
        self.lr = lr

    def set_lqr_feedback(self,K):
        """
        Set LQR feedback gain.

        :param K: LQR feedback gain
        :type K: casadi.DM, 1x4
        """
        self.K = K

    def lqr_ff_fb_integrator(self, x, r=10.0, K=None, lr=None, li=None):
        """
        State feedback LQR with feedforward gain and integral action.

        :param x: state
        :type x: casadi.DM
        :param r: cart position reference
        :type r: float
        :param K: LQR feedback gain
        :type K: casadi.DM, 1x4
        """
        
        x_t = x[0:4,:]
        i = x[4,:]

        # Argument checks
        if K is None and self.K is not None:
            K = self.K
        elif K is None:
            print("Please set feedforward gain.")

        if lr is None and self.lr is not None:
            lr = self.lr
        elif lr is None:
            print("Please set feedforward gain.")

        _K = K[0,0:4] 
        _li = K[0,4]

        # Fill the correct control law below
        u = -_K @ x_t - _li * i + lr * r
        #u = -K @ x + lr * r
        return u
