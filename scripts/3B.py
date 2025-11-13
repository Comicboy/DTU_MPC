import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, curve_fit
from scipy.linalg import expm
from scipy.linalg import eig
import scipy.linalg
import matplotlib.pyplot as plt
import control as ct
import cvxpy as cp

# import Modified_FourTank_functions as fun   # Import functions
# from Modified_FourTank_parameters import p, a1, a2, a3, a4, A1, A2, A3, A4, g, gamma1, gamma2, rho  # Import parameters

from MPC.functions import *
from MPC.parameters import p

# Model
class ClosedLoop_ModelB(ModelSimulation):
    '''
    Note: Should it be with process noise too? 
    
    Added:
    Piecewise constant disturbance
    Measurement noise
    
    Note: I return sensor, Y but do not plot them
    '''
    def __init__(self, ts, x0, u0, d0, p):
        super().__init__(ts, x0, u0, d0, p)
        self.integral_error = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.tspan = ts[:2]  # sampling time
        self.us = np.array([233, 167])
        self.r = np.array([58, 50]) # set point (steady state when input is [300,200,0,0])
    
    def dynamics(self, t, x, u, d, p):
        return FourTankSystem(t, x, u, d, p)

    def disturbance(self, t, d, dmin=0, dmax=500, sigma=5):
        delta_d = np.random.normal(0, sigma, size=d.shape)
        d = np.clip(d+delta_d, dmin, dmax)
        return d
    
    def control(self, t, y,  Kc=10,Ki=1,Kd=100):
        # PID controller
        us = self.us
        r = self.r
        y = y.flatten()
        tspan = self.tspan
        integral_error = self.integral_error
        prev_error = self.prev_error
        u_clipped, integral_error_new, derivative_error = PIDcontroller(r,y,us,Kc,Ki,Kd,tspan,integral_error,prev_error)
        self.integral_error = integral_error_new
        self.prev_error = derivative_error
        return u_clipped

    def step(self, t_span, x0, u, d):
        sol = solve_ivp(
            fun=lambda t, x: self.dynamics(t, x, u, d, self.p),
            t_span=t_span,
            y0=x0,
            method="BDF",
            dense_output=False
        )
        return sol.t, sol.y.T
    
    def full_output(self, T, X, U, D):
        # Helper variables
        nT, nX = X.shape
        
        # Unpack parameters
        a = self.p[0:4]     # Pipe cross-sectional areas [cm^2]
        A = self.p[4:8]     # Tank cross-sectional areas [cm^2]
        gamma = self.p[8:10] # Valve positions [-]
        g = self.p[10]      # Gravity [cm/s^2]
        rho = self.p[11]    # Density of water [g/cm^3]
        
        # Compute measured variables (liquid levels H)
        H = np.zeros((nT, nX))
        for i in range(nT):
            H[i, :] = X[i, :] / (rho * A)
            
        # Compute the flows out of each tank
        Qout = np.zeros((nT, nX))
        for i in range(nT):
            Qout[i, :] = a * np.sqrt(2 * g * H[i, :])
        # --------------------------------------------------------


        Ut = np.ones((len(self.ts),3))
        Dt = np.ones((len(self.ts),3)) 
        Ut[:,0] = self.ts
        Ut[:,1] = U[:,0]
        Ut[:,2] = U[:,1]
        Dt[:,0] = self.ts
        Dt[:,1] = D[:,0]
        Dt[:,2] = D[:,1]
        
        return T, X, H, Qout, Ut, Dt
    
    def simulate(self):
        """Main simulation loop"""
        ts = self.ts
        x_cur = self.x0
        d_cur = self.d0
        u_cur = self.u0
        

        Tfull = []
        Xfull = []
        Dfull = [d_cur]
        Ufull = [u_cur]
        
        
        #setup for sensor
        #measurement noise
        v = self.sensor_noise(N=len(ts))
        y_cur = self.SystemSensor(x_cur[None,:])+ v[:,0]
        Yfull = [y_cur]

        for i in range(1, len(ts)):
            t_cur = ts[i - 1]
            t_next = ts[i]
            
            u_cur = self.control(t_cur,y_cur)
            d_cur = self.disturbance(t_cur, d_cur)

            T_segment, X_segment = self.step((t_cur, t_next), x_cur, u_cur, d_cur)

            if i == 1:
                Tfull = T_segment
                Xfull = X_segment
            else:
                Tfull = np.concatenate((Tfull, T_segment[1:]))
                Xfull = np.vstack((Xfull, X_segment[1:]))

            
            Dfull.append(d_cur)
            Ufull.append(u_cur)
            x_cur = X_segment[-1, :]
            
            y_cur = self.SystemSensor(x_cur[None,:]) + v[:,i]
            Yfull.append(y_cur)

        T, X, H, Qout, Ut, Dt = self.full_output(Tfull, Xfull, np.array(Ufull), np.array(Dfull))
        return T, X, H, Qout, Ut, Dt, np.array(Yfull)
    
    def process_noise(self, N,Q=np.array([[20**2,0],[0,40**2]]), Nu = 2, seed = 73):
        'add to u'
        np.random.seed(seed)
        
        Lq = np.linalg.cholesky(Q)

        #process noise
        e =  np.random.randn(Nu,N)
        w = Lq @ e
        return w
        
    def sensor_noise(self, N, R=np.eye(2), Ny=2):
        'add to y'
        Lr = np.linalg.cholesky(R)
        e = np.random.randn(Ny,N)
        v = Lr @ e
        return v

# --------------------------------------------------------
# Simulation scenario
# --------------------------------------------------------

t0 = 0.0        # [s] Initial time
tf = 20 * 60    # [s] Final time
dt = 10         # sample time

ts = np.arange(t0, tf+dt, dt)

# Initial liquid masses in tanks [g]
m10 = 0.0
m20 = 0.0
m30 = 0.0
m40 = 0.0

# Flow rates from pumps [cm^3/s]
F1 = 300.0
F2 = 300.0

# Initial state vector
x0 = np.array([m10, m20, m30, m40])

# Input vector
u0 = np.array([F1, F2])
d0 = np.array([100.0, 100.0])  # disturbance starts at 100,100
# --------------------------------------------------------

ModelB = ClosedLoop_ModelB(ts=ts, x0=x0, u0=u0, d0 = d0, p=p)

T, X, H, Qout, Ut, Dt, y = ModelB.simulate()
plot_results(T, X, H, Qout, Ut, Dt, plot_outputs=['H'])

plt.figure()
plt.title('Measurements y')
plt.plot(ts, y[:,:,0].squeeze(), label='Tank 1 Level')
plt.plot(ts, y[:,:,1].squeeze(), label='Tank 2 Level')
plt.grid()
plt.legend()
plt.show()