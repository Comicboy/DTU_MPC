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

#Solver and drift/diffussion for disturbance models
class FourTankEM(EulerMaruyama):
    ''' 
    This class uses the Euler-Maruyama method to simulate the four-tank system with process noise
    2 distrubances affecting tanks 3 and 4
    No drift in disturbances, besdides that from the system dynamics
    Constant diffusion matrix, with diffusivity D
    '''
    
    def ffun(self, t, x, u, d, p):
        '''Drift function f for four-tank system
        
        This implies no drift in disturbances
        '''
        
        return FourTankSystem(t, x, u, d, p)

    def gfun(self, t, x, u, d, p, D=4):
        '''Diffusion matrix G for process noise, constant diffussion D, same for both disturbances'''
        G = np.zeros((4, 2))
        G[2, 0] = D  # noise 1 → tank 3
        G[3, 1] = D  # noise 2 → tank 4
        return G
    
# Model
class OpenLoop_ModelC(ModelSimulation): 
    def __init__(self, ts, x0, u0, d0, p, solver: EulerMaruyama, dt_wiener=0.1):
        super().__init__(ts, x0, u0, d0, p)
        self.solver = solver
        self.dt_wiener = dt_wiener
        self.nWiener = solver.gfun(0, x0, u0, d0, p).shape[1]  # infer noise dim

        # Preallocate full Wiener process
        self.total_steps = int((ts[-1] - ts[0]) / dt_wiener)
        _, _, self.dW = wiener_process(ts[1] - ts[0], self.total_steps, self.nWiener, seed=42)

    def dynamics(self, t, x, u, d, p):
        return self.solver.ffun(t, x, u, d, p)
    
    def control(self, t, u):
        return u

    def disturbance(self, t, d, dmin=0, dmax=500):
        # d is stochastic here; actual d handled in solver via Wiener increments
        return d
    
    def step(self, Tvec, x, u, d, dW_segment):
        X, ds = self.solver.run(Tvec, x, u, d, self.dt_wiener, dW_segment, self.p)
        return Tvec, X, ds

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
        Dt = np.ones((len(T),3)) 
        Ut[:,0] = self.ts
        Ut[:,1] = U[:,0]
        Ut[:,2] = U[:,1]
        Dt[:,0] = T
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
        Dfull = d_cur[None,:]
        Dfull = [d_cur]
        Ufull = [u_cur]
        
        # setup wiener increments storage
        tfinal = ts[-1]
        wienerN = int(tfinal / self.dt_wiener)
        _, _, dW = wiener_process(tfinal, wienerN, 2, seed=42) 
        
        #setup for sensor
        #measurement noise
        v = self.sensor_noise(N=len(ts))
        y_cur = self.SystemSensor(x_cur[None,:])+ v[:,0]
        Yfull = [y_cur]

        for i in range(1, len(ts)):
            t_cur = ts[i - 1]
            t_next = ts[i]
            
            N = int((t_next-t_cur) / self.dt_wiener)
            Tvec = np.linspace(t_cur, t_next, N+1)
            # Extract relevant Wiener increments for this segment
            dW_segment = dW[:, (i-1)*N : i*N]
            
            u_cur = self.control(t_cur,u_cur)
            d_cur = self.disturbance(t_cur, d_cur)

            T_segment, X_segment, D_segment = self.step(Tvec, x_cur, u_cur, d_cur, dW_segment)

            if i == 1:
                Tfull = T_segment
                Xfull = X_segment
                Dfull = D_segment
            else:
                Tfull = np.concatenate((Tfull, T_segment[1:]))
                Xfull = np.vstack((Xfull, X_segment[1:]))
                Dfull = np.vstack((Dfull, D_segment[1:]))
                
            Ufull.append(u_cur)
    
            x_cur = X_segment[-1, :]
            d_cur = D_segment[-1,:]    
            
            y_cur = self.SystemSensor(x_cur[None,:]) + v[:,i]
            Yfull.append(y_cur)

        T, X, H, Qout, Ut, Dt = self.full_output(Tfull, Xfull, np.array(Ufull), np.array(Dfull))
        return T, X, H, Qout, Ut, Dt, np.array(Yfull)
        
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

ModelC = OpenLoop_ModelC(ts=ts, x0=x0, u0=u0, d0 = d0, p=p, solver=FourTankEM())

T, X, H, Qout, Ut, Dt, y = ModelC.simulate()
plot_results(T, X, H, Qout, Ut, Dt, plot_outputs=['H'])

plt.figure()
plt.title('Measurements y')
plt.plot(ts, y[:,:,0].squeeze(), label='Tank 1 Level')
plt.plot(ts, y[:,:,1].squeeze(), label='Tank 2 Level')
plt.grid()
plt.legend()
plt.show()