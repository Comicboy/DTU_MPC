from utils_functions import FourTankSystemSensor, approx_derivative
from utils_Modified_FourTank_functions import Modified_FourTankSystem, sim23
from utils_functions import *
from parameters import p
from utils_Models import OpenLoop_ModelC
from utils_DisturbanceModels import BrownianMotion
import numpy as np
from scipy.optimize import minimize
from utils_KalmanFilters import CDEKF

np.random.seed(0)

class SimulationModelC(ModelSimulation):
    def __init__(self, 
                 ts: np.array, 
                 x0: np.array, 
                 u0: np.array, 
                 d0:np.array, 
                 p:np.array, 
                 disturbances: Tuple[DisturbanceModel,DisturbanceModel], 
                 dt_small: float=0.1,
                 control_schedule=None):
        
        super().__init__(ts, x0, u0, d0, p)
        self.disturbances = disturbances
        self.dt_small = dt_small
        
        self.control_schedule = control_schedule
        if control_schedule != None:
            self.c_times = control_schedule[0]
            self.c_inputs = control_schedule[1]
        
        
    def dynamics(self, t, x, u, d):
        return Modified_FourTankSystem_SDE(t, x, u, d, self.p, self.disturbances)
        
    def control(self, t, u):
        if self.control_schedule == None:
            return u
        else:
            idx = np.searchsorted(self.c_times, t, side="right") - 1
            idx = np.clip(idx, 0, len(self.c_inputs) - 1)
            
        u_new = np.array(self.c_inputs[idx])
        return u_new
    def disturbance(self, t, d, dmin=0, dmax=500):
        # d is stochastic here; actual d handled in solver via Brownian motion
        return d
        
    def step(self, tspan, x, u, d):
        Tvec = np.arange(tspan[0],tspan[1]+self.dt_small, self.dt_small)
        X = np.zeros((len(Tvec), len(x)))
        D = np.zeros((len(Tvec), len(d)))  # disturbance over time

        X[0, :] = x
        D[0, :] = d
        
        x_aug = np.hstack([x, d])  # shape (6,)

        for i in range(len(Tvec)-1):
            
            m = x_aug[:4]
            d = x_aug[4:]
            
            f, sigma = self.dynamics(Tvec[i], m, u, d)
            
            # Euler–Maruyama
            x_aug = x_aug + f*self.dt_small + sigma @ (np.sqrt(self.dt_small) * np.random.randn(2))
            
            X[i+1, :] = x_aug[:4]
            D[i+1, :] = np.clip(x_aug[4:],0,None)
        return Tvec, X,D
    
    def full_output(self, T, X):
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
        
        return T, X, H, Qout
    
    def simulate(self):
        """Main simulation loop"""
        ts = self.ts
        xk_minus1 = self.x0
        dk_minus1 = self.d0
        uk_minus1 = self.u0

        Tfull = [ts[0]]
        Xfull = [xk_minus1]
        Dfull = [dk_minus1]
        Uk = [uk_minus1]
        
        Xk = [xk_minus1]
        Dk = [dk_minus1]
        
        #setup for sensor
        #measurement noise
        v = self.sensor_noise(N=len(ts))
        yk_minus1 = self.SystemSensor(xk_minus1[:,None])+ v[:,0,None]
        Yk = [yk_minus1]
            
        for k in range(1, len(ts)):
            tk = ts[k]
            tk_minus1 = ts[k-1]
            
            T_segment, X_segment, D_segment = self.step([tk_minus1,tk], xk_minus1, uk_minus1, dk_minus1)

            Tfull = np.concatenate((Tfull, T_segment[1:]))
            Xfull = np.vstack((Xfull, X_segment[1:]))
            Dfull = np.vstack((Dfull, D_segment[1:]))
                
            xk = X_segment[-1, :]
            uk = self.control(tk,uk_minus1) # control update
            yk = self.SystemSensor(xk) + v[:,k,None]
            dk = D_segment[-1,:]
            
            
            #update
            xk_minus1 = xk
            uk_minus1 = uk.squeeze()
            yk_minus1 = yk
            dk_minus1 = dk
            
            #discrete samples
            Xk.append(xk)
            Dk.append(dk)
            Yk.append(yk)
            Uk.append(uk)
        
        #discrete states
        Xk = np.hstack([ts[:,None], Xk])
        Uk = np.hstack([ts[:,None], Uk])
        Dk = np.hstack([ts[:,None], Dk])
        Yk = np.array(Yk)
        
        D = np.hstack([Tfull[:,None], Dfull])
        T, X, H, Qout = self.full_output(Tfull, Xfull)
        
        return T, X, H, Qout, D, Xk, Uk, Dk, Yk
        
    def sensor_noise(self, N, R=np.eye(2)*2, Ny=2):
        'add to y'
        Lr = np.linalg.cholesky(R)
        e = np.random.randn(Ny,N)
        v = Lr @ e
        return v

a1, a2, a3, a4, A1, A2, A3, A4, gamma1, gamma2, g, rho = p

# Initial liquid levels [cm]
h10, h20, h30, h40 = 0.0, 0.0, 0.0, 0.0
# Convert levels to mass [g] => m = rho * A * h
m10 = rho * A1 * h10
m20 = rho * A2 * h20
m30 = rho * A3 * h30
m40 = rho * A4 * h40

x0 = np.array([m10, m20, m30, m40])  # Initial states
u0 = np.array([300,300])
d0 = np.array([100,100])

dt = 5
ts = np.arange(0,30*60+dt,dt)
N = len(ts)  # number of intervals
nx = len(x0)
F1 = np.ones_like(ts)*300
F2 = np.ones_like(ts)*300
F3 = np.ones_like(ts)*100
F4 = np.ones_like(ts)*100
u = np.array([F1,F2])
d = np.array([F3,F4])

xs = find_equilibrium(Modified_FourTankSystem, np.array([10000, 10000, 10000, 10000]), np.array([300, 300]), np.array([100, 100]), p)
Ac, Bc, Bdc, Cc, Dc = linearize_system(Modified_FourTankSystem, FourTankSystemSensor, xs, np.array([300, 300]), np.array([100, 100]), p)

#combine u and d
Bc = np.block([Bc, Bdc])
Dc = np.zeros((2,4)) #this is zero in our case

# #discretize 
_, A, B, C, D = discretize_system(Ac,Bc,Cc,Dc, Ts=dt)   
Bu = B[:,:2]
Bd = B[:,2:] 
    
xs, us, ys, ds = steady_state(Modified_FourTankSystem, np.array([10000, 10000, 10000, 10000]), np.array([300, 300]), np.array([100, 100]),p)

print(f'''
xs: {xs}
us: {us}
ys: {ys}
ds: {ds}''')
    
F3disturbance = BrownianMotion(sigma=1) # F3 disturbance model
F4disturbance = BrownianMotion(sigma=1) # F4 disturbance model
ModelC = OpenLoop_ModelC(ts=ts, x0=xs.ravel(), u0=u0, d0 = d0, p=p, disturbances=(F3disturbance,F4disturbance), dt_small=0.1)

T, X, H, Qout, D, Xk, Uk, Dk, Yk = ModelC.simulate()

plot_results(T, X, H, Qout, Uk, D, plot_outputs=['H'])
#Noise matrices
Rvv = np.eye(2)*2

#Test the kalman filter step by step
sigma = np.zeros((4,4))
sigma[2,2] = 1 # sigmaF3
sigma[3,3] = 1 # sigmaF4

kf = CDEKF(p, Modified_FourTankSystem, FourTankSystemSensor, f_jacobian, g_jacobian, sigma, x0=xs, P0=np.eye(4), Rv=Rvv)

#KF predictions
predictions = []
dt_small = dt
n_sub = int(dt / dt_small)
ts_small= np.linspace(0, ts[-1], n_sub * len(ts))
Ps = []
for uk, yk, dk in zip(Uk[:, 1:],Yk, Dk[:,1:]):
    uk = uk[:,None]
    yk = yk[:,None]
    dk = dk[:,None]
    
    Ps.append(kf.P)
    kf.measurement_update(yk)
    for _ in range(n_sub):
        kf.time_update(uk, dk, dt_small)
        predictions.append(kf.x.copy())
predictions = np.array(predictions).squeeze()

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('CDEKF')
for i in range(4):
    ax = axs[i//2, i%2]     # select subplot

    ax.plot(T, X[:,i], label='continuous', color='orange')
    ax.plot(ts, Xk[:,i+1], 'x', label='discrete', color='green')
    ax.plot(ts_small, predictions[:,i], 'r.', label='KF predictions')

    ax.set_title(f"Tank {i+1}")
    # ax.set_ylim(0, None)
    ax.grid(True)
plt.show()
    
y_predictions = np.array([C@pred for pred in predictions])

plt.figure()
plt.title('')
plt.plot(ts,Yk[:,0], 'o', label='Tank 1 Level', markerfacecolor='none')
plt.plot(ts,Yk[:,1], 'o', label='Tank 2 Level', markerfacecolor='none')
plt.plot(ts,y_predictions[:,0], '-', label='KF estimated Tank 1 Level')
plt.plot(ts,y_predictions[:,1], '-', label='KF estimated Tank 2 Level')
plt.grid()
plt.legend()
plt.show()

