'''
Script for evaluating Kalman Filters in the model with piecewise constant disturbances
'''

import numpy as np
from Modified_FourTank_functions import Modified_FourTankSystem
from utils_Modified_FourTank_functions import sim22,linearize_system, discretize_system, steady_state, find_equilibrium
from utils_functions import FourTankSystemSensor, StaticKalmanFilter, DynamicKalmanFilter
from scipy.linalg import expm
import matplotlib.pyplot as plt
from Modified_FourTank_parameters import a1,a2,a3,a4, A1,A2,A3,A4, gamma1,gamma2, g, rho, p

np.random.seed(0)

# Initial liquid levels [cm]
h10, h20, h30, h40 = 0.0, 0.0, 0.0, 0.0
# Convert levels to mass [g] => m = rho * A * h
m10 = rho * A1 * h10
m20 = rho * A2 * h20
m30 = rho * A3 * h30
m40 = rho * A4 * h40

x0 = np.array([m10, m20, m30, m40])  # Initial states
u0 = np.array([300,300])
d0 = np.array([0,0])

dt = 10
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
Steady-state operating point is
xs: {xs}
us: {us}
ys: {ys}
ds: {ds}''')

ys = ys[:2,None]
xs = xs.squeeze()
us = np.tile(us, (1, 361)) 
# us[:,100:] += np.array([100,-100])[:,None]
ds = np.tile(ds, (1, 361)) 

u =us
times = [0,240]
vals = [[300,300],[500,300]]
u = []
for t in ts:
    idx = np.searchsorted(times, t, side="right") - 1
    idx = np.clip(idx, 0, len(vals) - 1)
    
    u.append(vals[idx])        
u = np.array(u).T

x, y, z, T_all, X_all, H_all = sim22(ts, xs, u, ds, p, d_noise_level=1, v_noise_level=2, plot=False)

#Noise matrices (change these, especially Rww)
temp = expm(np.block([[-Ac, Bdc@Bdc.T],[np.zeros_like(Ac),Ac.T]])*dt)
phi11 = temp[:4,:4]
phi12 = temp[:4,4:]
phi22 = temp[4:,4:]

Q = phi22.T@phi12

G = np.eye(4)
Rww = np.eye(4)
Rvv = np.eye(2)*5

Rwv = np.zeros((4,2)) 
Rvw = Rwv.T            # 2x4 cross covariance 

staticKF = StaticKalmanFilter(A,Bu, C ,G,Rww,Rvv,Rwv,P0=np.eye(4)*100, x0=x0)

# #KF predictions
predictions = []
ypredictions = []
for uk, yk in zip(u.T,y.T):
    uk = uk[:,None]-us[:,0,None]
    yk = yk[:2,None]-ys
    staticKF.update(yk, uk) #update to estimate x[k|k]
    xhat, yhat = staticKF.x, staticKF.y
    
    predictions.append(xhat+xs[:,None])
    ypredictions.append(yhat+ys)
static_predictions = np.array(predictions).squeeze()
static_ypredictions = np.array(ypredictions).squeeze()

dynamicKF = DynamicKalmanFilter(A,Bu, C ,G,Rww,Rvv,Rwv,P0=np.eye(4)*100, x0=x0)

#KF predictions
predictions = []
ypredictions = []
for uk, yk in zip(u.T,y.T):
    uk = uk[:,None]-us[:,0,None]
    yk = yk[:2,None]-ys

    dynamicKF.update(yk, uk) #update to estimate x[k|k]
    xhat, yhat = dynamicKF.x, dynamicKF.y
    
    predictions.append(xhat+xs[:,None])
    ypredictions.append(yhat+ys)
dynamic_predictions = np.array(predictions).squeeze()
dynamic_ypredictions = np.array(ypredictions).squeeze()

fig, axs = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Static and dynamc KF')
for i in range(2):
    ax = axs[i]

    ax.plot(ts, y[i,:], 'o', label='Y true', color='blue', markerfacecolor='none')
    ax.plot(ts, static_ypredictions[:,i], '-', label='Y static KF', color = 'red')
    ax.plot(ts, dynamic_ypredictions[:,i], '--', label='Y dynamic KF', color = 'black')
    ax.set_title(f"Water level Tank {i+1}")
    # ax.set_ylim(0, None)
    ax.grid(True)
    
# Only one legend for all subplots:
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()