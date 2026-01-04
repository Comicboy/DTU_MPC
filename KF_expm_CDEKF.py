from utils_functions import FourTankSystemSensor, BrownianMotion, CDEKF, OpenLoop_ModelC
from utils_Modified_FourTank_functions import Modified_FourTankSystem
from utils_functions import *
from Modified_FourTank_parameters import p

import numpy as np

np.random.seed(0)


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

