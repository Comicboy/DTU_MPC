import numpy as np
from utils_functions import (linearize_system, discretize_system, steady_state, find_equilibrium, ModelSimulation, 
                          FourTankSystemSensor, Modified_FourTankSystem_SDE, plot_results, idare)
from utils_Models import SimulationModelC
from scipy.linalg import expm
from utils_Modified_FourTank_functions import Modified_FourTankSystem
from utils_DisturbanceModels import BrownianMotion
from utils_KalmanFilters import StaticKalmanFilter, DynamicKalmanFilter
import matplotlib.pyplot as plt
from Modified_FourTank_parameters import p, a1, a2, a3, a4, A1, A2, A3, A4, g, gamma1, gamma2, rho  # Import parameters
    
if __name__ == '__main__':  
    print('''
Description of what is outputted when running this script:

A nonlinear continuous-time simulation for the Modified Tank System is performed, where the disturbances are piecewise constant 
and increments away from mean [F3,F4] = [100,100] drawn from a normal distribution with variance of 100 (If disturbances are 
drawn to be negative, the value is clipped to 0)

Then the Kalman Filters, static and dynamic, are tested around the operating point using [F1,F2] = [300,300], initial value and mean [F3,F4] = [100,100].
The specific operating point is printed.


          ''')
    
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
    d0 = np.array([100,100])
    
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
    
    xs = xs.ravel()
    us = us.ravel()
    ds = ds.ravel()
    
    # c_schedule = [[0,60,120,240,480, 600, 1200],[[300,300], [400,300],[400,200],[300,200], [350,250], [250,350] ,[300,300]]]
    c_schedule = [[0,240], [[300,300],[500,300]]]
    
    F3disturbance = BrownianMotion(sigma=0.5) # F3 disturbance model
    F4disturbance = BrownianMotion(sigma=0.5) # F4 disturbance model
    ModelC = SimulationModelC(ts=ts, x0=xs, u0=us, d0 = ds, p=p, disturbances=(F3disturbance,F4disturbance), dt_small=0.1, control_schedule=c_schedule)
    T, X, H, Qout, D, Xk, Uk, Dk, Yk = ModelC.simulate()
    plot_results(T, X, H, Qout, Uk, Dk, plot_outputs=['H'])
    
    #Noise matrices (change these, especially Rww)
    temp = expm(np.block([[-Ac, Bdc@Bdc.T],[np.zeros_like(Ac),Ac.T]])*dt)
    phi11 = temp[:4,:4]
    phi12 = temp[:4,4:]
    phi22 = temp[4:,4:]

    Q = phi22.T@phi12
    
    G = np.eye(4)
    Rww = Q
    G = np.eye(4)
    Rww = np.eye(4)
    Rvv = np.eye(2)

    Rwv = np.zeros((4,2)) 
    Rvw = Rwv.T            # 2x4 cross covariance 

    staticKF = StaticKalmanFilter(A,Bu, C ,G,Rww,Rvv,Rwv,P0=np.eye(4)*100, x0=xs-xs)

    # #KF predictions
    predictions = []
    ypredictions = []
    for uk, yk in zip(Uk[:,1:],Yk):
        uk = uk[:,None]-us[:,None]
        yk = yk-ys
        
        staticKF.update(yk, uk) #update to estimate x[k|k]
        xhat, yhat = staticKF.x, staticKF.y
        
        predictions.append(xhat+xs[:,None])
        ypredictions.append(yhat+ys)
        
    static_predictions = np.array(predictions)
    static_ypredictions = np.array(ypredictions)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Static KF')
    for i in range(4):
        ax = axs[i//2, i%2]     # select subplot

        ax.plot(T, X[:,i], label='continuous', color='orange')
        ax.plot(ts, Xk[:,i+1], 'x', label='discrete', color='green')
        ax.plot(ts, static_predictions[:,i], 'r.', label='KF predictions')

        ax.set_title(f"Tank {i+1}")
        # ax.set_ylim(0, None)
        ax.grid(True)
        
    # Only one legend for all subplots:
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Static KF')
    for i in range(2):
        ax = axs[i]

        ax.plot(ts, Yk[:,i], label='continuous', color='orange')
        ax.plot(ts, static_ypredictions[:,i], 'r.', label='KF predictions')

        ax.set_title(f"Tank {i+1}")
        # ax.set_ylim(0, None)
        ax.grid(True)
        
    # Only one legend for all subplots:
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
        
    
    print('Prediction for next 8 steps')
    N = 8
    u_pred = u[:N,:N].reshape(-1,1) #u to shape (N*2,1)
    Zk = staticKF.j_step(u=u_pred, N=N)
    Zk = Zk.reshape(N, 2)
    print(Zk)

    dynamicKF = DynamicKalmanFilter(A,Bu, C ,G,Rww,Rvv,Rwv,P0=np.eye(4)*100, x0=xs-xs)
    
    # #KF predictions
    predictions = []
    ypredictions = []
    for uk, yk in zip(Uk[:,1:],Yk):
        uk = uk[:,None]-us[:,None]
        yk = yk-ys
    
        dynamicKF.update(yk, uk) #update to estimate x[k|k]
        xhat, yhat = dynamicKF.x, dynamicKF.y 

        predictions.append(xhat+xs[:,None])
        ypredictions.append(yhat+ys)
        
    dynamic_predictions = np.array(predictions)
    dynamic_ypredictions = np.array(ypredictions)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Dynamic KF')
    for i in range(4):
        ax = axs[i//2, i%2]     # select subplot

        ax.plot(T, X[:,i], label='continuous', color='orange')
        ax.plot(ts, Xk[:,i+1], 'x', label='discrete', color='green')
        ax.plot(ts, dynamic_predictions[:,i], 'r.', label='KF predictions')

        ax.set_title(f"Tank {i+1}")
        # ax.set_ylim(0, None)
        ax.grid(True)
        
    # Only one legend for all subplots:
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Dynamic KF')
    for i in range(2):
        ax = axs[i]

        ax.plot(ts, Yk[:,i], label='continuous', color='orange')
        ax.plot(ts, dynamic_ypredictions[:,i], 'r.', label='KF predictions')

        ax.set_title(f"Tank {i+1}")
        # ax.set_ylim(0, None)
        ax.grid(True)
        
    # Only one legend for all subplots:
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
        
    
    print('Prediction for next 8 steps')
    N = 8
    u_pred = u[:N,:N].reshape(-1,1) #u to shape (N*2,1)
    Zk = dynamicKF.j_step(u=u_pred, N=N)
    Zk = Zk.reshape(N, 2)
    print(Zk)


    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    for i in range(2):
        ax = axs[i]

        ax.plot(ts, Yk[:,i], 'o', label='Y measurements', color='blue', markerfacecolor='none')
        ax.plot(ts, static_ypredictions[:,i], label='Static KF filtered Y', color = 'red')
        ax.plot(ts, dynamic_ypredictions[:,i], label='Dynamic KF filtered Y', color = 'black')

        ax.set_title(f"Water level Tank {i+1}")
        # ax.set_ylim(0, None)
        ax.grid(True)
        
    # Only one legend for all subplots:
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()