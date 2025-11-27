import numpy as np
from Modified_FourTank_functions import Modified_FourTankSystem, find_equilibrium, linearize_system, discretize_system, sim22


def idare(A, C, G, Rww, Rvv, Rwv, P0=None, tol=1e-9, max_iter=200):
    n = A.shape[0]
    P = np.eye(n) if P0 is None else P0.copy()

    for _ in range(max_iter):
        Re = C @ P @ C.T + Rvv
        K = (A @ P @ C.T + G @ Rwv) @ np.linalg.inv(Re)
        P_new = A @ P @ A.T + G @ Rww @ G.T - K @ Re @ K.T

        if np.linalg.norm(P_new - P) < tol:
            break
        P = P_new

    return P

class StaticKalmanFilter:
    def __init__(self, A, B, C, G, Rww, Rvv, Rwv, P0, x0):

        self.A = A
        self.B = B 
        self.C = C
        self.G = G # process noise matrix

        self.Rww = Rww
        self.Rvv = Rvv
        self.Rwv = Rwv
        self.Rvw = Rwv.T

        # State estimate and covariance
        self.x = x0.reshape(-1, 1)
        self.P = P0

        #compute stationary filter by solving discrete algebraic riccati equation iteratively
        P = idare(A,C,G, self.Rww,self.Rvv, self.Rwv, P0=self.P)

        Re = C@P@C.T+self.Rvv
        Kx = P@C.T@np.linalg.inv(Re)
        Kw = (self.Rwv) @ np.linalg.inv(Re)
        
        self.P = P-Kx@Re@Kx.T
        
        self.Re, self.Kx, self.Kw = Re, Kx, Kw

    def update(self, u, y):
        """Kalman filter time update step."""
        
        # Predict
        x_pred = self.A @ self.x + self.B @ u
        
        y_pred = self.C @ x_pred

        # Innovation
        e = y - y_pred
        
        # update
        self.x = x_pred+self.Kx@e
        self.w = self.Kw @ e
        
    def one_step(self,u, y):
        self.update(u,y)
        x_k_p1 = self.A@self.x+self.B@u+self.G@self.w
        return x_k_p1
        
class DynamicKalmanFilter:
    def __init__(self, A, B, C, G, Rww, Rvv, Rwv, P0, x0):
        #model matrices for discrete time model
        self.A = A
        self.B = B
        self.C = C
        self.G = G

        self.Q = Rww    # process noise (Rww)
        self.R = Rvv    # measurement noise (Rvv)
        self.S = Rwv    # noise covariance

        self.x = x0.reshape(-1, 1)
        self.P = P0

    def predict(self, u):
        """Time update (prediction)."""
        u = u.reshape(-1, 1)

        # x^-_k = A x_{k-1} + B u_{k-1}
        self.x = self.A @ self.x + self.B @ u

        # P^-_k = A P_{k-1} A^T + G Q G^T
        self.P = self.A @ self.P @ self.A.T + self.G @ self.Q @ self.G.T

    def update(self, y):
        """Measurement update (correction)."""
        y = y.reshape(-1, 1)

        # Innovation covariance
        S = self.C @ self.P @ self.C.T + self.R  # this is your Re

        # Kalman gain
        K = self.P @ self.C.T @ np.linalg.inv(S)

        # Innovation
        e = y - self.C @ self.x

        # State update
        self.x = self.x + K @ e

        # Covariance update (simple form)
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.C) @ self.P

    def one_step(self, u, y):
        self.predict(u)
        self.update(y)
        return self.x
    
    
if __name__ == '__main__':    
    #Parameters
    a1 = 1.2272 # [cm2] Area of outlet pipe 1
    a2 = 1.2272
    a3 = 1.2272
    a4 = 1.2272

    A1 = 380.1327 #[cm2] Cross sectional area of tank 1
    A2 = 380.1327
    A3 = 380.1327
    A4 = 380.1327

    gamma1 = 0.58 # Flow distribution constant. Valve 1
    gamma2 = 0.72 # Flow distribution constant. Valve 2

    g = 981 #[cm/s2] The acceleration of gravity
    rho = 1.00 #[g/cm3] Density of water

    p = np.array([a1,a2,a3,a4, A1,A2,A3,A4, gamma1,gamma2, g, rho])
    
    # Initial liquid levels [cm]
    h10, h20, h30, h40 = 0.0, 0.0, 0.0, 0.0
    # Convert levels to mass [g] => m = rho * A * h
    m10 = rho * A1 * h10
    m20 = rho * A2 * h20
    m30 = rho * A3 * h30
    m40 = rho * A4 * h40

    x0 = np.array([m10, m20, m30, m40])  # Initial states
    dt = 10
    ts = np.arange(0,30*60+1,dt)
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

    x, y, z, T_all, X_all, H_all = sim22(ts, x0, u, d, p, noise_level=5, plot=False)

    #Noise matrices
    G = np.eye(4)
    Rww = np.eye(4)*5
    Rvv = np.eye(2)*1e-6

    Rwv = np.zeros((4,2)) 
    Rvw = Rwv.T            # 2x4 cross covariance 

    staticKF = StaticKalmanFilter(A,Bu, C ,G,Rww,Rvv,Rwv,P0=np.eye(4), x0=x0)

    # #KF predictions
    predictions = []
    for uk, yk in zip(u.T,y.T):
        uk = uk[:,None]
        yk = yk[:2,None]
        
        staticKF.one_step(uk ,y=yk)
        predictions.append(staticKF.x)
    predictions = np.array(predictions).squeeze()

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Static KF')
    for i in range(4):
        ax = axs[i//2, i%2]     # select subplot

        ax.plot(T_all, X_all[:,i], label='continuous', color='orange')
        ax.plot(ts, x[i,:], 'x', label='discrete', color='green')
        ax.plot(ts, predictions[:,i], 'r.', label='KF predictions')

        ax.set_title(f"Tank {i+1}")
        ax.set_ylim(0, None)
        ax.grid(True)
        
    # Only one legend for all subplots:
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
        
    error = []
    for i in range(len(ts)):
        error.append(np.mean(np.abs(x.T[i:]-predictions[i:]), axis=0))
    plt.figure()
    plt.title('Mean absolute error, for each tank, Static KF')
    plt.plot( error)
    plt.show()

    dynamicKF = DynamicKalmanFilter(A,Bu, C ,G,Rww,Rvv,Rwv,P0=np.eye(4), x0=x0)

    #KF predictions
    predictions = []
    for uk, yk in zip(u.T,y.T):
        uk = uk[:,None]
        yk = yk[:2,None]
        
        dynamicKF.one_step(uk ,y=yk)
        predictions.append(dynamicKF.x)
    predictions = np.array(predictions).squeeze()

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Dynamic KF')
    for i in range(4):
        ax = axs[i//2, i%2]     # select subplot

        ax.plot(T_all, X_all[:,i], label='continuous', color='orange')
        ax.plot(ts, x[i,:], 'x', label='discrete', color='green')
        ax.plot(ts, predictions[:,i], 'r.', label='KF predictions')

        ax.set_title(f"Tank {i+1}")
        ax.set_ylim(0, None)
        ax.grid(True)
        
    # Only one legend for all subplots:
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
        
    error = []
    for i in range(len(ts)):
        error.append(np.mean(np.abs(x.T[i:]-predictions[i:]), axis=0))
    plt.figure()
    plt.title('Mean absolute error, for each tank, Dynamic KF')
    plt.plot( error)
    plt.show()

