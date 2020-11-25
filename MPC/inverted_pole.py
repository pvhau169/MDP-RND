import matplotlib.pyplot as plt
import numpy as np
import math
import time
import cvxpy
import gym
import matplotlib.pyplot as plt

print("cvxpy version:", cvxpy.__version__)

#cartpole parameters follow cartPole-V0 of gymAI environment
l_bar = 0.5 
M = 1.0  
m = 0.1  
g = 9.8  

Q = np.diag([0.0, 1.0, 1.0, 0.0])
R = np.diag([0.01])
nx = 4   # number of state
nu = 1   # number of input
T = 40  # Horizon length
delta_t = 0.01   # environment tau

animation = True

env = gym.make('CartPole-v0')

threshold = 15
env.env.x_threshold = threshold
env.env.tau = delta_t

def showPlots(arrays):
    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    columns = 2
    rows = 2
    ax = []
    titles = ["x", "x_dot", "theta", "theta_dot"]
    for i in range(columns*rows):
        # img = np.random.randint(10, size=(h,w))
        # fig.add_subplot(rows, columns, i)
        # plt.imshow(img)
        ax.append( fig.add_subplot(rows, columns, i+1) )
        ax[-1].set_title(titles[i])
        print(arrays[i])
        ax[i].scatter(range(len(arrays[i])), arrays[i])
        # plt.plot(arrays[i], range(len(arrays[i])))
    plt.show()

def main():
    x0 = np.array([
        [3.0],
        [1.0],
        [1.2],
        [0.0]
    ])

    x = np.copy(x0)
    print(x.shape)


    print(type(x))
    print(x)



    env.reset()
    env.env.state = x0
    # env.tau = delta_t
    actions = []
    action = 0

    old_x_s =[]
    x_ds = []
    old_theta_s = []
    theta_ds = []
    for i in range(1000):
        env.render()
        observation, reward, done, info = env.step(action)

        x = np.copy(observation)
        # print(type(x))
        # print(x)
        # print(x.shape)
        try:
            old_x, x_dot, old_theta, theta_dot, force = mpcControl(x)
        except:
            showPlots([old_x_s, x_ds, old_theta_s, theta_ds])
            break
        old_x_s.append(old_x[0])
        x_ds.append(x_dot[0])
        old_theta_s.append(old_theta[0])
        theta_ds.append(theta_dot[0])
        # force = ou[0]
        actions.append(force)
        env.env.force_mag = abs(force)
        # env.env.force_mag = min(abs(force), 10)
        
        if force<0: action = 0
        else: action = 1
    showPlots([old_x_s, x_ds, old_theta_s, theta_ds])
    # print(actions)

def simulation(x, u):

    A, B = getAB()

    x = np.dot(A, x) + np.dot(B, u)

    return x


def mpcControl(x0):

    x = cvxpy.Variable((nx, T + 1))
    u = cvxpy.Variable((nu, T))

    A, B = getAB()

    cost = 0.0
    constr = []
    for t in range(T):
        cost += cvxpy.quad_form(x[:, t + 1], Q)
        cost += cvxpy.quad_form(u[:, t], R)
        constr += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]
        
        constr += [x[0, t + 1]<=threshold]
        constr += [x[0, t + 1]>=-threshold]
        # constr += [u[0, t] <=100]
        # constr += [u[0, t] >=-100]

    constr += [x[:, 0] == x0[:, 0]]
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)


    prob.solve(verbose=False)


    if prob.status == cvxpy.OPTIMAL:
        old_x = getArrayMatrix(x.value[0, :])
        x_dot = getArrayMatrix(x.value[1, :])
        old_theta = getArrayMatrix(x.value[2, :])
        theta_dot = getArrayMatrix(x.value[3, :])

        old_u = getArrayMatrix(u.value[0, :])

    return old_x, x_dot, old_theta, theta_dot, old_u[0]


def getArrayMatrix(x):

    return np.array(x).flatten()


def getAB():

    #design matrix A and B
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * (M + m) / (l_bar * M), 0.0]
    ])


    A = np.eye(nx) + delta_t * A


    B = np.array([
        [0.0],
        [1.0 / M],
        [0.0],
        [-1.0 / (l_bar * M)]
    ])

    B = delta_t * B

    return A, B


def flatten(a):
    return np.array(a).flatten()



if __name__ == '__main__':
    main()