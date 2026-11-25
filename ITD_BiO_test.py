import numpy as np
import matplotlib.pyplot as plt

d = 64 # Number of state features, size of columns
n = 20 # Total number of labels, number of columns
mu, sigma = 0, 1 # mean and standard deviation
V = np.random.normal(mu, sigma, (d, n)) # Column i, or v_i
norms = np.linalg.norm(V, 2, axis=0) # Norm of each column
V = V / norms # Normalize columns (magnitude 1)
VT = np.transpose(V)
p = np.random.uniform(0, 1, n) # Attacker probabilities

# Generate g dagger, clean gradient of DPO loss at target parameter
xStar = np.zeros(n)
for i in range(n):
    p1 = min(p[i], 1-p[i])
    p2 = max(p[i], 1-p[i])
    xStar[i] = np.random.uniform(p1, p2) # xStar is reachable for given p

gDagger = - np.linalg.matmul(V, xStar)

# Reset V, p, and gDagger for attack and defense
# Option, keep V as is.
def reset(vRST=True):
    global V, VT, p, xStar, gDagger
    # Reset V and p    
    if vRST:
        V = np.random.normal(mu, sigma, (d, n)) # Column i, or v_i
        norms = np.linalg.norm(V, 2, axis=0) # Norm of each column
        V = V / norms # Normalize columns (magnitude 1)
        VT = np.transpose(V)
    p = np.random.uniform(0, 1, n) # Attacker probabilities 
    # Generate g dagger, clean gradient of DPO loss at target parameter
    xStar = np.zeros(n)
    for i in range(n):
        p1 = min(p[i], 1-p[i])
        p2 = max(p[i], 1-p[i])
        xStar[i] = np.random.uniform(p1, p2)
    gDagger = - np.linalg.matmul(V, xStar)

q = np.random.uniform(0, 1, n) # Defender probabilities (redundant but oh well)

# Return D_p or D_q from definitions
def Dp():
    return np.diag(1- 2*p)

# Return D_q from definitions
def Dq():
    return np.diag(1- 2*q)

def customD(x):
    return np.diag(1- 2*x)

# Return u from definitions, 1xd
def u():
    return V @ (p + (Dp() @ q))

def customU(x, y, i, j):
    return V @ (x[i] + (customD(x[i]) @ y[j]))

def normAttack():
    return np.linalg.norm(gDagger + u())**2

def normDefense():
    return np.linalg.norm(u())**2

def ITD_BiO(K, D, alpha, beta, x0, y0, runs, cnt):
    # Initialize attack and defense norms
    attack_norm = np.zeros(K+1)
    defense_norm = np.zeros(K+1)

    for run in range(1, runs + 1):
        if (run % cnt == 0):
            print(run)
            reset(vRST=True)
        else:
            reset(vRST=False)
        # Initialize x, y
        x = np.zeros((K+2, n))
        x[0, :] = np.random.uniform(0, 1, n)
        y = np.zeros((D+1, n))
        
        for k in range(K+1):
            if (k > 0):
                y[0, :] = y[D, :]
            else:
                y[0, :] = np.random.uniform(0, 1, n)
        
            # Lower level gradient descent
            for t in range(D):
                lower_grad = np.linalg.matmul(Dp(), np.linalg.matmul(VT, customU(x, y, k, t)))
                y[t + 1, :] = y[t, :] - alpha*lower_grad
                y[t + 1, :] = np.clip(y[t + 1, :], 0, 1) # Bound probabilities between 0 and 1

            # Iterative update for average of attack_norm
            attack_norm_calc = np.linalg.norm(gDagger + customU(x, y, k, D))
            attack_norm[k] += (1/run) * (attack_norm_calc - attack_norm[k])

            # Backpropagation calculation for upper level
            upper_grad = customD(y[D, :]) @ VT @ (gDagger + customU(x, y, k, D))    
            other_grad = np.zeros(n)
            for t in range(D):
                u_out = customU(x, y, k, t)
                Dp_in = customD(x[k, :])
                W = np.diag(VT @ u_out)        
                grad2 = customD(y[t, :]) @ VT @ V @ customD(x[k, :]) - 2 * W
                grad4 = Dp_in @ VT @ (gDagger + customU(x, y, k, D))       
                grad3 = Dp_in @ VT @ V @ Dp_in               
                small_prod = np.eye(n) - alpha*grad3           
                big_prod = np.eye(n)
                pow_cnt = D - t - 1
                if (pow_cnt > 0):
                    big_prod = np.linalg.matrix_power(small_prod, D-t-1)

                other_grad += grad2 @ big_prod @ grad4

            upper_grad -= alpha * other_grad

            # Upper level gradient descent
            x[k+1, :] = x[k, :] - beta * upper_grad
            x[k+1, :] = np.clip(x[k+1, :], 0, 1) # Bound probabilities between 0 and 1

            # Iterative update for average of defense_norm
            defense_norm_calc = np.linalg.norm(customU(x, y, k+1, D))
            defense_norm[k] += (1/run) * (defense_norm_calc - defense_norm[k])

    print(x[K+1, :])
    print(y[D, :])
    steps = np.arange(1, K + 2, 1)
    fig, ax = plt.subplots()
    ax.set_title('Bilevel norms against ITD steps')
    ax.plot(steps, attack_norm, label='Attack norm', color='blue')
    ax.plot(steps, defense_norm, label='Defense norm', color='red')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Bilevel norms')
    ax.legend()
    plt.savefig("norms.png")
    plt.show()

ITD_BiO(K=200, D=10, alpha=0.01, beta=0.1, x0=p, y0=q, runs=500, cnt=50)