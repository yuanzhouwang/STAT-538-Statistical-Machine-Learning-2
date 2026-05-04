import numpy as np
import matplotlib.pyplot as plt
import time
import random

random.seed(20260128)

# Generate Data
d = 50
n = 100
lam = 1
w_opt = np.full(d, 1.0).reshape(-1, 1)
x = np.random.randn(n, d)

z = x @ w_opt

p = 1 / (1 + np.exp(-z))

print(p.min(), p.max())

u = np.random.rand(n, 1)
y = np.where(u < p, 1, -1).reshape(-1)

L = 100     # Smoothness parameter
alpha = 1   # Strong convexity parameter
eta = 1/L

# Gradient Descent
w = np.full((d, 1), 10.0)
T = 100
f_hist = []

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def objective(w, X, y, lam):
    """
    :y: vector (n,)
    :Xw: vector (n,1)
    
    Return:
    objective function evaluated with w
    """
    Xw = (X @ w).reshape(-1)
    yz = y * Xw
    loss = np.mean(np.log1p(np.exp(-yz)))   # Logistic loss
    reg = 0.5 * lam * float(w.T @ w)        # L2 Regularization
    return loss + reg

def grad(w, X, y, lam):
    """
    :y: vector (n,)
    :Xw: vector (n,1)
    
    Return:
    :g: gradient vector (d,1)
    """
    Xw = (X @ w).reshape(-1)          # (n,)
    yz = y * Xw                       # (n,)
    s = 1 / (1 + np.exp(yz))          # sigma(-yz) = 1/(1+exp(yz))
    # vector inside X^T: y * sigma(-yz)
    v = (y * s).reshape(-1, 1)        # (n,1)
    g = -(X.T @ v) / n + lam * w      # (d,1)
    return g

# Classical Gradient Descent Loop
for t in range(T):
    f_hist.append(objective(w, x, y, lam))
    g = grad(w, x, y, lam)
    w = w - eta * g
print(w)

# Plot: semi-log of function values
plt.figure()
plt.semilogy(range(1, T+1), f_hist)
plt.xlabel("Iteration")
plt.ylabel("f(w_t)")
plt.title("GD: objective vs iteration (semi-log)")
plt.show()

# Nesterov's Accelerated GD
gamma = (np.sqrt(L / alpha ) - 1) / (np.sqrt(L / alpha) + 1)
f_hist = []
w_prev = w.copy()
for t in range(T):
    f_hist.append(objective(w, x, y, lam))
    w_adj = gamma * (w - w_prev)
    g = grad(w + w_adj, x, y, lam)
    
    w_prev = w                  # Update w_{t-1}    
    w = w - eta * g + w_adj     # Update w_t

# Plot: semi-log of function values
plt.figure()
plt.semilogy(range(1, T+1), f_hist)
plt.xlabel("Iteration")
plt.ylabel("f(w_t)")
plt.title("AGD: objective vs iteration (semi-log)")
plt.show()


