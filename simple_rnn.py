#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RNN basics:
    S_{k} = f( S_{k-1} * W_{rec} + X_{k} * W_{x} )
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm

# Input parameters
n_samples = 20
seq_len = 10


def update_state(xk, sk, wx, wRec):
    return sk * wRec + xk * wx


def forward_states(X, wx, wRec):
    S = np.zeros((X.shape[0], X.shape[1] + 1))
    for k in range(X.shape[1]):
        S[:, k + 1] = update_state(X[:, k], S[:, k], wx, wRec)
    return S


def cost(y, t):
    return ((t - y)**2).sum() / n_samples


def output_gradient(y, t):
    return -2 * (t - y) / n_samples


def backward_gradient(X, S, grad_out, wRec):
    grad_over_time = np.zeros((X.shape[0], X.shape[1] + 1))
    grad_over_time[:, -1] = grad_out
    wx_grad = 0
    wRec_grad = 0
    for k in range(X.shape[1], 0, -1):
        wx_grad += np.sum(grad_over_time[:, k] * X[:, k - 1])
        wRec_grad += np.sum(grad_over_time[:, k] * S[:, k - 1])
        grad_over_time[:, k - 1] = grad_over_time[:, k] * wRec
    return (wx_grad, wRec_grad), grad_over_time


def grad_check(X, t, params, backprop_grads, grad_over_time, eps=1e-7):
    for pidx, _ in enumerate(params):
        tmp_val = params[pidx]
        backprop_grad = backprop_grads[pidx]
        # + eps
        params[pidx] = tmp_val + eps
        fplus = cost(forward_states(X, params[0], params[1])[:, -1], t)
        # - eps
        params[pidx] = tmp_val - eps
        fminus = cost(forward_states(X, params[0], params[1])[:, -1], t)
        # calculate numerical gradient
        numerical_grad = (fplus - fminus) / (2 * eps)
        # reset
        params[pidx] = tmp_val
        # check
        if not np.isclose(numerical_grad, backprop_grad):
            raise ValueError(
                'Numerical gradient of {:.6f} is not close to the backprop gradient of {:.6f}'.
                format(float(numerical_grad), float(backprop_grad)))
    print('No gradient errors found')


def update_rprop(X, t, W, W_delta, W_prev_sign, eta_p, eta_n):
    S = forward_states(X, W[0], W[1])
    grad_out = output_gradient(S[:, -1], t)
    W_grads, _ = backward_gradient(X, S, grad_out, W[1])
    W_sign = np.sign(W_grads)
    for i, _ in enumerate(W):
        if W_sign[i] == W_prev_sign[i]:
            W_delta[i] *= eta_p
        elif W_sign[i] != W_prev_sign[i]:
            W_delta[i] *= eta_n
    return W_delta, W_sign


def optimize_rprop(X, t, W, W_for_plot, eta_p=1.5, eta_n=0.5):
    W_delta = [0.001, 0.001]  # weight update value
    W_sign = [0, 0]  # previous sign of w

    n_iter = 500
    for i in range(n_iter):
        W_delta, W_sign = update_rprop(X, t, W, W_delta, W_sign, eta_p, eta_n)
        for i, _ in enumerate(W):
            W[i] -= W_sign[i] * W_delta[i]
        W_for_plot.append((W[0], W[1]))
    return W


def get_cost_surface(w1_min, w1_max, w2_min, w2_max, n_weights, cost_func):
    w1 = np.linspace(w1_min, w1_max, num=n_weights)
    w2 = np.linspace(w2_min, w2_max, num=n_weights)
    ws1, ws2 = np.meshgrid(w1, w2)
    cost_ws = np.zeros((n_weights, n_weights))
    for i in range(n_weights):
        for j in range(n_weights):
            cost_ws[i, j] = cost_func(ws1[i, j], ws2[i, j])
    return ws1, ws2, cost_ws


def plot_surface(ax, ws1, ws2, cost_ws):
    surf = ax.contourf(
        ws1,
        ws2,
        cost_ws,
        levels=np.logspace(-0.2, 8, 30),
        cmap=cm.pink,
        norm=LogNorm())
    ax.set_xlabel('$w_{in}$', fontsize=15)
    ax.set_ylabel('$w_{rec}$', fontsize=15)
    return surf


def plot_optimization(W_for_plot, cost_func):
    fig = plt.figure(figsize=(10, 4))

    ws1, ws2 = zip(*W_for_plot)
    ax_1 = fig.add_subplot(1, 2, 1)
    ws1_1, ws2_1, cost_ws_1 = get_cost_surface(-3, 3, -3, 3, 100, cost_func)
    surf_1 = plot_surface(ax_1, ws1_1, ws2_1, cost_ws_1 + 1)
    ax_1.plot(ws1, ws2, 'b.')
    ax_1.set_xlim([-3, 3])
    ax_1.set_ylim([-3, 3])

    ax_2 = fig.add_subplot(1, 2, 2)
    ws2_1, ws2_2, cost_ws_2 = get_cost_surface(0, 2, 0, 2, 100, cost_func)
    surf_2 = plot_surface(ax_2, ws2_1, ws2_2, cost_ws_2 + 1)
    ax_2.plot(ws1, ws2, 'b.')
    ax_2.set_xlim([0, 2])
    ax_2.set_ylim([0, 2])

    plt.show()


def main():
    # Input data consists of 20 binary seqences of 10 timestamps
    X = np.zeros((n_samples, seq_len), dtype=np.int)
    for row_idx in range(n_samples):
        X[row_idx, :] = np.around(np.random.rand(seq_len))
    t = np.sum(X, axis=1)
    print('X.shape:', X.shape)
    print('t.shape:', t.shape)

    # forward propagation
    params = [1.2, 1.2]
    S = forward_states(X, params[0], params[1])

    # backward propagation
    out_grad = output_gradient(S[:, -1], t)
    backprop_grads, grad_over_time = backward_gradient(X, S, out_grad,
                                                       params[1])

    # gradient check
    grad_check(X, t, params, backprop_grads, grad_over_time, eps=1e-7)

    # Rprop optimization
    W = [-1.5, 2]
    W_for_plot = [(W[0], W[1])]
    W_opt = optimize_rprop(X, t, W, W_for_plot, eta_p=1.5, eta_n=0.5)
    print('Final weights are: wx = {0}, wRec = {1}'.format(W_opt[0], W_opt[1]))

    # plot the cost surface
    plot_optimization(W_for_plot,
                      lambda w1, w2: cost(forward_states(X, w1, w2)[:, -1], t))

    # The final model is tested on a test sequence.
    test_input = np.asmatrix([[0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1]])
    test_output = forward_states(test_input, W_opt[0], W_opt[1])[:, -1]
    print('Target output: {:d} vs Model output: {:.2f}'.format(
        test_input.sum(), test_output[0]))


if __name__ == '__main__':
    main()
