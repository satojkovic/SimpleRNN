#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RNN basics:
    S_{k} = f( S_{k-1} * W_{rec} + X_{k} * W_{x} )
"""

import numpy as np

# Input parameters
n_samples = 20
seq_len = 10


def update_state(xk, sk, wx, wRec):
    return sk * wRec + xk * wx


def forward_states(X, wx, wRec):
    S = np.zeros((X.shape[0], X.shape[1] + 1))
    for k in range(X.shape[1]):
        S[:, k + 1] = update_state(X[:, k], S[:, k - 1], wx, wRec)
    return S


def cost(y, t):
    return ((t - y)**2).sum() / n_samples



def main():
    pass


if __name__ == '__main__':
    main()
