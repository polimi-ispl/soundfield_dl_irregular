import numpy as np
import tensorflow as tf

c_complex = 343
pi_complex = np.pi

# Soundfield params (this is not right place)
nfft = 256  # Number of fft points
d = 0.063  # Spacing between sensors
c = 343  # sound speed at 20 degrees
f_s = c / (2 * d)
s_r = 2 * f_s  # Sampling rate
f_axis = np.arange(0, s_r / 2 + s_r / nfft, s_r / nfft)
wc = 2 * np.pi * f_axis
d = 0.063  # Spacing between sensors
M = 64

# We need to compute the following data in order to apply green function while training
Nx = 64
Ny = 64
x_range_listening = np.array([-2, 2])
y_range_listening = np.array([2, 4])
x = np.linspace(x_range_listening[0], x_range_listening[1], Nx)
y = np.linspace(y_range_listening[0], y_range_listening[1], Ny)
[X, Y] = np.meshgrid(x, y)
x_m = np.array([np.arange((-d * M / 2) + d / 2, (d * M / 2), d), np.zeros(M) + 1])
x_u = np.array([X, Y])
x_u_tile = np.tile(np.expand_dims(x_u, axis=1), (1, M, 1, 1))
x_m_tile = np.tile(np.expand_dims(np.expand_dims(x_m, axis=2), axis=3), (1, 1, Nx, Ny))
dist = np.linalg.norm(x_m_tile - x_u_tile, axis=0)
dist = tf.convert_to_tensor(dist, dtype=tf.complex128)
wc = tf.convert_to_tensor(wc, dtype=tf.complex128)
