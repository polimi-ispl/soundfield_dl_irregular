import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_lib import soundfield_generation_wideband as s_g_w
import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm

AUTOTUNE = tf.compat.v1.data.experimental.AUTOTUNE


def main():
    COMPUTE_TRAIN = True
    COMPUTE_TEST = True

    parser = argparse.ArgumentParser(description='Sounfield reconstruction')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=0)
    parser.add_argument('--directivity_pattern', type=str, help='point/dipole', default='point')

    args = parser.parse_args()
    directivity_pattern = args.directivity_pattern
    PLOT = False
    nfft = 256  # Number of fft points
    d = 0.063  # Spacing between sensors
    c = 343  # sound speed at 20 degrees
    # Nyquist frequency
    f_s = c/(2*d)
    s_r = 2*f_s  # Sampling rate
    f_axis = np.arange(0, s_r/2+s_r/nfft, s_r/nfft)
    wc_axis = 2*np.pi*f_axis
    M = 64  # Number of loudspeakers
    W = 23  # Sub array sizes
    Nx = 64
    Ny = 64

    x_m = np.array([np.arange((-d * M / 2) + d / 2, (d * M / 2), d), np.zeros(M) + 1])
    # Sources positions training
    x = np.linspace(-4, 4, 50)
    y = np.linspace(-4, 0, 50)
    [X, Y] = np.meshgrid(x, y)
    src_pos_train = np.array([X.ravel(), Y.ravel()])

    x_test = np.linspace(-4+0.08, 4+0.08, 50)
    y_test = np.linspace(-4, 0, 50)
    [X_test, Y_test] = np.meshgrid(x_test, y_test)
    src_pos_test = np.array([X_test .ravel(), Y_test .ravel()])
    n_src_test = src_pos_test.shape[1]

    n_src_train = src_pos_train.shape[1]
    plt.figure()
    plt.plot(X[:], Y[:], 'r*'), \
    plt.plot(X_test[:], Y_test[:], 'g*'), \
    plt.plot(x_m[0, :], x_m[1, :], 'k*')
    plt.title('Setup')
    plt.show()


    # Train data
    if COMPUTE_TRAIN:
        P_est = np.zeros((n_src_train, Nx, Ny, int(nfft/2)+1), dtype=complex)
        P_gt = np.zeros((n_src_train, Nx, Ny, int(nfft/2)+1), dtype=complex)
        h_ = np.zeros((n_src_train, M, int(nfft/2)+1), dtype=complex)

        for n_s in tqdm(range(n_src_train)):
            SRG_inst = s_g_w.SoundFieldRenderingGeneration(M, W, d, nfft=nfft, Nx = 64, Ny=64)
            x_s = np.expand_dims(src_pos_train[:, n_s], axis=1)
            # Source position
            h, p_est, p_gt = s_g_w.compute_data_full_array(SRG_inst, x_s, wc_axis,  directivity_pattern=directivity_pattern)
            P_est[n_s] = p_est
            P_gt[n_s] = p_gt
            h_[n_s] = h

            shift = SRG_inst.compute_shift(x_s)
            SRG_inst.shift_data(shift)
        np.savez('dataset/data_src_wideband_'+directivity_pattern+'_W_'+str(W)+'_train.npz', P_est=P_est, P_gt=P_gt, h_=h_)

    # Train data
    if COMPUTE_TEST:

        P_est = np.zeros((n_src_test, Nx, Ny, int(nfft/2)+1), dtype=complex)
        P_gt = np.zeros((n_src_test, Nx, Ny, int(nfft/2)+1), dtype=complex)
        h_ = np.zeros((n_src_test, M,  int(nfft/2)+1), dtype=complex)

        for n_s in tqdm(range(n_src_test)):

            SRG_inst = s_g_w.SoundFieldRenderingGeneration(M, W, d, Nx=64, Ny=64)

            x_s = np.expand_dims(src_pos_test[:, n_s], axis=1)  # Source position
            h, p_est, p_gt = s_g_w.compute_data_full_array(SRG_inst, x_s, wc_axis, directivity_pattern=directivity_pattern)
            P_est[n_s] = p_est
            P_gt[n_s] = p_gt
            h_[n_s] = h

            shift = SRG_inst.compute_shift(x_s)
            SRG_inst.shift_data(shift)
        np.savez('dataset/data_src_wideband_'+directivity_pattern+'_W_'+str(W)+'_test.npz',
                 P_est=P_est, P_gt=P_gt, h_=h_)


if __name__ =='__main__':
    main()


