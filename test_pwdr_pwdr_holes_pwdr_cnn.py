import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_ALLOW_GROWTH'] = 'True'
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from train_lib import  params_wideband
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim


def normalize(x):
    min_x = x.min()
    max_x = x.max()
    x_norm = (x - min_x)/(max_x-min_x)
    return x_norm
"""
def soundfield_NMSE(P_gt, P_est):
    # Compute NMSE W.R.T. GT
    # Compute NMSE W.R.T. GT
    # Compute NMSE W.R.T. GT
    examples_gt = P_gt.shape[0]
    examples_est = P_est.shape[0]
    assert examples_est == examples_est
    gt_reshaped = np.reshape(P_gt, newshape=(examples_gt, -1))
    est_reshaped = np.reshape(P_est, newshape=(examples_est, -1))

    abs_diff = np.abs(gt_reshaped - est_reshaped)
    NMSE = 10 * np.log10(np.mean(np.mean(abs_diff, axis=1))/np.mean(np.mean(np.abs(gt_reshaped), axis=1)))
    return NMSE
"""

def NMSE(P_gt, P_hat):
    P_gt = normalize(np.abs(P_gt))
    P_hat = normalize(np.abs(P_hat))

    P_gt = np.reshape(P_gt, newshape=(P_gt.shape[0], -1))
    P_hat = np.reshape(P_hat, newshape=(P_hat.shape[0], -1))

    NMSE_s = np.sum(np.power(np.abs(P_gt-P_hat), 2), axis=1)/np.sum(np.power(np.abs(P_gt), 2))

    NMSE_ = 10*np.log10(np.mean(NMSE_s))

    #NMSE_ = 10*np.log10((NMSE_s))

    return NMSE_


def ssim_abs_soundfield(p, p_est):
    p = np.abs(p)
    p_est = np.abs(p_est)
    ssim_ = np.zeros(p_est.shape[0])
    for i in range(0, p_est.shape[0]):
        ssim_[i] = ssim(p[i], p_est[i], data_range=p_est[i].max() - p_est[i].min())
    return np.mean(ssim_)


def compute_green(h, wc, dist):

    h = h[:, :, :, 0]
    h_complex = tf.cast(h[:, :int(h.shape[1] / 2)], dtype=tf.complex128) + (1j * tf.cast(h[:, int(h.shape[1] / 2):],
                                                                                         dtype=tf.complex128))
    for n_f in range(len(wc)):
        p_est_temp = tf.transpose(
            tf.linalg.matmul(
                tf.transpose(tf.exp(-1j * (wc[n_f] / params_wideband.c_complex)
                                    * dist) / (4 * params_wideband.pi_complex * dist)),
                tf.transpose(h_complex[:, :,  n_f])))
        p_est_cast = tf.expand_dims(p_est_temp, axis=3)

        if n_f == 0:
            p_est = p_est_cast
        else:
            p_est = tf.concat([p_est, p_est_cast], axis=3)
    return p_est

AUTOTUNE = tf.data.experimental.AUTOTUNE

def main():
    parser = argparse.ArgumentParser(description='Sounfield reconstruction')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=0)

    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--log_dir', type=str, help='Tensorboard log directory',
                        default='/nas/home/lcomanducci/soundfield_synthesis/logs/scalars')
    parser.add_argument('--log_name', type=str, help='Name to identify training on tensorboard', default='prova')
    parser.add_argument('--directivity_pattern', type=str, help='Name to identify training on tensorboard', default='dipole')
    parser.add_argument('--number_mics', type=int, help='Number of  microphones', default=64)
    parser.add_argument('--number_missing_mics', type=int, help='Number of missing microphones', default=32)
    parser.add_argument('--n_realization', type=int, help='Realization corresponding to mask index', default=0)


    args = parser.parse_args()
    directivity_pattern = args.directivity_pattern
    number_mics = args.number_mics
    number_missing_mics = args.number_missing_mics
    n_realization = args.n_realization
    #directivity_pattern ='dipole'
    # Soundfield params (this is not right place)
    d = 0.063  # Spacing between sensors
    M = 64
    W = 23

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

    missing_loudspeakers = 22
    N_realizations = 1
    PLOT = False

    for n_r in range(N_realizations):
        print('ciao')
        print('ciaone')
        data = np.load(
            '/nas/home/lcomanducci/soundfield_synthesis/dataset/data_src_wideband_'
            + directivity_pattern + '_W_' + str(W) + '_test.npz')
        print('diosialodoato')
        h_freq = data['h_']
        P_gt = data['P_gt']
        print('diosialodoatodenovo')

        for number_missing_mics in [16, 32, 48]:
            print('Number of missing loudspeakers '+str(number_missing_mics))
            NMSE_method = np.zeros(len(params_wideband.wc))
            NMSE_method_holes = np.zeros(len(params_wideband.wc))
            NMSE_method_holes_NN = np.zeros(len(params_wideband.wc))

            SSIM_method = np.zeros(len(params_wideband.wc))
            SSIM_method_holes = np.zeros(len(params_wideband.wc))
            SSIM_method_holes_NN = np.zeros(len(params_wideband.wc))

            h = np.concatenate([np.real(h_freq), np.imag(h_freq)], axis=1).astype(np.float32)
            h = np.expand_dims(h, axis=3)
            mask = np.load('/nas/home/lcomanducci/'
                           'soundfield_synthesis/'
                           'dataset/masks/mask_missing_loudspeakers_'+str(number_missing_mics)+'_mics_'+str(number_mics)+'_realization_'+str(n_realization)+'.npy')
            mask = np.concatenate([mask, mask])
            mask = np.expand_dims(mask, axis=(0, 2))
            mask = np.tile(mask, (h.shape[0], 1, h.shape[2])).astype('float32')
            mask = np.expand_dims(mask, axis=3)


            network_model = tf.keras.models.load_model('/nas/home/lcomanducci/soundfield_synthesis/models/wideband_'+directivity_pattern+'_W_' +
                                                       str(W)+'_missing_loudspeakers_'+str(number_missing_mics)+'_mics_'+str(number_mics)+'_realization_'+str(n_realization)+'_complex')
            h_hole = np.multiply(h, mask.astype(np.float32))

            # nn-compensated filters
            h_compensated_nn = np.zeros(shape=h_hole.shape)
            for n_s in range(h_hole.shape[0]):
                h_compensated_nn[n_s] = network_model(np.expand_dims(h_hole[n_s],axis=0))
            h_compensated_nn = tf.math.multiply(h_compensated_nn, mask)

            P_est = compute_green(h, params_wideband.wc, dist)
            P_est_hole = compute_green(h_hole, params_wideband.wc, dist)
            P_est_hole_comp = compute_green(h_compensated_nn, params_wideband.wc, dist)

            for n_f in tqdm(range(len(params_wideband.wc))):
                wc = params_wideband.wc[n_f]
                NMSE_method[n_f] = NMSE(P_gt[:, :, :, n_f], P_est[:, :, :, n_f])
                NMSE_method_holes[n_f] = NMSE(P_gt[:, :, :, n_f], P_est_hole[:, :, :, n_f])
                NMSE_method_holes_NN[n_f] = NMSE(P_gt[:, :, :, n_f], P_est_hole_comp[:, :, :, n_f])
                SSIM_method[n_f] = ssim_abs_soundfield(P_gt[:, :, :, n_f], P_est[:, :, :, n_f].numpy())
                SSIM_method_holes[n_f] = ssim_abs_soundfield(P_gt[:, :, :, n_f], P_est_hole[:, :, :, n_f].numpy())
                SSIM_method_holes_NN[n_f] = ssim_abs_soundfield(P_gt[:, :, :, n_f], P_est_hole_comp[:, :, :, n_f].numpy())

            np.savez('results/NMSE_wideband_' + directivity_pattern + '_W_' + str(W) + '_loudspeaker_'+str(number_missing_mics)+'_realization_'+str(n_realization)+'_complex.npz', NMSE_method=NMSE_method, NMSE_method_holes=NMSE_method_holes,
                     NMSE_method_holes_NN=NMSE_method_holes_NN)
            np.savez('results/SSIM_wideband_' + directivity_pattern + '_W_' + str(W) + '_loudspeaker_' + str(number_missing_mics) + '_realization_' + str(n_realization) + '_complex.npz', SSIM_method=SSIM_method,
                     SSIM_method_holes=SSIM_method_holes,
                     SSIM_method_holes_NN=SSIM_method_holes_NN)


if __name__ == '__main__':
    main()

