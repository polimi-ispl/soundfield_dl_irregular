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
    parser.add_argument('--directivity_pattern', type=str, help='Name to identify training on tensorboard', default='point')
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
    c = 343  # sound speed at 20 degrees Celsius

    # Nyquist frequency
    nfft = 256  # Number of fft points

    f_s = c / (2 * d)
    s_r = 2 * f_s  # Sampling rate
    f_axis = np.arange(0, s_r / 2 + s_r / nfft, s_r / nfft)
    wc_axis = 2. * np.pi * f_axis

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

        # Green function matrix (needed for pressure matching)
        x_u_tile = np.tile(np.expand_dims(x_u, axis=1), (1, M, 1, 1))
        x_m_tile = np.tile(np.expand_dims(np.expand_dims(x_m, axis=2), axis=3), (1, 1, Nx, Ny))
        dist = np.linalg.norm(x_m_tile - x_u_tile, axis=0)
        G = np.zeros(shape=(Nx, Ny, M, int(nfft / 2 + 1)), dtype=complex)
        for n_f in range(int(nfft / 2 + 1)):
            G[:, :, :, n_f] = np.transpose(np.exp(-1j * (wc_axis[n_f] / c) * dist) / (4 * np.pi * dist), (1, 2, 0))

        # Select Control Points
        control_mic_axis = 16
        y_idx = np.around(np.linspace(0, Nx - 1, control_mic_axis)).astype(int)
        y_idx_ = np.zeros(control_mic_axis ** 2, dtype=np.int)
        for i in range(control_mic_axis ** 2):
            y_idx_[i] = y_idx[i // control_mic_axis]
        x_idx_ = np.tile(y_idx, control_mic_axis)

        # Temp--> plot as test
        plt.figure(), plt.plot(x_u[0], x_u[1], 'g*'), plt.plot(x_u[0, y_idx_, x_idx_], x_u[1, y_idx_, x_idx_],
                                                               'r*'), plt.show()
        print(' Number of control points: '+str(len(x_idx_)))

        N_sources = P_gt.shape[0]
        """
        n_f = 48
        n_s = 2071
        number_missing_mics = 32
        """
        for number_missing_mics in [16, 32, 48]:
            print('Number of missing loudspeakers '+str(number_missing_mics))
            P_est_hole_comp_pm = np.zeros_like(P_gt)
            NMSE_method_holes_pm = np.zeros(len(params_wideband.wc))
            SSIM_method_holes_pm = np.zeros(len(params_wideband.wc))


            mask = np.load('/nas/home/lcomanducci/'
                           'soundfield_synthesis/'
                           'dataset/masks/mask_missing_loudspeakers_'+str(number_missing_mics)+'_mics_'+str(number_mics)+'_realization_'+str(n_realization)+'.npy')

            # Compute indices corresponding to missing loudspeakers
            idx_missing_mics = np.where(mask == 0)
            """ Now let's perform pressure matching following 
                - Koyama, Shoichi, Keisuke Kimura, and Natsuki Ueno. 
                "Sound field reproduction with weighted mode matching and infinite-dimensional harmonic analysis: 
                An experimental evaluation." 2021 Immersive and 3D Audio: from Architecture to Automotive (I3DA). IEEE, 2021.
                
                - 
                - code partially taken from https://github.com/sh01k/MeshRIR
            """

            # This was for test
            #n_s = 2071  # source_index
            #idx_missing_mics = []
            #number_missing_mics=0


            # extract only control points from Green function matrix
            G_cp = np.transpose(G[y_idx_, x_idx_, :], axes=[2, 0, 1])

            # Remove green function corresponding to missing loudspeakers
            G_cp = np.delete(G_cp, idx_missing_mics, axis=-1)

            # Evaluation Green function matrix
            G_eval = G
            G_eval = np.delete(G_eval, idx_missing_mics, axis=-2)

            # Regularization parameter
            reg = 1e+2

            numSrc = M - number_missing_mics  # secondary sources i.e. the array

            # Pre-compute matrix for pressure matching
            C = np.linalg.inv(np.transpose(G_cp.conj(), (0, 2, 1)) @ G_cp + reg * np.eye(numSrc)) @ np.transpose(G_cp.conj(), (0, 2, 1))

            print('Cycle through sources')
            for n_s in tqdm(range(N_sources)):

                p_gt = P_gt[n_s]  # Ground truth soundfield at current source idx
                des = np.transpose(p_gt[y_idx_, x_idx_, :])

                # Driving functions through pressure matching
                drvPM = np.squeeze(C @ des[:, :, None])

                # Now let's generate soundfield through PM
                p_est_pm = np.zeros_like(p_gt)
                for n_x in range(Nx):
                    for n_y in range(Ny):
                        p_est_pm[n_x, n_y, :] = np.sum(G_eval[n_x, n_y, :, :] * drvPM.T, axis=0)
                P_est_hole_comp_pm[n_s] = p_est_pm

                do_plot = False
                if do_plot:
                    #n_f = 48
                    ssim_temp = ssim_abs_soundfield\
                        (np.expand_dims(P_gt[n_s, :, :, n_f],axis=0),
                         np.expand_dims(P_est_hole_comp_pm[n_s, :, :, n_f],axis=0))
                    plt.figure(figsize=(20, 5))
                    plt.subplot(121)
                    plt.title('SSIM -> '+str(ssim_temp))

                    plt.imshow(np.abs(p_gt[:, :, n_f]), aspect='auto'), plt.colorbar()
                    plt.gca().invert_yaxis()
                    plt.subplot(122)
                    plt.imshow(np.abs(P_est_hole_comp_pm[n_s,:, :, n_f]), aspect='auto'), plt.colorbar()
                    plt.gca().invert_yaxis()
                    plt.show()
                    np.save('soundfield_plot/pm_point.npy', np.real(P_est_hole_comp_pm[n_s, :, :, n_f]))

            """
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": ["Helvetica"],
                'font.size': 20})

            print('PWDR_hole_CNN')
            plt.figure()
            plt.imshow(np.real(P_est_hole_comp_pm[n_s, :, :, n_f]),
                       aspect='auto'), plt.colorbar(), plt.gca().invert_yaxis()
            plt.xlabel('$x[m]$', fontsize=35), plt.ylabel('$y[m]$', fontsize=35)
            plt.tick_params(axis='both', which='major', labelsize=35)
            # plt.title('$\mathrm{PWDR-CNN}^{\circ}$')

            plt.savefig('soundfield_plot/PWDR_hole_pm_' + str(directivity_pattern) + '.pdf', bbox_inches='tight')

            plt.show()
            """
            print('Cycle through frequencies')
            for n_f in tqdm(range(len(params_wideband.wc))):
                #wc = params_wideband.wc[n_f]
                NMSE_method_holes_pm[n_f] = NMSE(P_gt[:, :, :, n_f], P_est_hole_comp_pm[:, :, :, n_f])
                SSIM_method_holes_pm[n_f] = ssim_abs_soundfield(P_gt[:, :, :, n_f], P_est_hole_comp_pm[:, :, :, n_f])

            # Save corresponding results
            np.savez('results/NMSE_wideband_' + directivity_pattern + '_W_' + str(W) + '_loudspeaker_' +
                     str(number_missing_mics)+'_realization_'+str(n_realization) + '_complex_pressure_matching.npz',
                     NMSE_method_holes_pm=NMSE_method_holes_pm)
            np.savez('results/SSIM_wideband_' + directivity_pattern + '_W_' + str(W) + '_loudspeaker_' +
                     str(number_missing_mics) + '_realization_' + str(n_realization) + '_complex_pressure_matching.npz',
                     SSIM_method_holes_pm=SSIM_method_holes_pm)


if __name__ == '__main__':
    main()

