import numpy as np


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def compute_dipole(x_s, wc, c, x_u, y_u):
    [VX, XX] = np.meshgrid(x_s[0], x_u)
    [VY, YY] = np.meshgrid(x_s[1], y_u)
    dist = np.sqrt(np.power(VX - XX, 2) + np.power(VY - YY, 2))
    k_versor = np.array([np.cos(0), np.sin(0)])
    dot_prod1 = (XX - VX) * k_versor[0] + (YY - VY) * k_versor[1]
    p = (1 / (4 * np.pi)) * (1j * wc / c + (1 / dist)) * (dot_prod1 / np.power(dist, 2)) * np.exp(
        -1j * wc / c * dist)
    return p


class SoundFieldRenderingGeneration:
    def __init__(self, M, W, d, Nx=256, Ny=256, Fs=16000, l=1, L=1, directivity_pattern='point', nfft=256):
        self.M = M  # number of loudspeakers/microphones
        self.W = W  # number of microphones
        self.N_subarrays = M - W + 1
        self.c = 343  # Speed of sound
        self.x_range_listening = np.array([-2, 2])
        self.y_range_listening = np.array([2, 4])
        self.Nx = Nx
        self.Ny = Ny
        self.x, self.step_x = np.linspace(self.x_range_listening[0], self.x_range_listening[1], Nx, retstep=True)
        self.y, self.step_y = np.linspace(self.y_range_listening[0], self.y_range_listening[1], Ny, retstep=True)
        [X, Y] = np.meshgrid(self.x, self.y)
        self.x_m = np.array([np.arange((-d * M / 2) + d / 2, (d * M / 2), d), np.zeros(M) + 1])
        self. x_u = np.array([X, Y])
        self.Fs = Fs  # Sampling rate
        self.l = l
        self.L = L
        self.d = d # Loudspeaker spacing
        self.nfft = int(nfft/2 + 1)

        # FRST params
        self.mbar = 0.060
        self.W_rst = 101
        self.qbar = d
        self.sigma = 5*d
        self.N = 128
        self.B = 5

    def compute_shift(self, x_s):
        shift = -x_s
        return shift

    def shift_data(self, shift):
        self.x_m = self.x_m + shift
        self.x_u = self.x_u + np.expand_dims(shift, axis=2)

    def estimate_subarray_centers(self):
        """
        Args:
            x_m, signal at the microphones
            W, subarray width
            N_subarrays, number of subarrays
        Returns:
            alpha_I, steering direction subarrays
        """

        # Subarray centers
        x_sub_center = np.zeros((2, self.N_subarrays))

        # Plane-wave component direction reproduced by each sub array
        alpha_i = np.zeros(self.N_subarrays)

        for n_sub in range(self.N_subarrays):
            c_idx_sub = n_sub + (self.W // 2)
            x_sub_center[:, n_sub] = self.x_m[:, c_idx_sub]
            alpha_i[n_sub] = np.arctan2(x_sub_center[1, n_sub], x_sub_center[0, n_sub])
        return alpha_i

    def estimate_directivity_function(self, wc, alpha_i, x_s, shift, directivity_pattern, r=1):
        """
        Args:
            x_m, signal at the microphones
            W, subarray width
            N_subarrays, number of subarrays
            r, radius
        Returns:
            D, directivity function at frequency wc
        """

        # ADD DIPOLE SUPPORT
        # Shift positions
        x_s = x_s + shift

        D = np.zeros(shape=(self.N_subarrays, self.nfft), dtype=complex)
        for n_sub in range(self.N_subarrays):
            x, y = pol2cart(r, alpha_i[n_sub])
            dist = np.linalg.norm(x_s - np.array([x, y]))
            if directivity_pattern == 'point':
                p = np.exp(-1j * (wc / self.c) * dist) / (4 * np.pi * dist)
            if directivity_pattern == 'dipole':
                p = compute_dipole(x_s, wc, self.c, x, y)
            D[n_sub] = r / np.exp(r * 1j * wc / self.c) * p
        return D

    def estimate_travel_time_vector(self, x_s, shift):
        """

        :param N_subarrays:
        :param W:
        :param x_s:
        :param c:
        :param x_m:
        :return:
        """

        # Shift positions
        x_s = x_s + shift

        tau = np.zeros(shape=(self.N_subarrays, self.W))
        for n_sub in range(self.N_subarrays):
            tau[n_sub] = np.sqrt(
                np.power(x_s[0] - self.x_m[0, n_sub:n_sub + self.W], 2) + np.power(x_s[1] - self.x_m[1, n_sub:n_sub + self.W], 2))/ self.c
        tau = tau.T
        return tau

    def estimate_steering_vectors(self, alpha_i, wc):

        a = np.zeros(shape=(self.W, self.N_subarrays, self.nfft), dtype=complex)
        for n_s in range(self.N_subarrays):
            w_si = (wc / self.c) * self.d * np.cos(alpha_i[n_s])
            for idx_row in range(0, self.W):
                w = self.W // 2 - idx_row
                a[idx_row, n_s] = np.exp(1j * w * w_si)
        return a

    def estimate_beamforming_filters(self, a):
        F = np.zeros((self.W, self.N_subarrays, self.nfft), dtype=complex)
        band_factor = np.arange(1, self.nfft+1)/self.nfft
        for n_s in range(self.N_subarrays):
            Gamma = np.sinc(2 * np.pi * np.expand_dims(
                    np.linalg.norm(self.x_m[:, n_s + np.tile(np.expand_dims(np.arange(self.W), axis=1), (1, self.W))] -
                                   self.x_m[:, n_s + np.tile(np.expand_dims(np.arange(self.W), axis=0), (self.W, 1))], axis=0)
                    / self.c,axis=2) * self.Fs * band_factor).astype('complex128')
            for n_f in range(self.nfft):
                F[:, n_s, n_f] = (np.linalg.inv(Gamma[:, :, n_f]) @ a[:, n_s, n_f]) / (np.conj(a[:, n_s, n_f]).T @ np.linalg.inv(Gamma[:, :, n_f]) @ a[:, n_s, n_f])
        return F

    def estimate_driving_coefficients(self, D, F, wc, tau):

        z = 0.5 * np.power(1 - np.cos(2 * np.pi * np.arange(0, self.W) / (self.W - 1)), 2)
        h = np.zeros(shape=(self.M, self.nfft), dtype=complex)
        for m in range(self.M):
            for i in range(np.maximum(0, m - self.W + 1), np.minimum(m + 1, self.M - self.W + 1)):
                h[m] = h[m] + F[m - i, i] * np.exp(-1j * wc * tau[m - i, i]) * D[i] * z[m - i]

        return h

    def estimate_soundfield(self, wc, h):
        x_u_tile = np.tile(np.expand_dims(self.x_u, axis=1), (1, self.M, 1, 1))

        x_m_tile = np.tile(np.expand_dims(np.expand_dims(self.x_m, axis=2), axis=3), (1, 1, self.Nx, self.Ny))

        dist = np.linalg.norm(x_m_tile-x_u_tile, axis=0)
        p_est = np.zeros(shape=(self.Nx, self.Ny, self.nfft), dtype=complex)
        for n_f in range(self.nfft):
            p_est[:, :, n_f] = ((np.exp(-1j*(wc[n_f]/self.c)*dist)/(4*np.pi*dist)).T @ h[:, n_f].T).T

        return p_est

    def estimate_gt_soundfield(self, x_s, wc, shift, directivity_pattern):
        # ADD DIPOLE SUPPORT
        x_s = x_s + shift
        p_gt = np.zeros(shape=(self.Nx, self.Ny, self.nfft), dtype=complex)

        if directivity_pattern == 'point':
            x_s_tile = np.tile(np.expand_dims(x_s, axis=2), (1, self.Nx, self.Ny))
            dist_gt = np.linalg.norm(self.x_u - x_s_tile, axis=0)
            for n_f in range(self.nfft):
                p_gt[:, :, n_f] = np.exp(-1j * (wc[n_f] / self.c) * dist_gt) / (4 * np.pi * dist_gt)

        if directivity_pattern == 'dipole':
            for n_f in range(self.nfft):
                p_gt_temp = compute_dipole(x_s, wc[n_f], self.c, self.x_u[0], self.x_u[1])
                p_gt[:, :, n_f] = np.reshape(p_gt_temp, (self.Nx, self.Ny))
        return p_gt


def compute_data_full_array(SRG_inst, x_s, wc_axis, directivity_pattern):

    # directivity_pattern = 'point', 'dipole'

    shift = SRG_inst.compute_shift(x_s)
    SRG_inst.shift_data(shift)
    alpha_i = SRG_inst.estimate_subarray_centers()
    
    # Estimate Directivity function
    D = SRG_inst.estimate_directivity_function(wc_axis,  alpha_i, x_s, shift, directivity_pattern, r=4)

    tau = SRG_inst.estimate_travel_time_vector(x_s, shift)

    # Estimate Steering vectors
    a = SRG_inst.estimate_steering_vectors(alpha_i, wc_axis)

    # Estimate beamforming filters
    F = SRG_inst.estimate_beamforming_filters(a)

    # Compute driving coefficients for one freq
    h = SRG_inst.estimate_driving_coefficients(D, F, wc_axis, tau)

    # Estimate soundfields
    p_est = SRG_inst.estimate_soundfield(wc_axis, h)

    p_gt = SRG_inst.estimate_gt_soundfield(x_s, wc_axis, shift, directivity_pattern)

    return h, p_est, p_gt


