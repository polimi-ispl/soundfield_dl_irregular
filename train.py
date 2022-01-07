import numpy as np
import os
import argparse
import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_ALLOW_GROWTH'] = 'True'
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from train_lib import network_utils
from train_lib import train_utils
from train_lib import params_wideband
from tqdm import tqdm


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
        p_est_cast_real = tf.expand_dims(tf.cast(tf.math.real(p_est_temp), dtype=tf.float32), axis=3)
        p_est_cast_imag = tf.expand_dims(tf.cast(tf.math.imag(p_est_temp), dtype=tf.float32), axis=3)

        if n_f == 0:
            p_est_real = p_est_cast_real
            p_est_imag = p_est_cast_imag
        else:
            p_est_real = tf.concat([p_est_real, p_est_cast_real], axis=3)
            p_est_imag = tf.concat([p_est_imag, p_est_cast_imag], axis=3)

    return p_est_real, p_est_imag


AUTOTUNE = tf.data.experimental.AUTOTUNE


def main():

    parser = argparse.ArgumentParser(
        description='Sounfield reconstruction')

    parser.add_argument('--epochs', type=int, help='Number of epochs', default=1000)

    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--log_dir', type=str, help='Tensorboard log directory',
                        default='/nas/home/lcomanducci/soundfield_synthesis/logs/scalars')
    parser.add_argument('--log_name', type=str, help='Name to identify training on tensorboard', default='prova')
    parser.add_argument('--freq', type=float, help='Frequency in Hz', default=1000)
    parser.add_argument('--dataset_path', type=str, help='path to dataset', default='/nas/home/lcomanducci/soundfield_synthesis/dataset/data_src_wideband_point_W_23_train' )
    parser.add_argument('--mask_path', type=str, help='path to mask array', default='/nas/home/lcomanducci/soundfield_synthesis/dataset/masks/mask_missing_loudspeakers_32_mics_64_realization_0.npy')
    parser.add_argument('--saved_model_path', type=str, help='path to mask array', default='/nas/home/lcomanducci/soundfield_synthesis/models/wideband_point_W_23_train.npy' )
    parser.add_argument('--learning_rate', type=float, help='LEarning rate', default=0.0001)


    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    log_dir = args.log_dir
    log_name = args.log_name
    dataset_path = args.dataset_path
    mask_path = args.mask_path
    saved_model_path = args.saved_model_path
    lr = args.learning_rate

    # Tensorboard and logging
    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + log_name)
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Training params
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    epoch_to_plot = 5  # Plot evey epoch_to_plot epochs
    val_perc = 0.2

    N_mics = 64
    filter_shape = int(N_mics * 2)
    nfft = int(129)
    early_stop_patience = 1000


    plt.figure()
    plt.plot(params_wideband.X[:], params_wideband.Y[:], 'r*'), \
    plt.plot(params_wideband.x_m[0, :], params_wideband.x_m[1, :], 'k*')
    plt.title('Setup')
    plt.show()


    # Load tfrecord paths
    tfrecord_paths = tf.io.gfile.glob(dataset_path+'/*.tfrecord')
    random.shuffle(tfrecord_paths)
    n_val_tfrecords = int(val_perc*len(tfrecord_paths))
    tfrecord_paths_val = tfrecord_paths[:n_val_tfrecords]
    tfrecord_paths_train = tfrecord_paths[n_val_tfrecords:]

    # Reshape mask to take into account different frequencies

    def _parse(serialized_example):
        feature = {
            'P_gt_real': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'P_gt_imag': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),

            'h_real': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'h_imag': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),

            'P_gt_shape': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'h_shape': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        }
        example = tf.io.parse_single_example(serialized_example, feature)

        P_gt_real = tf.reshape(example['P_gt_real'], example['P_gt_shape'])
        P_gt_imag = tf.reshape(example['P_gt_imag'], example['P_gt_shape'])

        h_real = tf.reshape(example['h_real'], example['h_shape'])
        h_imag = tf.reshape(example['h_imag'], example['h_shape'])
        h = tf.concat([h_real, h_imag], axis=0)

        # Add image channel to filter matrix
        h = tf.expand_dims(h, axis=2)

        return h, P_gt_real, P_gt_imag


    train_ds = tf.data.TFRecordDataset(tfrecord_paths_train, compression_type='ZLIB', num_parallel_reads=AUTOTUNE)
    train_ds = train_ds.map(_parse, num_parallel_calls=AUTOTUNE)

    val_ds = tf.data.TFRecordDataset(tfrecord_paths_val, compression_type='ZLIB')
    val_ds = val_ds.map(_parse, num_parallel_calls=AUTOTUNE)

    loss_fn = tf.keras.losses.MeanSquaredError()
    metric_fn_train_real = tf.keras.metrics.MeanSquaredError()
    metric_fn_train_imag = tf.keras.metrics.MeanSquaredError()

    metric_fn_val_real = tf.keras.metrics.MeanSquaredError()
    metric_fn_val_imag = tf.keras.metrics.MeanSquaredError()

    mask = np.load(mask_path)
    mask = np.concatenate([mask, mask])
    mask = np.expand_dims(mask, axis=(0, 2))
    mask = np.tile(mask, (batch_size, 1, nfft)).astype('float32')
    mask = tf.convert_to_tensor(np.expand_dims(mask, axis=3))

    # Second training
    network_model_filters = network_utils.filter_compensation_model_wideband_skipped(filter_shape, nfft)
    network_model_filters.summary()


    train_ds_2 = train_ds.shuffle(buffer_size=int(batch_size*2))
    val_ds_2 = val_ds.shuffle(buffer_size=int(batch_size*2))

    train_ds_2 = train_ds_2.batch(batch_size=batch_size)
    val_ds_2 = val_ds_2.batch(batch_size=batch_size)

    train_ds_2 = train_ds_2.prefetch(AUTOTUNE)
    val_ds_2 = val_ds_2.prefetch(AUTOTUNE)

    @tf.function
    def train_step_2(P_gt_real, P_gt_imag, h_, mask_):
        with tf.GradientTape() as network_1_tape, tf.GradientTape() as network_2_tape:
            h_hat = tf.cast(h_, dtype=tf.float32)
            h_hat = tf.math.multiply(h_hat, mask_)
            h_hat = network_model_filters(h_hat, training=True)
            h_hat = tf.math.multiply(h_hat, mask_)
            P_hat_real, P_hat_imag = compute_green(h_hat, params_wideband.wc,  params_wideband.dist)

            loss_value_P = loss_fn(P_gt_real, P_hat_real) + loss_fn(P_gt_imag, P_hat_imag)

        network_model_filters_grads = network_1_tape.gradient(loss_value_P, network_model_filters.trainable_weights)

        optimizer.apply_gradients(zip(network_model_filters_grads, network_model_filters.trainable_weights))

        metric_fn_train_real.update_state(P_gt_real, P_hat_real)
        metric_fn_train_imag.update_state(P_gt_imag, P_hat_imag)

        return loss_value_P


    @tf.function
    def val_step_2(P_gt_real, P_gt_imag, h_, mask_):
        h_hat = tf.cast(h_, dtype=tf.float32)
        h_hat = tf.math.multiply(h_hat, mask_)
        h_hat = network_model_filters(h_hat, training=False)
        h_hat = tf.math.multiply(h_hat, mask_)
        P_hat_real, P_hat_imag = compute_green(h_hat, params_wideband.wc, params_wideband.dist)

        loss_value_P = loss_fn(P_gt_real, P_hat_real) + loss_fn(P_gt_imag, P_hat_imag)

        metric_fn_val_real.update_state(P_gt_real, P_hat_real)
        metric_fn_val_imag.update_state(P_gt_imag, P_hat_imag)

        return loss_value_P, h_hat, P_hat_real, P_hat_imag

    for n_e in tqdm(range(epochs)):
        plot_val = True

        for h, P_real, P_imag in train_ds_2:
            loss_value_P = train_step_2(P_real, P_imag, h, mask[:h.shape[0]])
        train_loss = metric_fn_train_imag.result() + metric_fn_train_real.result()
        # Log to tensorboard
        with summary_writer.as_default():
            tf.summary.scalar('train_loss_P_2', train_loss, step=n_e)
        metric_fn_train_imag.reset_states()
        metric_fn_val_real.reset_states()


        for h, P_real, P_imag in val_ds_2:
            loss_value_P_val, h_hat, P_hat_real, P_hat_imag = val_step_2(P_real, P_imag, h, mask[:h.shape[0]])

        val_loss = metric_fn_val_imag.result() + metric_fn_val_real.result()

        # Every epoch_to_plot epochs plot an example of validation
        if not n_e % epoch_to_plot and plot_val:
            print('Train loss: ' + str(train_loss.numpy()))
            print('Val loss: ' + str(val_loss.numpy()))

            idx_plot = np.random.randint(0, len(params_wideband.wc))

            figure_soundfield = train_utils.est_vs_gt_soundfield(tf.expand_dims(P_hat_real[:, :, :, idx_plot], axis=3), tf.expand_dims(P_real[:, :, :, idx_plot], axis=3))

            with summary_writer.as_default():
                tf.summary.image("soundfield second training", train_utils.plot_to_image(figure_soundfield), step=n_e)

            filtess2 = plt.figure()
            plt.plot(h_hat.numpy()[0, :, :, 0])
            with summary_writer.as_default():
                tf.summary.image("Filters true second", train_utils.plot_to_image(filtess2), step=n_e)

        # Select best model
        if n_e == 0:
            lowest_val_loss = val_loss
            network_model_filters.save(saved_model_path)
            early_stop_counter = 0

        else:
            if val_loss < lowest_val_loss:
                network_model_filters.save(saved_model_path)
                lowest_val_loss = val_loss
                early_stop_counter = 0
            else:
                network_model_filters.save(saved_model_path)
                early_stop_counter = early_stop_counter + 1

        # Early stopping
        if early_stop_counter > early_stop_patience:
            print('Training finished at epoch '+str(n_e))
            break

        # Log to tensorboard
        with summary_writer.as_default():
            tf.summary.scalar('val_loss_P_2', val_loss, step=n_e)
        metric_fn_val_real.reset_states()
        metric_fn_val_imag.reset_states()

if __name__ == '__main__':
    main()

