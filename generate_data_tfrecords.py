import argparse
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

"""
    Converts audio files contained in base_dir/dataset_name/audio into tfrecords contained in base_dir/dataset_name/tfrecords,
    containing the following features:
    audio_track
     

"""

EXAMPLES_PER_RECORD = 50


def main():
    # Arguments parse
    parser = argparse.ArgumentParser(description='Convert audio to TFRecords')
    parser.add_argument('--base_dir', type=str, help="Base Data Directory", default='/nas/home/lcomanducci/soundfield_synthesis/dataset')
    parser.add_argument('--dataset_name', type=str, help="Base Data Directory", default='data_src_wideband_point_W_23_train')
    parser.add_argument('--dataset_path', type=str, help='path to dataset', default='/nas/home/lcomanducci/soundfield_synthesis/dataset/data_src_wideband_point_W_23_train.npz' )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    base_dir = args.base_dir
    dataset_dir = args.dataset_name

    data = np.load(dataset_path)
    P_gt_data = data['P_gt']
    h_data = data['h_']

    N_examples = h_data.shape[0]
    # Create Train TFRECORDS
    N_SHARDS = int(np.ceil(N_examples/EXAMPLES_PER_RECORD))
    tfrecord_dataset_path = os.path.join(base_dir, dataset_dir)

    if not os.path.exists(tfrecord_dataset_path):
        os.makedirs(tfrecord_dataset_path)
    for n_s in tqdm(range(N_SHARDS)):
        tfrecord_name = str(n_s)+'.tfrecord'
        tfrecord_path = os.path.join(tfrecord_dataset_path, tfrecord_name)

        with tf.io.TFRecordWriter(tfrecord_path, options='ZLIB') as out:
            begin = n_s*EXAMPLES_PER_RECORD
            end = n_s*EXAMPLES_PER_RECORD+EXAMPLES_PER_RECORD

            if end > N_examples:
                end = N_examples

            for n_t in range(begin, end):
                # Read data
                P_gt = P_gt_data[n_t]  # Soundfield
                h = h_data[n_t]  # Filters

                feature = {
                    'P_gt_real': tf.train.Feature(float_list=tf.train.FloatList(value=np.real(P_gt.flatten()))),
                    'P_gt_imag': tf.train.Feature(float_list=tf.train.FloatList(value=np.imag(P_gt.flatten()))),

                    'h_real': tf.train.Feature(float_list=tf.train.FloatList(value=np.real(h.flatten()))),
                    'h_imag': tf.train.Feature(float_list=tf.train.FloatList(value=np.imag(h.flatten()))),

                    'P_gt_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=P_gt.shape)),
                    'h_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=h.shape)),

                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                out.write(example.SerializeToString())

    print('TFRecords SAVED')


if __name__== '__main__':
    main()
