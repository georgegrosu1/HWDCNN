import os
import cv2
import json
import random
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from pathlib import Path
from models.deblurring_model import build_deblurring_model, build_unet_model, build_deblurring_second_model, build_autoencoder_model
from processing import TFImageGenerator


def seed_everything(seed=42):
    random.seed(seed)
    cv2.setRNGSeed(seed)
    # cv2.ocl.setUseOpenCL(False)
    # cv2.setNumThreads(1)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.RandomState(seed=42)
    tf.random.set_seed(seed)


def get_abs_path(relative_path) -> Path:
    root_path = Path(__file__).resolve().parent
    final_path = Path(str(root_path) + f'{relative_path}')
    return final_path


def get_saving_model_path(configs, model_name: str):
    save_dir = get_abs_path(configs['train_cfg']['model_save_path']) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_name + '_epoch{epoch:02d}_vloss{val_loss:.2f}.hdf5'
    return save_dir / model_name


def get_x_y_paths(configs, mode='train'):
    x_path = get_abs_path(configs['train_cfg']['dataset_root']) / mode / 'x_set_17_std-2p4'
    y_path = get_abs_path(configs['train_cfg']['dataset_root']) / mode / 'y_set'

    return x_path, y_path


def train_dcnn(json_cfg, model_name, use_pretrained=False):

    x_train_path, y_train_path = get_x_y_paths(json_cfg, 'train')
    x_val_path, y_val_path = get_x_y_paths(json_cfg, 'test')

    train_gen = TFImageGenerator(x_train_path, y_train_path,
                                 batch_size=json_cfg['train_cfg']['batch_size'],
                                 x_shape=json_cfg['train_cfg']['x_shape'][:-1],
                                 y_shape=json_cfg['train_cfg']['y_shape'][:-1],
                                 normalize=None,
                                 standardize=False)

    val_gen = TFImageGenerator(x_val_path, y_val_path,
                               batch_size=json_cfg['train_cfg']['batch_size'],
                               x_shape=json_cfg['train_cfg']['x_shape'][:-1],
                               y_shape=json_cfg['train_cfg']['y_shape'][:-1],
                               normalize=None,
                               standardize=False)

    if use_pretrained:
        pretrained_path = get_abs_path(json_cfg['train_cfg']['pretrained_model_path'])
        model = build_deblurring_second_model(json_cfg['train_cfg']['x_shape'],
                                              batch_size=json_cfg['train_cfg']['batch_size'],
                                              pretrained_path=pretrained_path)
    else:
        model = build_deblurring_second_model(json_cfg['train_cfg']['x_shape'],
                                              batch_size=json_cfg['train_cfg']['batch_size'])

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.0000001)

    tf_board = tf.keras.callbacks.TensorBoard(log_dir=f'logdir/{model_name}',
                                              histogram_freq=0, write_graph=True, write_images=True)
    checkpoint_filepath = get_saving_model_path(json_cfg, model_name)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='mse',
        mode='min',
        save_freq='epoch',
        save_best_only=False)

    model.fit(train_gen,
              validation_data=val_gen,
              epochs=json_cfg['train_cfg']['epochs'],
              callbacks=[model_checkpoint_callback, tf_board, reduce_lr])


def init_training(config_path, model_name):
    abs_cfg_path = get_abs_path(config_path)
    with open(abs_cfg_path, 'r') as cfg_file:
        cfg_info = json.load(cfg_file)

    train_dcnn(cfg_info, model_name)


def main():

    seed_everything()

    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)

    args_parser = argparse.ArgumentParser(description='Training script for WDCNN')
    args_parser.add_argument('--config_path', '-c', type=str, help='Path to config file',
                             default=r'/configs/dcnn_train.json')
    args_parser.add_argument('--model_name', '-n', type=str, help='Path to model',
                             default=r'x_set_17_std-2p4_dcnn_256')
    args = args_parser.parse_args()

    init_training(args.config_path, args.model_name)


if __name__ == '__main__':
    main()
