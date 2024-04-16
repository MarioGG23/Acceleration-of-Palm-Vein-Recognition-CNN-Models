# test_singlenet_palmvein.py
#
# Copyright 2020 by Ruber Hernández-García (linkedin.com/in/ruberhg) and LITRP (www.litrp.cl)
# All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License GPLv3 as published by
# the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  Please see the
# GNU General Public License for more details: https://www.gnu.org/licenses/gpl-3.0.html

# __description__ = "Test Pre-trained CNN singlenet model for palm vein recognition"
# __author__ = "Ruber Hernández-García"
# __copyright__ = "Copyright (C) 2020 Ruber Hernández-García"
# __license__ = "GNU General Public License GPLv3"
# __version__ = "1.0"


from __future__ import absolute_import, division, print_function, unicode_literals
from imgaug import augmenters as iaa
import imageio
import cv2
from PIL import Image
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.utils import to_categorical
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model, load_model
import warnings
from sklearn.model_selection import train_test_split
# Helper libraries
import argparse
import numpy as np
from scipy import ndimage
import math
from matplotlib import pyplot
import sys
import os
import os.path as osp
import glob
import pathlib
import pickle
import random
import skimage
import deepdish as dd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from skimage import transform
from skimage.color import rgb2gray
# Ignore tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras

# TensorFlow
#import tensorflow as tf
#from tensorflow import keras

# Keras


# Library to load and manipulate data

# Manual augmentation

# ----------------------------------------------------------------------------

# Name for the callback output
NAME = 'palmvein-singlenet'

# Ignore tf warnings
# tf.compat.v1.enable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.debugging.set_log_device_placement(True)

theSEED = 232323
config = tf.compat.v1.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # gpu_rate # TODO

tf.executing_eagerly()
graph = tf.Graph()
graph.as_default()

session = tf.compat.v1.Session(graph=graph, config=config)
session.as_default()

MAIN_PATH = pathlib.Path(__file__).parent.absolute()

# IMG_SIZE = 128
IMG_SIZE = 64

# ----------------------------------------------------------------------------


def read_data(datadir, dataset, dataset_variant, nclasses, hands, nsamples, ftsamples=1, mode="all"):
    print('read_data: ', dataset)
    db = {
        "CASIA/850": {"percent": 0.8, "ext": "jpg"},
        "CASIA/940": {"percent": 0.8, "ext": "jpg"},
        "FYO": {"percent": 0.0, "ext": "png"},
        "IIT": {"percent": 0.5, "ext": "png"},
        "POLYU": {"percent": 0.6, "ext": "jpg"},
        "PUT": {"percent": 0.8, "ext": "jpg"},
        "TONGJI": {"percent": 0.8, "ext": "png"},
        "VERA": {"percent": 0.8, "ext": "png"},
        "NS-PVDB": {"percent": 0.8, "ext": "png"},
        "Synthetic-sPVDB": {"percent": 0.8, "ext": "png"}
    }

    CLASSES = []
    if len(hands) == 0:
        for i in range(1, nclasses+1):
            c = str(i).zfill(5)
            CLASSES.append(c)
    else:
        for i in range(1, nclasses+1):
            for h in hands:
                c = str(i).zfill(3)
                CLASSES.append(osp.join(c, h))

    nclasses = len(CLASSES)
    ft_data = []
    test_data = []
    for c in CLASSES:
        path = osp.join(datadir, dataset, dataset_variant, c)
        class_num = CLASSES.index(c)
        if dataset in db:
            percent, ext = db[dataset]["percent"], db[dataset]["ext"]
            split_data(nsamples, ftsamples, path, percent, ext,
                       class_num, mode, ft_data, test_data)

    print('ft_data: ', len(ft_data))
    print('test_data: ', len(test_data))
    print('nclasses: ', nclasses)
    print('end_read_data')
    # exit()
    return ft_data, test_data, nclasses


def split_data(nsamples, ftsamples, path, percent, ext, class_num, mode, ft_data, test_data):
    if mode == "ft+aug":
        for s in range(int(nsamples*percent)+1, int(nsamples*percent)+ftsamples+1):
            # augmented data for fine-tuning
            images = glob.glob(osp.join(path, '*_{:02d}_*.'.format(s)+ext))
            for img in images:
                img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                ft_data.append([new_array, class_num])

        for s in range(int(nsamples*percent)+ftsamples+1, nsamples+1):
            # original data for testing
            images = glob.glob(
                osp.join(path, '*_{:02d}*rot0_cx0_cy0.'.format(s)+ext))
            for img in images:
                img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                test_data.append([new_array, class_num])
    elif mode == "ft":
        for s in range(int(nsamples*percent)+1, int(nsamples*percent)+ftsamples+1):
            # original data for fine-tuning
            images = glob.glob(
                osp.join(path, '*_{:02d}*rot0_cx0_cy0.'.format(s)+ext))
            for img in images:
                img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                ft_data.append([new_array, class_num])

        for s in range(int(nsamples*percent)+ftsamples+1, nsamples+1):
            # original data for testing
            images = glob.glob(
                osp.join(path, '*_{:02d}*rot0_cx0_cy0.'.format(s)+ext))
            for img in images:
                img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                test_data.append([new_array, class_num])
    else:
        info = []
        for s in range(int(nsamples*percent)+1, nsamples+1):
            # original data for testing
            images = glob.glob(
                osp.join(path, '*_{:02d}*rot0_cx0_cy0.'.format(s)+ext))
            info.append(images)
        print("=================================================================")
        print(info)
            # for img in images:
            #     img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            #     new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

            #     test_data.append([new_array, class_num])


def preprocessing_data(ft_data, test_data, nclasses, mode="all"):
    if mode == "all":
        test_samples = []
        test_labels = []

        for s, l in test_data:
            test_samples.append(s)
            test_labels.append(l)

        test_samples = np.array(
            test_samples).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        test_labels = np.array(test_labels)

        meanTest = np.mean(test_samples, axis=0)
        test_samples = test_samples-meanTest
        test_samples = test_samples/255

        test_samples = test_samples.reshape(
            test_samples.shape[0], IMG_SIZE, IMG_SIZE, 1)
        test_labels = to_categorical(test_labels, nclasses)

    else:
        ft_samples = []
        test_samples = []
        ft_labels = []
        test_labels = []

        for s, l in ft_data:
            ft_samples.append(s)
            ft_labels.append(l)

        for s, l in test_data:
            test_samples.append(s)
            test_labels.append(l)

        ft_samples = np.array(ft_samples).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        ft_labels = np.array(ft_labels)

        test_samples = np.array(
            test_samples).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        test_labels = np.array(test_labels)

        meanFt = np.mean(ft_samples, axis=0)
        meanTest = np.mean(test_samples, axis=0)

        ft_samples = ft_samples-meanFt
        ft_samples = ft_samples/255

        test_samples = test_samples-meanTest
        test_samples = test_samples/255

        ft_samples = ft_samples.reshape(
            ft_samples.shape[0], IMG_SIZE, IMG_SIZE, 1)
        test_samples = test_samples.reshape(
            test_samples.shape[0], IMG_SIZE, IMG_SIZE, 1)

    return ft_samples, test_samples, ft_labels, test_labels

# ----------------------------------------------------------------------------

# Load pre-trained CNN model


def load(initnet, include_top=False, encode_layer=None, verbose=1):

    if verbose > 0:
        print("++++++++++++++++++++++++++++++++++++++++++++")
        print("Loading model from: " + initnet)

    model = load_model(initnet)
    if not include_top:
        if encode_layer is None:
            model_encode = Model(model.input, model.layers[-5].input)
        else:
            out_layer = model.get_layer(encode_layer).output
            model_encode = Model(model.input, out_layer)

        if verbose > 0:
            print("Done!")
            model_encode.summary()

        return model_encode

    if verbose > 0:
        print("Done!")
        model.summary()

    return model

# ----------------------------------------------------------------------------

# Encode data by using model embeddings


def encodeData(data, labels, model_encode, batch_size=32):
    all_gt_labs = labels
    all_feats = []

    all_feats = model_encode.predict(data, batch_size=batch_size)

    return all_feats, all_gt_labs

# ----------------------------------------------------------------------------

# Evaluate kNN classifier


def evaluate_kNN(model, nclasses, ftsamples, ft_data, ft_gt_labels, test_data, test_gt_labels, metric="L2", mode="ft", knn=7):
    # Test the CNN model
    testdir = os.path.join(experdir, "results")
    os.makedirs(testdir, exist_ok=True)
    outpath = os.path.join(
        testdir, "gallery_N{:05d}_{}{}_knn.h5".format(nclasses, mode, ftsamples))

    # Obtain gallery encodings
    if not os.path.exists(outpath):
        print("Encoding gallery samples...")
        all_feats_gallery, all_gt_labs_gallery = encodeData(
            ft_data, ft_gt_labels, model)

        # Save gallery encodings
        with open(outpath, 'wb') as output:
            pickle.dump([all_feats_gallery, all_gt_labs_gallery],
                        output, pickle.HIGHEST_PROTOCOL)
        print("Data saved to: " + outpath)
    else:
        exper = pickle.load(open(outpath, "rb"))
        all_feats_gallery = np.asarray(exper[0])
        all_gt_labs_gallery = np.asarray(exper[1])

    exper_gallery = {}
    exper_gallery["feats"] = all_feats_gallery
    exper_gallery["gtlabs"] = all_gt_labs_gallery

    embpath = os.path.join(
        testdir, "probe_N{:05}_{}{}_knn.h5".format(nclasses, mode, ftsamples))

    cm = []
    acc = []
    for n in knn:
        outpath = os.path.join(testdir, "results_N{:05}_{}{}_knn_{:01}_{}.h5".format(
            nclasses, mode, ftsamples, n, metric))
        

        # Obtain probe encodings
        if not os.path.exists(outpath) and not os.path.exists(embpath):
            print("Encoding probe samples...")
            all_feats_probe, all_gt_labs_probe = encodeData(
                test_data, test_gt_labels, model)

            # Save probe encodings
            with open(embpath, 'wb') as output:
                pickle.dump([all_feats_probe, all_gt_labs_probe],
                            output, pickle.HIGHEST_PROTOCOL)
            print("Data saved to: " + embpath)

        elif os.path.exists(outpath):
            exper_results = dd.io.load(outpath)

            all_gt_labs_probe = np.asarray(exper_results["gtlabs"])
            all_pred_labs = np.asarray(exper_results["predlabs"])

            # Compute accuracy from Confusion Matrix
            CM = confusion_matrix(all_gt_labs_probe, all_pred_labs)
            Acc = CM.diagonal().sum() / len(all_gt_labs_probe)
            cm.append(CM)
            acc.append(Acc)

        else:
            print("Different K, use previous activations and recompute kNN results.")
            exper = pickle.load(open(embpath, "rb"))
            all_feats_probe = np.asarray(exper[0])
            all_gt_labs_probe = np.asarray(exper[1])

        if not os.path.exists(outpath):
            exper_probe = {}
            exper_probe["feats"] = all_feats_probe
            exper_probe["gtlabs"] = all_gt_labs_probe
            

            # Evaluate kNN
            print("Evaluating kNN classifier...")
            clf = KNeighborsClassifier(n_neighbors=n)
            clf.fit(np.asarray(exper_gallery["feats"]),
                    np.asarray(exper_gallery["gtlabs"]))
            all_pred_labs = clf.predict(np.asarray(exper_probe["feats"]))

            # Compute accuracy from Confusion Matrix
            CM = confusion_matrix(all_gt_labs_probe, all_pred_labs)
            Acc = CM.diagonal().sum() / len(all_gt_labs_probe)
            cm.append(CM)
            acc.append(Acc)

            # Save experiment results
            exper_results = {}
            exper_results["feats"] = all_feats_probe
            exper_results["gtlabs"] = all_gt_labs_probe
            exper_results["predlabs"] = all_pred_labs
            if outpath is not None:
                dd.io.save(outpath, exper_results)
                print("Results saved to: " + outpath)

    return acc, cm

# ----------------------------------------------------------------------------


_examples = '''examples:

  # Test the pre-trained Singlenet model for Palm vein recognition
  python %(prog)s --epochs 10 --datadir /path/of/dataset/ --nclasses 100 --bs 64 --mode ft --initnet /path/of/model.h5
'''

# ----------------------------------------------------------------------------


if __name__ == "__main__":

    # Input arguments
    parser = argparse.ArgumentParser(
        description='''Test pre-trained CNN singlenet model for palm vein recognition.\n\n  Run 'python %(prog)s --help' for argument help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--datadir', type=str, required=False,
                        default=osp.join(MAIN_PATH, 'datasets', 'CASIA'),
                        help="Full path to data directory")
    parser.add_argument('--dataset', type=str, required=False,
                        default='850',
                        help="Name of the dataset")
    parser.add_argument('--dataset_variant', type=str, required=False,
                        default='normal',
                        help="Preprocessing variant of the dataset")
    parser.add_argument('--nclasses', type=int, required=False,
                        default=100, #cambiar
                        help='Number of individual classes')
    parser.add_argument('--hands', nargs='*', required=False,
                        default= ['Left', 'Right'],#['l', 'r'],
                        help='Notation of hands per individual')
    parser.add_argument('--nsamples', type=int, required=False,
                        default=6, #cambiar
                        help='Number of samples per hand')
    parser.add_argument('--ftsamples', type=int, required=False,
                        default=1,
                        help='Number of samples for fine-tuning')
    parser.add_argument('--prefix', type=str, required=False,
                        default=NAME,
                        help="String to prefix experiment directory name.")
    parser.add_argument('--bs', type=int, required=False,
                        default=64,
                        help='Batch size (default=64)')
    parser.add_argument('--initnet', type=str, required=False,
                        default="",
                        help="Path to model network to initialize")
    parser.add_argument('--knn', type=int, required=False,
                        default=7,
                        help='Number of noighbours for kNN classifier')
    parser.add_argument('--mode', type=str, required=False,
                        default="all",
                        help="Testing mode: all - classify testing data; ft/ft+aug - divide testing data in fine-tuning and testing, then classify with kNN")
    parser.add_argument('--metric', type=str, required=False,
                        default="L2",
                        help="kNN metric: L1 | L2 | Chebyshev | Minkowski")
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument("--verbose", type=int,
                        nargs='?', const=False, default=1,
                        help="Whether to enable verbosity of output")

    parser.usage = parser.format_help()
    args = parser.parse_args()

    # Process input arguments
    verbose = args.verbose
    datadir = args.datadir
    dataset = args.dataset
    dataset_variant = args.dataset_variant
    prefix = args.prefix
    batch_size = args.bs
    nclasses = args.nclasses
    hands = args.hands
    nsamples = args.nsamples
    ftsamples = args.ftsamples
    IS_DEBUG = args.debug
    initnet = args.initnet
    mode = args.mode
    # knn = args.knn
    knn = [1,3,5]
    metric = args.metric

    experdir, filename = os.path.split(initnet)

    if not osp.exists(experdir):
        os.makedirs(experdir)

    # Read the testing dataset
    ft_data, test_data, nclasses = read_data(datadir=datadir, dataset=dataset, dataset_variant=dataset_variant,
                                             nclasses=nclasses, hands=hands, nsamples=nsamples, ftsamples=ftsamples, mode=mode)

    # Pre-process the data
    ft_samples, test_samples, ft_labels, test_labels = preprocessing_data(ft_data=ft_data, test_data=test_data,
                                                                          nclasses=nclasses, mode=mode)

    print("[INFO]: Fine-tuning samples: {} ".format(len(ft_samples)))
    print("[INFO]: Testing samples: {} ".format(len(test_samples)))
    print("[INFO]: Individual classes: {} ".format(nclasses))

    if mode == "all":
        # Load the pre-trained CNN model
        model = load(initnet=initnet, include_top=True, verbose=verbose)

        # Test the CNN model
        results = model.evaluate(
            x=test_samples, y=test_labels, batch_size=batch_size, verbose=1)
        print('Test loss, Test acc:', results)
        print("[INFO]: Training samples: {} ".format(len(ft_samples)))
        print("[INFO]: Testing samples: {} ".format(len(test_samples)))
        print("[INFO]: Individual classes: {} ".format(nclasses))

    else:
        # Load the pre-trained CNN model
        model = load(initnet=initnet, include_top=False, verbose=verbose)

        # Evaluate the model by using kNN classifier
        acc, cm = evaluate_kNN(model=model, nclasses=nclasses, ftsamples=ftsamples,
                               metric=metric, mode=mode, knn=knn,
                               ft_data=ft_samples, ft_gt_labels=ft_labels,
                               test_data=test_samples, test_gt_labels=test_labels)

        for i in range(len(knn)):
            print("Testing acc knn {:01}: {:.4f}".format(knn[i],acc[i]))
        print("[INFO]: Training samples: {} ".format(len(ft_samples)))
        print("[INFO]: Testing samples: {} ".format(len(test_samples)))
        print("[INFO]: Individual classes: {} ".format(nclasses))
