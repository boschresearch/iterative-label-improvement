########################################################
# train_ILI.py                                         #
#                                                      #
# train a baseline model, on erroneous labels          #
# with iterative label improvement                     #
#                                                      #
# will save the resulting accuracy to                  #
#
# reports/ili_(augmentation)_dataset-name_error-type_model-name_mode(_th)_num-iter_noise-frac_ili-iter_acc.txt
#
# create the following directories first
# reports/
# reports/history/
# reports/figures/
# e.g. using setup_dirs
########################################################
from __future__ import print_function
import keras
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.datasets import mnist, cifar10, cifar100
from sklearn.model_selection import train_test_split

import numpy as np
import os
import argparse

from ili.models.helpers_ext import preprocess_data
from ili.models.helpers_ext import mnist_load_model, cifar_load_model, resnet_v1_load_model
from ili.models.helpers_ext import lr_schedule
from ili.models.helpers_ext import datagen

from ili.models.helpers import augment_label_bias_partial, augment_label_random_partial
from ili.datasets import tinyimagenet
from ili.models.augmentations import AugmentationSequence, AUGMENTATIONS_TRAIN

from ili.models.config import mnist_batch_size, cifar_batch_size, resnet_batch_size
from ili.models.config import mnist_epochs, cifar_epochs, resnet_epochs
from ili.models.config import resnet_depth
from ili.models.config import resnet50_batch_size, resnet50_epochs, resnet50_depth

##############
# Parameters #
##############
parser = argparse.ArgumentParser(description="ILI Experiments")

parser.add_argument("dataset_name", type=str, help="one of: mnist | cifar10 | cifar100 | tinyimagenet")
parser.add_argument("error_type",   type=str, help="one of: bias | random")
parser.add_argument("model_name",   type=str, help="one of: mnist_cnn | cifar_cnn | resnet32 | resnet50")

subparsers = parser.add_subparsers(dest="mode", help="one of: plain | confidence")
confidence_parser = subparsers.add_parser("confidence",  help="confidence threshold, e.g.: >> confidence 0.3")
confidence_parser.add_argument("th", type=float, help="Confidence threshold")
plain_parser = subparsers.add_parser("plain")

parser.add_argument("num_iter",     type=int, default=10, help="number of optimization iterations")
parser.add_argument("frac",         type=float, help="Noise fraction [0,1)")

parser.add_argument("--AUG",          action="store_true", help="activate data augmentation")
parser.add_argument("--SAVE",         action="store_true", help="save model weights after training")
parser.add_argument("--SAVEHIST",     action="store_true", help="save history after training")


args = parser.parse_args()

dataset_name = args.dataset_name    # one of: mnist | cifar10 | cifar100 | tinyimagenet
error_type = args.error_type        # one of: bias | random
model_name = args.model_name        # one of: mnist_cnn | cifar_cnn | resnet32 | resnet50
mode = args.mode                    # one of: plain | confidence
if mode == "confidence":            # confidence mode options
    th = args.th                    # confidence threshold
num_iter = args.num_iter            # number of optimization iterations
frac = args.frac                    # Noise fraction
AUG = args.AUG                      # activate data augmentation
SAVE = args.SAVE                    # save model weight after training
SAVEHIST = args.SAVEHIST            # save history after training
# -------------------------------------------------------------------------------------------------------------- #
dirname = os.path.dirname(os.path.abspath(__file__))

# Data
if dataset_name == "mnist":
    dataset = mnist
elif dataset_name == "cifar10":
    dataset = cifar10
elif dataset_name == "cifar100":
    dataset = cifar100
elif dataset_name == "tinyimagenet":
    dataset = tinyimagenet
else:
    raise ValueError("Unknown dataset: " + dataset_name)

# Model
if model_name == "mnist_cnn":
    batch_size = mnist_batch_size
    epochs = mnist_epochs
elif model_name == "cifar_cnn":
    batch_size = cifar_batch_size
    epochs = cifar_epochs
elif model_name == "resnet32":
    epochs = resnet_epochs
    batch_size = resnet_batch_size
elif model_name == "resnet50":
    epochs = resnet50_epochs
    batch_size = resnet50_batch_size
else:
    raise ValueError("Unknown model: " + model_name)

(x_train, y_train), (x_test, y_test) = dataset.load_data()

# Error type
if frac > 0:
    if error_type == "bias":
        x_train, y_train = augment_label_bias_partial(x_train, y_train, 4, 7, frac)
    elif error_type == "random":
        x_train, y_train = augment_label_random_partial(x_train, y_train, frac)

x_train, y_train, x_test, y_test, input_shape = preprocess_data(x_train, y_train, x_test, y_test)

num_classes = len(np.unique(y_train.argmax(axis=1)))

# for ILI we use noisy validation data for meta-early stopping.
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

is_aug = ""
if AUG:
    is_aug = "augmentation_"
if mode == "plain":
    save_as = "ili_" + is_aug + dataset_name + "_" + error_type + "_" + model_name + "_" + mode + \
              "_" + str(num_iter) + "_" + str(frac)
elif mode == "confidence":
    save_as = "ili_" + is_aug + dataset_name + "_" + error_type + "_" + model_name + "_" + mode + \
              "_" + str(th) + "_" + str(num_iter) + "_" + str(frac)

# -------------------------------------------------------------------------------------------------------------- #
prev_accs = [0, 0]  # patience for accuracy: 2 iterations

# optimization iterations
for i in range(num_iter):

    callbacks = []

    if model_name == "mnist_cnn":
        model = mnist_load_model(input_shape, num_classes=num_classes)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

    elif model_name == "cifar_cnn":
        model = cifar_load_model(input_shape, num_classes=num_classes)

        opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

    elif model_name == "resnet32":
        model = resnet_v1_load_model(input_shape, resnet_depth, num_classes=num_classes)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)
        callbacks = [lr_reducer, lr_scheduler]
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])

    elif model_name == "resnet50":
        model = resnet_v1_load_model(input_shape, resnet50_depth, num_classes=num_classes)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)
        callbacks = [lr_reducer, lr_scheduler]
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])

    if AUG:
        # advanced data augmentation for resnet50, using albumenations
        if model_name == "resnet50":
            train_gen = AugmentationSequence(x_train, y_train, batch_size, augmentations=AUGMENTATIONS_TRAIN)
            history = model.fit_generator(train_gen,
                                          steps_per_epoch=x_train.shape[0] // batch_size,
                                          epochs=epochs,
                                          callbacks=callbacks,
                                          validation_data=(x_test, y_test),
                                          workers=1)
        else:
            datagen.fit(x_train)
            history = model.fit_generator(datagen.flow(x_train, y_train,
                                                       batch_size=batch_size),
                                          steps_per_epoch=x_train.shape[0] // batch_size,
                                          epochs=epochs,
                                          callbacks=callbacks,
                                          validation_data=(x_test, y_test),
                                          workers=1)
    else:
        history = model.fit(x_train, y_train,  # chs
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=(x_test, y_test))

    if SAVE:
        model_path = os.path.abspath(os.path.join(dirname, "models", save_as + "_" + str(i)))
        model.save(model_path)
    if SAVEHIST:
        log_path = os.path.abspath(os.path.join(dirname,
                                                "reports", "history", save_as + "_" + str(i)))
        np.save(log_path, np.array([history.history["loss"], history.history["val_loss"]]))

    # current performance on test data
    print("### Iteration: " + str(i) + ", current performance: ")
    score = model.evaluate(x_test, y_test, verbose=0)
    print('    * Test loss:', score[0])
    print('    * Test accuracy:', score[1])

    acc = score[1]
    if mode == "plain":
        y_train = to_categorical(model.predict(x_train).argmax(axis=1), num_classes=num_classes)
    elif mode == "confidence":
        preds = model.predict(x_train)
        confs = preds.max(axis=1)
        idx = confs > th
        y_train[idx] = to_categorical(preds[idx].argmax(axis=1), num_classes=num_classes)

    # save checkpoints
    log_path = os.path.abspath(os.path.join(dirname, "reports", save_as + "_" + str(i) + ".npy"))
    np.save(log_path, [i, acc])

    val_score = model.evaluate(x_val, y_val, verbose=0)
    noisy_val_acc = val_score[1]
    prev_accs.append(noisy_val_acc)
    if prev_accs[-2] > noisy_val_acc and prev_accs[-3] > noisy_val_acc:
        print("Noisy validation accuracy did not increase for two iterations, stopping. ")
        break

K.clear_session()
