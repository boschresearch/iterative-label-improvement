########################################################
# train_opILI.py                                       #
#                                                      #
# train a baseline model, on erroneous labels          #
# with iterative label improvement with partitioning   #
#                                                      #
# will save the resulting accuracy to                  #
#
# reports/opili_(augmentation)_dataset-name_error-type_model-name_mode(_th)_num-iter_noise-frac_ili-iter_acc_A/B/final.txt
# reports/opili_(augmentation)_dataset-name_error-type_model-name_mode(_th)_num-iter_noise-frac_acc_final.npy
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
parser = argparse.ArgumentParser(description="opILI Experiments")

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
    save_as = "opili_" + is_aug + dataset_name + "_" + error_type + "_" + model_name + "_" + mode + \
              "_" + str(num_iter) + "_" + str(frac)
elif mode == "confidence":
    save_as = "opili_" + is_aug + dataset_name + "_" + error_type + "_" + model_name + "_" + mode + \
              "_" + str(th) + "_" + str(num_iter) + "_" + str(frac)

# -------------------------------------------------------------------------------------------------------------- #
# split into A/B for opILI
x_train_A, x_train_B, y_train_A, y_train_B = train_test_split(x_train, y_train, test_size=0.5)

prev_accs = [0, 0]  # patience for accuracy: 2 iterations

# optimization iterations
for i in range(num_iter):

    callbacks = []

    if model_name == "mnist_cnn":
        model_A = mnist_load_model(input_shape, num_classes=num_classes)
        model_A.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adadelta(),
                        metrics=['accuracy'])

        model_B = mnist_load_model(input_shape, num_classes=num_classes)
        model_B.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adadelta(),
                        metrics=['accuracy'])

    elif model_name == "cifar_cnn":
        model_A = cifar_load_model(input_shape, num_classes=num_classes)
        opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
        model_A.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])

        model_B = cifar_load_model(input_shape, num_classes=num_classes)
        opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
        model_B.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])

    elif model_name == "resnet32":
        model_A = resnet_v1_load_model(input_shape, resnet_depth, num_classes=num_classes)
        lr_scheduler_A = LearningRateScheduler(lr_schedule)
        lr_reducer_A = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                         cooldown=0,
                                         patience=5,
                                         min_lr=0.5e-6)
        callbacks = [lr_reducer_A, lr_scheduler_A]
        model_A.compile(loss='categorical_crossentropy',
                        optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                        metrics=['accuracy'])

        model_B = resnet_v1_load_model(input_shape, resnet_depth, num_classes=num_classes)
        lr_scheduler_B = LearningRateScheduler(lr_schedule)
        lr_reducer_B = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                         cooldown=0,
                                         patience=5,
                                         min_lr=0.5e-6)
        callbacks = [lr_reducer_B, lr_scheduler_B]
        model_B.compile(loss='categorical_crossentropy',
                        optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                        metrics=['accuracy'])

    elif model_name == "resnet50":
        model_A = resnet_v1_load_model(input_shape, resnet50_depth, num_classes=num_classes)
        lr_scheduler_A = LearningRateScheduler(lr_schedule)
        lr_reducer_A = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)
        callbacks = [lr_reducer_A, lr_scheduler_A]
        model_A.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])

        model_B = resnet_v1_load_model(input_shape, resnet50_depth, num_classes=num_classes)
        lr_scheduler_B = LearningRateScheduler(lr_schedule)
        lr_reducer_B = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)
        callbacks = [lr_reducer_B, lr_scheduler_B]
        model_B.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])

    if AUG:
        # advanced data augmentation for resnet50, using albumenations
        if model_name == "resnet50":
            train_gen = AugmentationSequence(x_train_A, y_train_A, batch_size, augmentations=AUGMENTATIONS_TRAIN)
            history_A = model_A.fit_generator(train_gen,
                                          steps_per_epoch=x_train_A.shape[0] // batch_size,
                                          epochs=epochs,
                                          callbacks=callbacks,
                                          validation_data=(x_test, y_test),
                                          workers=1)
        else:
            datagen.fit(x_train_A)
            history_A = model_A.fit_generator(datagen.flow(x_train_A, y_train_A,
                                                       batch_size=batch_size),
                                          steps_per_epoch=x_train_A.shape[0] // batch_size,
                                          epochs=epochs,
                                          callbacks=callbacks,
                                          validation_data=(x_test, y_test),
                                          workers=1)
    else:
        history_A = model_A.fit(x_train_A, y_train_A,  # chs
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                callbacks=callbacks,
                                validation_data=(x_test, y_test))
    if mode == "plain":
        y_train_B = to_categorical(model_A.predict(x_train_B).argmax(axis=1), num_classes=num_classes)
    elif mode == "confidence":
        preds = model_A.predict(x_train_B)
        confs = preds.max(axis=1)
        idx = confs > th
        y_train_B[idx] = to_categorical(preds[idx].argmax(axis=1), num_classes=num_classes)

    if AUG:
        # advanced data augmentation for resnet50, using albumenations
        if model_name == "resnet50":
            train_gen = AugmentationSequence(x_train_B, y_train_B, batch_size, augmentations=AUGMENTATIONS_TRAIN)
            history_B = model_B.fit_generator(train_gen,
                                          steps_per_epoch=x_train_B.shape[0] // batch_size,
                                          epochs=epochs,
                                          callbacks=callbacks,
                                          validation_data=(x_test, y_test),
                                          workers=1)
        else:
            datagen.fit(x_train_B)
            history_B = model_B.fit_generator(datagen.flow(x_train_B, y_train_B,
                                                       batch_size=batch_size),
                                          steps_per_epoch=x_train_B.shape[0] // batch_size,
                                          epochs=epochs,
                                          callbacks=callbacks,
                                          validation_data=(x_test, y_test),
                                          workers=4)
    else:
        history_B = model_B.fit(x_train_B, y_train_B,  # chs
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                callbacks=callbacks,
                                validation_data=(x_test, y_test))
    if mode == "plain":
        y_train_A = to_categorical(model_B.predict(x_train_A).argmax(axis=1), num_classes=num_classes)
    elif mode == "confidence":
        preds = model_B.predict(x_train_A)
        confs = preds.max(axis=1)
        idx = confs > th
        y_train_A[idx] = to_categorical(preds[idx].argmax(axis=1), num_classes=num_classes)

    if SAVE:
        model_path = os.path.abspath(os.path.join(dirname, "models", save_as + "_" + str(i)))  # chs
        model_A.save(model_path + "_A")
        model_B.save(model_path + "_B")

    if SAVEHIST:
        log_path = os.path.abspath(os.path.join(dirname,
                                                "reports", "history", save_as + "_" + str(i) + "_A"))
        np.save(log_path, np.array([history_A.history["loss"], history_A.history["val_loss"]]))
        log_path = os.path.abspath(os.path.join(dirname,
                                                "reports", "history", save_as + "_" + str(i) + "_B"))
        np.save(log_path, np.array([history_B.history["loss"], history_B.history["val_loss"]]))

    # current performance on test data
    print("### Iteration: " + str(i) + ", current performance: ")
    score_A = model_A.evaluate(x_test, y_test, verbose=0)
    score_B = model_B.evaluate(x_test, y_test, verbose=0)
    print('    * Test loss:', score_A[0], score_B[0])
    print('    * Test accuracy:', score_A[1], score_B[1])
    acc_A = score_A[1]
    acc_B = score_B[1]

    # save checkpoints
    log_path = os.path.abspath(
        os.path.join(dirname, "reports", save_as + "_" + str(i) + "_A.npy"))
    np.save(log_path, [i, acc_A])

    # save checkpoints
    log_path = os.path.abspath(
        os.path.join(dirname, "reports", save_as + "_" + str(i) + "_B.npy"))
    np.save(log_path, [i, acc_B])

    val_score = model_A.evaluate(x_val, y_val, verbose=0)
    noisy_val_acc = val_score[1]
    prev_accs.append(noisy_val_acc)
    if prev_accs[-2] > noisy_val_acc and prev_accs[-3] > noisy_val_acc:
        print("Noisy validation accuracy did not increase for two iterations, stopping. ")
        break

##################
# Final Training #
##################
if mode == "plain":
    y_train_A = to_categorical(model_B.predict(x_train_A).argmax(axis=1), num_classes=num_classes)
    y_train_B = to_categorical(model_A.predict(x_train_B).argmax(axis=1), num_classes=num_classes)
elif mode == "confidence":
    preds = model_A.predict(x_train_B)
    confs = preds.max(axis=1)
    idx = confs > th
    y_train_B[idx] = to_categorical(preds[idx].argmax(axis=1), num_classes=num_classes)

    preds = model_B.predict(x_train_A)
    confs = preds.max(axis=1)
    idx = confs > th
    y_train_A[idx] = to_categorical(preds[idx].argmax(axis=1), num_classes=num_classes)

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
    x_train = np.concatenate((x_train_A, x_train_B), axis=0)
    y_train = np.concatenate((y_train_A, y_train_B), axis=0)

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
                                      workers=4)
else:
    history = model.fit(np.concatenate((x_train_A, x_train_B), axis=0), np.concatenate((y_train_A, y_train_B), axis=0),
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
log_path = os.path.abspath(os.path.join(dirname, "reports", save_as + "_acc_final"))
np.save(log_path, score[1])

K.clear_session()
