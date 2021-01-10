########################################################
# train.py                                             #
#                                                      #
# train a baseline model, on erroneous labels          #
# with no iterative label improvement                  #
#                                                      #
# will save the resulting accuracy to                  #
#
# reports/noisy_baseline_(augmentation)_dataset-name_error-type_model-name_noise-frac_run-idx_acc.txt
#
# create directory ../../reports/ first 
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
parser = argparse.ArgumentParser(description="Noisy Baseline Experiments")

parser.add_argument("dataset_name", type=str, help="one of: mnist | cifar10 | cifar100 | tinyimagenet")
parser.add_argument("error_type",   type=str, help="one of: bias | random")
parser.add_argument("model_name",   type=str, help="one of: mnist_cnn | cifar_cnn | resnet32 | resnet50")

parser.add_argument("frac",         type=float, help="Noise fraction [0,1)")
parser.add_argument("run_idx",      type=int, help="index of run")

parser.add_argument("--AUG",          action="store_true", help="activate data augmentation")
parser.add_argument("--SAVE",         action="store_true", help="save model weights after training")
parser.add_argument("--SAVEHIST",     action="store_true", help="save history after training")


args = parser.parse_args()

dataset_name = args.dataset_name    # one of: mnist | cifar10 | cifar100 | tinyimagenet
error_type = args.error_type        # one of: bias | random
model_name = args.model_name        # one of: mnist_cnn | cifar_cnn | resnet32 | resnet50
run_idx = args.run_idx              # index of run
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
# here we need to split as well, to ensure baseline and ILI use the same amount of data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

is_aug = ""
if AUG:
    is_aug = "augmentation_"
save_as = "noisy_baseline_" + is_aug + dataset_name + "_" + error_type + "_" + model_name + "_" + str(frac) + "_" + str(run_idx)

# -------------------------------------------------------------------------------------------------------------- #
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
                                      workers=2,
                                      use_multiprocessing=False)
                                      #workers=4)
else:
    # without augmentation
    history = model.fit(x_train, y_train,  # chs
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(x_test, y_test))

if SAVE:
    model_path = os.path.abspath(os.path.join(dirname, "models", save_as))
    model.save(model_path)  # chs
if SAVEHIST:
    log_path = os.path.abspath(os.path.join(dirname,
                                            "reports", "history", save_as))
    np.save(log_path, np.array([history.history["loss"], history.history["val_loss"]]))

# current performance on test data
print("### Run: " + str(run_idx) + ", current performance: ")
score = model.evaluate(x_test, y_test, verbose=0)
print('    * Test loss:', score[0])
print('    * Test accuracy:', score[1])

acc = score[1]

acc = np.array(acc, ndmin=1)
log_path = os.path.abspath(os.path.join(dirname, "reports", save_as + "_acc.txt"))
np.savetxt(log_path, acc)

K.clear_session()
