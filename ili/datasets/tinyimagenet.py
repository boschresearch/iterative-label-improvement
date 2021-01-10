import os
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    x_train, y_train = _load_timgnet_data(subset="train")
    x_val, y_val = _load_timgnet_data(subset="val")
    return (x_train, y_train), (x_val, y_val)


def _load_timgnet_data(subset="train", path="<your-path-to-tinyimagenet>/tinyImageNet/tiny-imagenet-200/"):
    if path == "<your-path-to-tinyimagenet>/tinyImageNet/tiny-imagenet-200/":
        raise ValueError("Please set the directory to tiny-imagenet in src/datasets/tinyimagenet.py")
    wnids = np.loadtxt(path + "wnids.txt", dtype=np.str)

    # int labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}
    label_to_wnid = {i: wnid for i, wnid in enumerate(wnids)}

    # words --> names for each class
    with open(os.path.join(path, 'words.txt'), 'r') as f:
        wnid_to_name = dict(line.split('\t') for line in f)
        for wnid, names in wnid_to_name.items():
            wnid_to_name[wnid] = [w.strip() for w in names.split(',')]
        class_names = [wnid_to_name[wnid] for wnid in wnids]

    if not subset in ["train", "val", "test"]:
        raise ValueError("subset: " + subset + " unkown. Please enter: train/val/test.")

    # lists for images and labels
    x, y = [], []

    if subset == "train":
        train_ids = wnids
        for i, wnid in enumerate(wnids):

            filenames = os.listdir(path + subset + "/" + wnid + "/images/")

            # iterate the file, read the imgs
            for file in filenames:
                # if subset == "train":
                img = plt.imread(path + subset + "/" + wnid + "/images/" + file)
                # else:
                #    img = plt.imread(path + subset + "/images/" + file)

                if img.ndim == 2:
                    img = img[:, :, np.newaxis]

                    img = np.concatenate([np.zeros((img.shape[0], img.shape[1], 1)),
                                          np.zeros((img.shape[0], img.shape[1], 1)),
                                          img], axis=-1)

                x.append(img)
                y.append(wnid_to_label[wnid])

    elif subset == "val":
        val_file = path + "val/" + "val_annotations.txt"
        with open(val_file) as f:

            for line in f:

                filename, wnid = line.split("\t")[:2]
                label = wnid_to_label[wnid]

                img = plt.imread(path + subset + "/images/" + filename)
                if img.ndim == 2:
                    img = img[:, :, np.newaxis]

                    img = np.concatenate([np.zeros((img.shape[0], img.shape[1], 1)),
                                          np.zeros((img.shape[0], img.shape[1], 1)),
                                          img], axis=-1)
                x.append(np.array(img))

                y.append(label)

    elif subset == "test":
        filenames = os.listdir(path + subset + "/images/")

        # iterate the file, read the imgs
        for file in filenames:
            img = plt.imread(path + subset + "/images/" + file)

            if img.ndim == 2:
                img = img[:, :, np.newaxis]

                img = np.concatenate([np.zeros((img.shape[0], img.shape[1])),
                                      np.zeros((img.shape[0], img.shape[1])),
                                      img], axis=-1)
            x.append(np.array(img))

    # return labels only for train and val
    if subset == "test":
        return np.array(x)
    else:
        return np.array(x), np.array(y)
