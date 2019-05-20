import numpy as np

import os
import imageio
from os.path import join
from skimage.transform import resize
import pickle

traindir = 'galaxy_mergers/noninteracting/training'
valdir = 'galaxy_mergers/noninteracting/validation'
testdir = 'galaxy_mergers/noninteracting/test'


def read_images(path, resolution=None):
    X = []
    for image_name in os.listdir(path):
        if image_name[0] != '.' and image_name[-5:] == '.jpeg':
            x = imageio.imread(join(path, image_name))

            if resolution is not None:
                if x.shape[0] < resolution[0] or x.shape[1] < resolution[1]:
                    print('Trying to reshape {} to {}'.format(
                        x.shape, resolution))

                x = resize(x, resolution, mode='constant', anti_aliasing=True)
            X += [x.reshape(1, *x.shape)]
        else:
            print('Not using file named: {}'.format(image_name))
    X = np.concatenate(X, axis=0)

    X = (X * 256).astype('uint8')

    print(np.max(X), np.min(X))
    print(X.shape)

    return X


X_train = np.concatenate(
    [
        read_images(
            'galaxy_mergers/noninteracting/training', resolution=(64, 64)),
        # read_images(
        #     'galaxy_mergers/merger/training', resolution=(64, 64))
    ], axis=0)

X_val = np.concatenate(
    [
        read_images(
            'galaxy_mergers/noninteracting/validation', resolution=(64, 64)),
        # read_images(
        #     'galaxy_mergers/merger/validation', resolution=(64, 64))
    ], axis=0)

X_test = np.concatenate(
    [
        read_images(
            'galaxy_mergers/noninteracting/test', resolution=(64, 64)),
        # read_images(
        #     'galaxy_mergers/merger/test', resolution=(64, 64))
    ], axis=0)


data = (X_train, X_val, X_test)

with open('galaxy.pkl', 'wb') as f:
    pickle.dump(data, f)

with open('galaxy.pkl', 'rb') as f:
    X_train, X_val, X_test = pickle.load(f)


# testdir = 'galaxy_mergers/noninteracting/test'
