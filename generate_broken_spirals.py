"""Generate images for CNNs from ATTPC events.

Author: Ryan Strauss
edited-simulated function: Dulce Jarquin 11 June 2019
"""
import math

import click
import h5py
import matplotlib
import numpy as np
import os
import pandas as pd
import pytpc
from sklearn.utils import shuffle


import data_discretization as dd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Real data-processing runs to use
RUNS = ['0130', '0210']


def _l(a):
    return 0 if a == 0 else math.log10(a)


def real_labeled(projection, data_dir, save_path, prefix):
    print('Processing data...')
    data = []
    for run in RUNS:
        events_file = os.path.join(data_dir, 'run_{}.h5'.format(run))
        labels_file = os.path.join(data_dir, 'run_{}_labels.csv'.format(run))

        events = pytpc.HDFDataFile(events_file, 'r')
        labels = pd.read_csv(labels_file, sep=',')

        proton_indices = labels.loc[(labels['label'] == 'p')]['evt_id'].values
        carbon_indices = labels.loc[(labels['label'] == 'c')]['evt_id'].values
        junk_indices = labels.loc[(labels['label'] == 'j')]['evt_id'].values

        for evt_id in proton_indices:
            event = events[str(evt_id)]
            xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                              baseline_correction=False,
                              cg_times=False)
            data.append([xyzs, 0])

        for evt_id in carbon_indices:
            event = events[str(evt_id)]
            xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                              baseline_correction=False,
                              cg_times=False)

            data.append([xyzs, 1])

        for evt_id in junk_indices:
            event = events[str(evt_id)]
            xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                              baseline_correction=False,
                              cg_times=False)

            data.append([xyzs, 2])

    log = np.vectorize(_l)

    for event in data:
        event[0][:, 3] = log(event[0][:, 3])

     #Shuffle data-processing
    data = shuffle(data)

    # Split into train and test sets
    partition = int(len(data) * 0.8)
    train = data[:partition]
    test = data[partition:]

    # Normalize
    max_charge = np.array(list(map(lambda x: x[0][:, 3].max(), train))).max()

    for e in train:
        for point in e[0]:
            point[3] = point[3] / max_charge

    for e in test:
        for point in e[0]:
            point[3] = point[3] / max_charge

    print('Making images...')

    # Make train numpy sets
    train_features = np.empty((len(train), 128, 128, 3), dtype=np.uint8)
    train_targets = np.empty((len(train),), dtype=np.uint8)

    for i, event in enumerate(train):
        e = event[0]
        if projection == 'zy':
            x = e[:, 2].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        elif projection == 'xy':
            x = e[:, 0].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        else:
            raise ValueError('Invalid projection value.')
        fig = plt.figure(figsize=(1, 1), dpi=128)
        if projection == 'zy':
            plt.xlim(0.0, 1250.0)
        elif projection == 'xy':
            plt.xlim(-275.0, 275.0)
        plt.ylim((-275.0, 275.0))
        plt.axis('off')
        plt.scatter(x, z, s=0.6, c=c, cmap='Greys')
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
        data = np.delete(data, 3, axis=2)
        train_features[i] = data
        train_targets[i] = event[1]
        plt.close()

    # Make test numpy sets
    test_features = np.empty((len(test), 128, 128, 3), dtype=np.uint8)
    test_targets = np.empty((len(test),), dtype=np.uint8)

    for i, event in enumerate(test):
        e = event[0]
        if projection == 'zy':
            x = e[:, 2].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        elif projection == 'xy':
            x = e[:, 0].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        else:
            raise ValueError('Invalid projection value.')
        fig = plt.figure(figsize=(1, 1), dpi=128)
        if projection == 'zy':
            plt.xlim(0.0, 1250.0)
        elif projection == 'xy':
            plt.xlim(-275.0, 275.0)
        plt.ylim((-275.0, 275.0))
        plt.axis('off')
        plt.scatter(x, z, s=0.6, c=c, cmap='Greys')
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
        data = np.delete(data, 3, axis=2)
        test_features[i] = data
        test_targets[i] = event[1]
        plt.close()

    print('Saving file...')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = os.path.join(save_path, prefix + 'images.h5')

    # Save to HDF5
    h5 = h5py.File(filename, 'w')
    h5.create_dataset('train_features', data=train_features)
    h5.create_dataset('train_targets', data=train_targets)
    h5.create_dataset('test_features', data=test_features)
    h5.create_dataset('test_targets', data=test_targets)
    h5.create_dataset('max_charge', data=np.array([max_charge]))
    h5.close()


def real_unlabeled(projection, data_dir, save_path, prefix):
    print('Processing data...')
    data = []
    for run in RUNS:
        events_file = os.path.join(data_dir, 'run_{}.h5'.format(run))
        events = pytpc.HDFDataFile(events_file, 'r')

        for event in events:
            xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                              baseline_correction=False,
                              cg_times=False)

            data.append([xyzs, -1])

    # Take the log of charge data-processing
    log = np.vectorize(_l)

    for event in data:
        event[0][:, 3] = log(event[0][:, 3])

     #Shuffle data-processing
    data = shuffle(data)
    data=data[0:10]

    # Normalize
    max_charge = np.array(list(map(lambda x: x[0][:, 3].max(), data))).max()

    for e in data:
        for point in e[0]:
            point[3] = point[3] / max_charge

    print('Making images...')

    # Make numpy sets
    images = np.empty((len(data), 128, 128, 3), dtype=np.uint8)

    for i, event in enumerate(data):
        e = event[0]
        if projection == 'zy':
            x = e[:, 2].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        elif projection == 'xy':
            x = e[:, 0].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        else:
            raise ValueError('Invalid projection value.')
        fig = plt.figure(figsize=(1, 1), dpi=128)
        if projection == 'zy':
            plt.xlim(0.0, 1250.0)
        elif projection == 'xy':
            plt.xlim(-275.0, 275.0)
        plt.ylim((-275.0, 275.0))
        plt.axis('off')
        plt.scatter(x, z, s=0.6, c=c, cmap='Greys')
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
        data = np.delete(data, 3, axis=2)
        images[i] = data
        plt.close()

    print('Saving file...')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = os.path.join(save_path, prefix + 'images.h5')

    # Save to HDF5
    h5 = h5py.File(filename, 'w')
    h5.create_dataset('images', data=images)
    h5.create_dataset('max_charge', data=np.array([max_charge]))
    h5.close()


def simulated(projection, noise, data_dir, save_path, prefix):

    print("We are in the simulared() function!")
    print('Processing data...')

    proton_events = pytpc.HDFDataFile(os.path.join(data_dir, prefix + 'proton.h5'), 'r')
    carbon_events = pytpc.HDFDataFile(os.path.join(data_dir, prefix + 'carbon.h5'), 'r')

    # Create empty arrays to hold data-processing
    data = []

    # Add proton events to data-processing array
    for i, event in enumerate(proton_events):




        xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=True,
                          baseline_correction=False, cg_times=False)

        #if noise:
            # Add artificial noise
            #xyzs = dd.add_noise(xyzs).astype('float32')
        data.append([xyzs, 0])

        if i % 50 == 0:
            print('Proton event ' + str(i) + ' added.')


    for i, event in enumerate(carbon_events):



        xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=True, baseline_correction=False, cg_times=False)

        #if noise:
            # Add artificial noise
        #    xyzs = dd.add_noise(xyzs).astype('float32')

        data.append([xyzs, 1])

        if i % 50 == 0:
            print('Carbon event ' + str(i) + ' added.')

    # Create junk events
    #for i in range(len(proton_events)):

        #xyzs = np.empty([1, 4])
        #xyzs = dd.add_noise(xyzs).astype('float32')
        #data.append([xyzs, 2])

        #if i % 50 == 0:
            #print('Junk event ' + str(i) + ' added.')
    # Take the log of charge data-processing
    log = np.vectorize(_l)

    for event in data:
        event[0][:, 3] = log(event[0][:, 3])

    # Split into train and test sets
    data = shuffle(data)
    partition = int(len(data) * 0.8)
    train = data[:partition]
    test = data[partition:]

    # Normalize
    max_charge = np.array(list(map(lambda x: x[0][:, 3].max(), train))).max()

    for e in train:
        for point in e[0]:
            point[3] = point[3] / max_charge

    for e in test:
        for point in e[0]:
            point[3] = point[3] / max_charge

    # Make train numpy sets
    train_features = np.empty((len(train), 128, 128, 3), dtype=np.uint8)
    train_targets = np.empty((len(train),), dtype=np.uint8)
    # Make train numpy sets for the broken spiral
    train_features_broken = np.empty((len(train), 32, 32, 3), dtype=np.uint8)
    train_targets_broken = np.empty((len(train),), dtype=np.uint8)
    for i, event in enumerate(train):
        e = event[0]
        c = e[:, 3].flatten()
        p = e[:, 4].flatten()
        if projection == 'zy':
            x = e[:, 2].flatten()
            z = e[:, 1].flatten()

        elif projection == 'xy':
            x = e[:, 0].flatten()
            z = e[:, 1].flatten()

        else:
            raise ValueError('Invalid projection value.')

# deleting part of the Spiral
        x_b = []
        z_b = []
        c_b = []
        for j in range(len(x)):
            if -10<x[j]<127.5 and -117.5<z[j]<20:
                x_b.append(x[j])
                z_b.append(z[j])
                c_b.append(c[j])
                c[j]=0

        # Make numpy array image of spiral context
        fig = plt.figure(figsize=(1, 1), dpi=128)
        if projection == 'zy':
            plt.xlim(0.0, 1250.0)
        elif projection == 'xy':
            plt.xlim(-275.0, 275.0)

        plt.ylim((-275.0, 275.0))

        plt.axis('off')
        plt.scatter(x, z, s=0.6, c=c, cmap='Greys')
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
        data = np.delete(data, 3, axis=2)
        train_features[i] = data
        train_targets[i] = event[1]
        plt.close()

        # Make numpy array image of broken spiral
        fig = plt.figure(figsize=(1, 1), dpi=32)
        if projection == 'zy':
            plt.xlim(0.0, 1250.0)
        elif projection == 'xy':
            plt.xlim(-10, 127.5)
        plt.ylim((-117.5, 20))
        plt.axis('off')
        plt.scatter(x_b, z_b, s=0.6, c=c_b, cmap='Greys')

        fig.canvas.draw()
        data_broken = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
        data_broken = np.delete(data_broken, 3, axis=2)
        train_features_broken[i] = data_broken
        train_targets[i] = event[1]
        plt.close()

    # Make test numpy sets
    test_features = np.empty((len(test), 128, 128, 3), dtype=np.uint8)
    test_targets = np.empty((len(test),), dtype=np.uint8)

    test_features_broken = np.empty((len(test), 32, 32, 3), dtype=np.uint8)
    test_targets_broken = np.empty((len(test),), dtype=np.uint8)

    for i, event in enumerate(test):
        e = event[0]
        p = e[:, 4].flatten()
        if projection == 'zy':
            x = e[:, 2].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        elif projection == 'xy':
            x = e[:, 0].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        else:
            raise ValueError('Invalid projection value.')
        fig = plt.figure(figsize=(1, 1), dpi=128)
        x_b = []
        z_b = []
        c_b = []
        for j in range(len(x)):
            if -10<x[j]<127.5 and -117.5<z[j]<20:
                x_b.append(x[j])
                z_b.append(z[j])
                c_b.append(c[j])
                c[j]=0
        fig = plt.figure(figsize=(1, 1), dpi=128)
        if projection == 'zy':
            plt.xlim(0.0, 1250.0)
        elif projection == 'xy':
            plt.xlim(-275.0, 275.0)
        plt.ylim((-275.0, 275.0))
        plt.axis('off')
        plt.scatter(x, z, s=0.6, c=c, cmap='Greys')
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)


        data = np.delete(data, 3, axis=2)

        test_features[i] = data
        test_targets[i] = event[1]
        plt.close()

        fig = plt.figure(figsize=(1, 1), dpi=32)
        if projection == 'zy':
            plt.xlim(0.0, 1250.0)
        elif projection == 'xy':
            plt.xlim(-10, 127.5)
        plt.ylim((-117.5, 20))
        plt.axis('off')
        plt.scatter(x_b, z_b, s=0.6, c=c_b, cmap='Greys')
        fig.canvas.draw()
        data_broken = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)


        data_broken = np.delete(data_broken, 3, axis=2)

        test_features_broken[i] = data_broken
        test_targets_broken[i] = event[1]
        plt.close()




    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = os.path.join(save_path, prefix + 'images.h5')

    # Save to HDF5
    h5 = h5py.File(filename, 'w')
    h5.create_dataset('train_features', data=train_features)
    h5.create_dataset('train_targets', data=train_targets)
    h5.create_dataset('test_features', data=test_features)
    h5.create_dataset('test_targets', data=test_targets)
    h5.create_dataset('max_charge', data=np.array([max_charge]))

    h5.create_dataset('train_features_broken', data=train_features_broken)
    h5.create_dataset('train_targets_broken', data=train_targets_broken)
    h5.create_dataset('test_features_broken', data=test_features_broken)
    h5.create_dataset('test_targets_broken', data=test_targets_broken)
    h5.close()


@click.command()
@click.argument('type', type=click.Choice(['real', 'sim']), nargs=1)
@click.argument('projection', type=click.Choice(['xy', 'zy']), nargs=1)
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), nargs=1)
@click.option('--save_dir', type=click.Path(exists=False, file_okay=False, dir_okay=True), default='',
              help='Where to save the generated data.')
@click.option('--prefix', type=click.STRING, default='',
              help='Prefix for the saved file names and/or files to read in. By default, there is no prefix.')
@click.option('--labeled', type=click.BOOL, default=True,
              help='If true, only the labeled data will be processed.')
@click.option('--noise', type=click.BOOL, default=True,
              help='Whether or not to add artificial noise to simulated data.')
def main(type, projection, data_dir, save_dir, prefix, labeled, noise):
    """This script will generate and save images from ATTPC event data to be used for CNN training.

    When using real data, this script will look for runs 0130 and 0210, as these are the runs that have
    been partially hand-labeled.
    """
    if type == 'real':
        if labeled:
            real_labeled(projection, data_dir, save_dir, prefix)
        else:
            real_unlabeled(projection, data_dir, save_dir, prefix)
    else:
        for i in range(5):
            prefix = 'batch_{}_'.format(i+1)
            simulated(projection, noise, data_dir, save_dir, prefix)


if __name__ == '__main__':
    main()
