from os.path import join, isfile

import numpy as np
from sklearn.model_selection import LeaveOneOut

DATA_DIR = '../data/'

class NotProvidedError(Exception):
    def __init__(self, msg):
        super().__init__(msg)

class TrainTestDataset:
    def __init__(self, train_raw, train_labels, test_raw, test_labels, electrode_names, name):
        self.train_raw_ = train_raw
        self.train_labels_ = train_labels
        self.test_raw_ = test_raw
        self.test_labels_ = test_labels
        self.electrode_names_ = electrode_names
        self.name_ = name

    @property
    def name(self):
        return self.name_

    @property
    def X_train(self):
        return self.train_raw_

    @property
    def X_test(self):
        return self.test_raw_

    @property
    def y_train(self):
        return self.train_labels_

    @property
    def y_test(self):
        return self.test_labels_

    @property
    def electrode_names(self):
        return self.electrode_names_

    @property
    def nb_electrodes(self):
        return len(self.electrode_names)

    @property
    def prop_train_set(self):
        return self.train_labels_.size / (self.train_labels_.size + self.test_labels_.size)

    def __filter_on_electrodes(self, electrodes):
        return TrainTestDataset(
            train_raw=self.X_train[:,electrodes,:],
            train_labels=self.y_train,
            test_raw=self.X_test[:,electrodes,:],
            test_labels=self.y_test,
            electrode_names=self.electrode_names[electrodes],
            name=self.name
        )

    def filter_on_electrodes_by_name(self, electrode_names):
        indices = np.zeros(self.nb_electrodes, dtype=np.bool)
        for electrode in electrode_names:
            indices[np.where(self.electrode_names_ == electrode)[0]] = True
        return self.__filter_on_electrodes(indices)

class Dataset:
    def __init__(self, signals, covs, labels, electrode_names, name):
        self.signals_ = signals
        self.covs_ = covs
        self.labels_ = labels
        self.electrode_names_ = electrode_names
        self.name_ = name

    @property
    def name(self):
        return self.name_

    @property
    def nb_patients(self):
        if self.signals_ is None:
            return len(self.covs_)
        else:
            return len(self.signals_)

    @property
    def signals(self):
        if self.signals_ is None:
            raise NotProvidedError('Raw EEG signals are not provided')
        return self.signals_

    @property
    def covs(self):
        return self.covs_

    @property
    def labels(self):
        return self.labels_

    @property
    def electrode_names(self):
        return self.electrode_names_

    @property
    def nb_electrodes(self):
        return self.signals[0].shape[1]

    def loo(self):
        loo = LeaveOneOut()
        for train_idx, test_idx in loo.split(self.signals):
            test_idx = test_idx[0]
            X_train = np.vstack(list(map(np.asarray, self.signals[train_idx])))
            y_train = np.hstack(self.labels[train_idx])
            X_test = self.signals[test_idx]
            y_test = self.labels[test_idx]
            yield TrainTestDataset(
                train_raw=X_train,
                train_labels=y_train,
                test_raw=self.signals[test_idx],
                test_labels=self.labels[test_idx],
                electrode_names=self.electrode_names,
                name=f'{self.name}.{test_idx}'
            )

def load_data():
    signals_path = join(DATA_DIR, 'raw_eeg_per_patient.npy')
    if isfile(signals_path):
        signals = np.load(signals_path, allow_pickle=True)
    else:
        signals = None
    covs = np.load(join(DATA_DIR, 'eeg_xdawn_cov.npy'), allow_pickle=True)
    labels = np.load(join(DATA_DIR, 'labels_per_subject.npy'), allow_pickle=True)
    electrode_names = np.load(join(DATA_DIR, 'all_channels.npy'))
    return Dataset(signals, covs, labels, electrode_names, 'Tunnel_Checkerboard')

def load_data_sources():
    signals_path = join(DATA_DIR, 'sources_raw_eeg_signals.npy')
    if isfile(signals_path):
        signals = np.load(signals_path, allow_pickle=True)
    else:
        signals = None
    covs = np.load(join(DATA_DIR, 'src_xdawn_cov.npy'), allow_pickle=True)
    labels = np.load(join(DATA_DIR, 'labels_per_subject.npy'), allow_pickle=True)
    electrode_names = None
    return Dataset(signals, covs, labels, electrode_names, 'sources_Tunnel_Checkerboard')
    
