import functools
from os.path import normpath, join, isfile
from time import time

import numpy as np

class ProgressBar:
    def __init__(self, N, nb_subdivisions=100, symbol='#'):
        self.N_ = N
        self.nb_subdivisions_ = nb_subdivisions
        self.symbol_ = symbol
        self.start_ = time()
        self.counter_ = 0

    def update(self):
        assert self.counter_ < self.N_
        self.counter_ += 1
        now = time()
        prop = self.counter_ / self.N_
        current_state = int(prop*self.nb_subdivisions_)
        print(f'[{self.symbol_*current_state}{" "*(self.nb_subdivisions_-current_state)}]  estimated remaining time: {(now-self.start_)*(1-prop)/prop:.3f}s{" "*10}', end='\r')
        if self.counter_ == self.N_:
            print('')

CACHE_DIR = 'tmp/'

def cache(path):
    path = normpath(join(CACHE_DIR, path))
    def sub_cache(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if isfile(path):
                return np.load(path, allow_pickle=True)
            else:
                tmp = func(*args, **kwargs)
                np.save(path, tmp, allow_pickle=True)
                return tmp
        return wrapper
    return sub_cache
