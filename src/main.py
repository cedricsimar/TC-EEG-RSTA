from sys import argv

from datasets import *
from figures import *

if __name__ == '__main__':
    dataset = load_data()
    sources = load_data_sources()
    if 'fig4' in argv:
        mkfig4(dataset)
    if 'fig8' in argv:
        mkfig8(dataset, sources)
