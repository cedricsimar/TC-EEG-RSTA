import matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np

from results import *

rc('text',usetex=True)

def mkfig4(dataset):
    mkfig4a(dataset)
    mkfig4b(dataset)

def mkfig4a(dataset):
    print('Making figure 4A')
    all_scores_clf = 100*compute_scores(dataset)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    pos = np.arange(len(all_scores_clf))+1
    means_clf = all_scores_clf.mean(axis=1)
    c = 'sandybrown'
    C = 'chocolate'
    ax.boxplot(
        all_scores_clf.T, positions=pos, widths=.5,
        patch_artist=True, notch=False, showmeans=True,
        boxprops=dict(facecolor=c, color=c),
        capprops=dict(color=c),
        whiskerprops=dict(color=c),
        flierprops=dict(color=c, markeredgecolor=c, markersize=4, marker='o'),
        medianprops=dict(color=C),
        meanprops=dict(mfc=C, mec=C, marker='*', ms=7),
        zorder=-1,
    )
    ax.plot(pos, means_clf, ':', c='chocolate', ms=0, lw=1)
    ax.text(-.2, .95, 'A', fontsize=18, fontweight='bold', transform=ax.transAxes)
    ax.set_xlabel('Number of electrodes')
    ax.set_xticks(pos)
    ax.set_xticklabels(pos)
    ax.set_ylabel(r'Accuracy ($\%$)')
    ax.set_ylim(50, 100)
    plt.savefig(f'figs/fig4A.png', bbox_inches='tight', dpi=150)

def mkfig4b(dataset):
    print('Making figure 4B')
    distributions, area_names = get_all_scores(dataset)
    xs = np.arange(len(area_names))
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    c = 'sandybrown'
    ax.boxplot(
        100*np.asarray(distributions).T, positions=xs, widths=.5,
        patch_artist=True, notch=False,
        boxprops=dict(facecolor=c, color=c),
        capprops=dict(color=c),
        whiskerprops=dict(color=c),
        flierprops=dict(color=c, markeredgecolor=c, markersize=4, marker='o'),
        medianprops=dict(color=c),
        zorder=-1,
    )
    ax.plot(xs, 100*np.asarray(distributions).mean(axis=1), marker='*', ls=':', lw=0, c='chocolate')
    ax.set_xticks(xs)
    ax.set_xticklabels(area_names)
    ax.set_ylabel(r'Accuracy ($\%$)')
    ax.text(-.2, .95, 'B', fontsize=18, fontweight='bold', transform=ax.transAxes)
    plt.savefig('figs/fig4B.png', bbox_inches='tight', dpi=150)

def mkfig8(raw_dataset, sources_dataset):
    print('Making figure 8')
    Xs = np.array([
        get_cv_scores_inter(raw_dataset),
        get_cv_scores_intra(raw_dataset),
        get_cv_scores_inter(sources_dataset),
        get_cv_scores_intra(sources_dataset)
    ])
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    colours = [
        'sandybrown',
        'cornflowerblue',
        'indianred',
        'royalblue'
    ]
    C = 'black'
    for j, X in enumerate(Xs):
        c = colours[j]
        ax.boxplot(
            100*X, positions=[j], widths=.5,
            patch_artist=True, notch=False, showmeans=True,
            boxprops=dict(facecolor=c, color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c, markersize=4, marker='o'),
            medianprops=dict(color=c),
            meanprops=dict(mfc=C, mec=C, marker='*', ms=6),
            zorder=-1,
        )
    ax.set_xticks(range(4))
    ax.set_xticklabels([
        'EEG signals\n(inter-subject)',
        'EEG signals\n(intra-subject)',
        'Cortical sources\n(inter-subject)',
        'Cortical sources\n(intra-subject)'
    ])
    ymin = min(ax.get_ylim()[0], 50)
    eps = 2.5
    ax.set_ylim(ymin-eps, 100+eps)
    yticks = ax.get_yticks()
    yticks = yticks[np.where(np.logical_and(yticks >= ymin, yticks <= 100))[0]]
    ax.set_yticks(yticks)
    xlim = ax.get_xlim()
    for ytick in yticks:
        ax.plot(xlim, [ytick, ytick], c='lightgrey', ls=':', lw=1)
    ax.set_xlim(xlim)
    ax.set_ylabel(r'Accuracy (\%)')
    plt.savefig('figs/fig8.png', bbox_inches='tight', dpi=200)
