import pickle
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from itertools import product
from sklearn.metrics import precision_recall_curve, average_precision_score

DATA_DIR = '../data/'

def draw_roc_curve(plot_infos, title="Receiver Operating Characteristic", letter=None, letter_size=40, save_title='Test'):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()

    for y_prob, y_test, color, label in plot_infos:
        
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        ax.plot(false_positive_rate, true_positive_rate, lw=2, color=color, label = label + ' (AUC = %0.2f)' % roc_auc)

    ax.legend(loc = 'lower right', fontsize=21)
    ax.plot([0, 1], [0, 1],linestyle='--', color='black')
    ax.axis('tight')
    ax.set_ylabel('True Positive Rate', fontsize=26)
    ax.set_xlabel('False Positive Rate', fontsize=26)
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)

    if letter is not None:
        ax.text(-0.18, 0.95, letter, size=letter_size, weight='bold', transform=ax.transAxes)

    plt.savefig("./figs/" + save_title + '.png', dpi=300, bbox_inches='tight')
    plt.close()


def draw_precision_recall(plot_infos, title="Precision Recall Curve", letter=None, letter_size=40, save_title='Test', legend_pos='lower left'):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    
    for y_prob, y_test, color, label in plot_infos:
        
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        average_precision = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, color=color, lw=2, label = label + ' (AP = %0.2f)' % average_precision)

    ax.legend(loc=legend_pos, fontsize=21)
    # plt.plot([0, 1], [0, 1],linestyle='--')
    ax.axis('tight')
    ax.set_ylabel('Precision', fontsize=26)
    ax.set_xlabel('Recall', fontsize=26)
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)

    if letter is not None:
        ax.text(-0.18, 0.95, letter, size=letter_size, weight='bold', transform=ax.transAxes)
    
    plt.savefig("./figs/" + save_title + '.png', dpi=300, bbox_inches='tight')
    plt.close()


def draw_confusion_matrix(y_prob, y_test, ax=None, fig=None, normalize='true', include_values=True, cmap=plt.cm.Blues, values_format=None, colorbar=False, xticks_rotation='horizontal', letter=None, letter_size=20):

    y_prob = np.round(y_prob)
    cm = confusion_matrix(y_test, y_prob, normalize=normalize)
    display_labels = ["Tunnel", "Checkerboard"]     # Tunnel: 0 - Checkerboard: 1

    if ax is None:
        fig, ax = plt.subplots()

    n_classes = cm.shape[0]
    im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    text_ = None
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    if include_values:

        text_ = np.empty_like(cm, dtype=object)

        # Print the text with an appropriate color depending on the background
        thresh = (cm.max() + cm.min()) / 2.0

        for i, j in product(range(n_classes), range(n_classes)):
            color = cmap_max if cm[i, j] < thresh else cmap_min

            if values_format is None:
                text_cm = format(cm[i, j], '.2f')
                if cm.dtype.kind != 'f':
                    text_d = format(cm[i, j], 'd')
                    if len(text_d) < len(text_cm):
                        text_cm = text_d
            else:
                text_cm = format(cm[i, j], values_format)

            text_[i, j] = ax.text(
                j, i, text_cm,
                ha="center", va="center",
                color=color, fontsize=14)

    if colorbar:
        fig.colorbar(im_, ax=ax)
    
    ax.set(xticks=np.arange(n_classes), yticks=np.arange(n_classes))

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

    for tick in ax.get_yticklabels():
        tick.set_rotation('vertical')
        tick.set_va('center')

    ax.set_yticklabels(display_labels, fontsize=10)
    ax.set_xticklabels(display_labels, fontsize=10)
    ax.set_ylabel("True labels", fontsize=14, labelpad=30)

    return ax, im_
    

def draw_confusion_matrix_figure(plot_infos, title="Precision Recall Curve", letter=None, letter_size=20, save_title='Test'):

    fig = plt.figure(figsize=(10,20))
    grid = ImageGrid(fig, 111, nrows_ncols=(1,2), axes_pad=0.50, cbar_location="right", cbar_mode="single", cbar_size="7%", cbar_pad=0.20)

    for i, ax in enumerate(grid[:2]):

        y_prob, y_test = plot_infos[i]
        ax, im = draw_confusion_matrix(y_prob, y_test, ax, fig)

        if not i and letter is not None:
            ax.text(-.45, .9, letter, transform=ax.transAxes, size=letter_size, weight='bold')
            ax.text(0.82, -0.20, "Predicted labels", size=14, transform=ax.transAxes)


    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    fig.colorbar(im, cax=ax.cax)

    plt.savefig("./figs/" + save_title + '.png', dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":    

    with open(join(DATA_DIR, "full_pred_labels_results.pickle"), "rb") as handle:
        results = pickle.load(handle)

    # Figure 5
    draw_roc_curve([(results["Raw"]["Inter-patients"][0], results["Raw"]["Inter-patients"][1], 'sandybrown', 'T/C Inter-subject'), (np.concatenate(results["Raw"]["Intra-patients"][0], axis=0), np.concatenate(results["Raw"]["Intra-patients"][1], axis=0), 'cornflowerblue', 'T/C Intra-subject')], letter='A', save_title='Figure7A_new')
    draw_precision_recall([(results["Raw"]["Inter-patients"][0], results["Raw"]["Inter-patients"][1], 'sandybrown', 'T/C Inter-subject'), (np.concatenate(results["Raw"]["Intra-patients"][0], axis=0), np.concatenate(results["Raw"]["Intra-patients"][1], axis=0), 'cornflowerblue', 'T/C Intra-subject')], letter='B', save_title='Figure7B_new')
    draw_confusion_matrix_figure([(results["Raw"]["Inter-patients"][0], results["Raw"]["Inter-patients"][1]), (np.concatenate(results["Raw"]["Intra-patients"][0], axis=0), np.concatenate(results["Raw"]["Intra-patients"][1], axis=0))], letter='C', save_title='Figure7C_new')

    # Figure 6
    draw_roc_curve([(results["Sources"]["Inter-patients"][0], results["Sources"]["Inter-patients"][1], 'indianred', 'T/C Inter-subject'), (np.concatenate(results["Sources"]["Intra-patients"][0], axis=0), np.concatenate(results["Sources"]["Intra-patients"][1], axis=0), 'royalblue', 'T/C Intra-subject')], letter='A', save_title='Figure8A_new')
    draw_precision_recall([(results["Sources"]["Inter-patients"][0], results["Sources"]["Inter-patients"][1], 'indianred', 'T/C Inter-subject'), (np.concatenate(results["Sources"]["Intra-patients"][0], axis=0), np.concatenate(results["Sources"]["Intra-patients"][1], axis=0), 'royalblue', 'T/C Intra-subject')], letter='B', save_title='Figure8B_new', legend_pos='lower right')
    draw_confusion_matrix_figure([(results["Sources"]["Inter-patients"][0], results["Sources"]["Inter-patients"][1]), (np.concatenate(results["Sources"]["Intra-patients"][0], axis=0), np.concatenate(results["Sources"]["Intra-patients"][1], axis=0))], letter='C', save_title='Figure8C_new')

    # Figure 12
    draw_roc_curve([(results["Raw"]["Inter-patients"][0], results["Raw"]["Inter-patients"][1], 'sandybrown', 'Tunnel/Checkerboard'), (results["Grey"]["Inter-patients"][0], results["Grey"]["Inter-patients"][1], 'slategrey', 'Grey screens')], letter='A', save_title='Figure7A')
    draw_precision_recall([(results["Raw"]["Inter-patients"][0], results["Raw"]["Inter-patients"][1], 'sandybrown', 'Tunnel/Checkerboard'), (results["Grey"]["Inter-patients"][0], results["Grey"]["Inter-patients"][1], 'slategrey', 'Grey screens')], letter='B', save_title='Figure7B')
    draw_confusion_matrix_figure([(results["Raw"]["Inter-patients"][0], results["Raw"]["Inter-patients"][1]), (results["Grey"]["Inter-patients"][0], results["Grey"]["Inter-patients"][1])], letter='C', save_title='Figure7C')
    
    # Figure 13
    draw_roc_curve([(np.concatenate(results["Raw"]["Intra-patients"][0], axis=0), np.concatenate(results["Raw"]["Intra-patients"][1], axis=0), 'sandybrown', 'Tunnel/Checkerboard'), (np.concatenate(results["Grey"]["Intra-patients"][0], axis=0), np.concatenate(results["Grey"]["Intra-patients"][1], axis=0), 'slategrey', 'Grey screens')], letter='A', save_title='Figure8A')
    draw_precision_recall([(np.concatenate(results["Raw"]["Intra-patients"][0], axis=0), np.concatenate(results["Raw"]["Intra-patients"][1], axis=0), 'sandybrown', 'Tunnel/Checkerboard'), (np.concatenate(results["Grey"]["Intra-patients"][0], axis=0), np.concatenate(results["Grey"]["Intra-patients"][1], axis=0), 'slategrey', 'Grey screens')], letter='B', save_title='Figure8B')
    draw_confusion_matrix_figure([(np.concatenate(results["Raw"]["Intra-patients"][0], axis=0), np.concatenate(results["Raw"]["Intra-patients"][1], axis=0)), (np.concatenate(results["Grey"]["Intra-patients"][0], axis=0), np.concatenate(results["Grey"]["Intra-patients"][1], axis=0))], letter='C', save_title='Figure8C')
