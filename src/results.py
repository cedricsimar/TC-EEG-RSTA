import numpy as np

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

from utils import cache, ProgressBar

def mkpipeline():
    return make_pipeline(
        XdawnCovariances(3, estimator='oas'),
        TangentSpace(),
        LogisticRegression(penalty='l2', C=1., max_iter=1000)
    )

@cache('fig4a.npy')
def compute_scores(dataset):
    print('Computing CV results...')
    remaining_electrodes = list(dataset.electrode_names)
    current = []
    nb_patients = dataset.nb_patients
    all_scores = np.zeros((dataset.nb_electrodes, nb_patients), dtype=np.float)
    idx = 0
    bar = ProgressBar(nb_patients * (dataset.nb_electrodes*(dataset.nb_electrodes+1))//2)
    while len(remaining_electrodes) > 0:
        results = np.empty((len(remaining_electrodes), dataset.nb_patients), dtype=np.float)
        for i, electrode in enumerate(remaining_electrodes):
            current.append(electrode)
            for j, cv_step in enumerate(dataset.loo()):
                cv_step = cv_step.filter_on_electrodes_by_name(current)
                pipeline = mkpipeline()
                pipeline.fit(cv_step.X_train, cv_step.y_train)
                results[i,j] = pipeline.score(cv_step.X_test, cv_step.y_test)
                bar.update()
            current.pop()
        best_idx = results.mean(axis=1).argmax()
        current.append(remaining_electrodes[best_idx])
        remaining_electrodes.pop(best_idx)
        all_scores[idx,:] = results[best_idx,:]
        idx += 1
    return all_scores


def get_scores_for_family(electrodes, dataset, bar=None):
    scores = np.empty(dataset.nb_patients, dtype=np.float)
    for i, cv_step in enumerate(dataset.loo()):
        cv_step = cv_step.filter_on_electrodes_by_name(electrodes)
        pipeline = mkpipeline()
        pipeline.fit(cv_step.X_train, cv_step.y_train)
        scores[i] = pipeline.score(cv_step.X_test, cv_step.y_test)
        if bar is not None:
            bar.update()
    return scores

@cache('fig4b.distributions.npy')
def _get_all_scores(dataset, area_names):
    print('Computing CV results per electrode family...')
    families = [[e for e in dataset.electrode_names if e.startswith(area[0].upper())] for area in area_names]
    distributions = np.empty((len(families), dataset.nb_patients), dtype=np.float)
    N = distributions.size
    bar = ProgressBar(N, N)
    for i, family in enumerate(families):
        distributions[i] = get_scores_for_family(family, dataset, bar)
    return distributions

def get_all_scores(dataset):
    area_names = ['Occipital', 'Parietal', 'Central', 'Frontal']
    return _get_all_scores(dataset, area_names), area_names

def get_cv_scores_inter(dataset):
    @cache(f'fig8_inter_{dataset.name}.npy')
    def _get_cv_scores_inter(dataset):
        print(f'Computing CV on {dataset.name} (inter-patient)')
        ret = np.empty(dataset.nb_patients, dtype=np.float)
        bar = ProgressBar(dataset.nb_patients, dataset.nb_patients)
        for i, cv_step in enumerate(dataset.loo()):
            pipeline = mkpipeline()
            pipeline.fit(cv_step.X_train, cv_step.y_train)
            ret[i] = pipeline.score(cv_step.X_test, cv_step.y_test)
            bar.update()
        return ret
    return _get_cv_scores_inter(dataset)

def do_kfold(pipeline, X, y, n_splits=4):
    X = np.asarray(X)
    y = np.asarray(y)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=0xCAFE)
    ret = 0
    for train_idx, test_idx in kfold.split(X):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        pipeline.fit(X_train, y_train)
        ret += pipeline.score(X_test, y_test)
    return ret / n_splits

def get_cv_scores_intra(dataset):
    @cache(f'fig8_intra_{dataset.name}.npy')
    def _get_cv_score_intra(dataset):
        print(f'Computing CV on {dataset.name} (intra-patient)')
        ret = np.empty(dataset.nb_patients, dtype=np.float)
        bar = ProgressBar(dataset.nb_patients, dataset.nb_patients)
        for i, cv_step in enumerate(dataset.loo()):
            pipeline = mkpipeline()
            ret[i] = do_kfold(pipeline, cv_step.X_test, cv_step.y_test)
            bar.update()
        return ret
    return _get_cv_score_intra(dataset)
