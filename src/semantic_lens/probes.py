"""Train/validation-only selection; test scores are produced only in evaluate."""
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from .data import prompt
from .metrics import metrics, weights

CS = (.01, .1, 1., 10.)


def fit_classifier(x, y, weights_, c, seed):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        return LogisticRegression(C=c, max_iter=3000, random_state=seed).fit(
            x, y, sample_weight=weights_)


def selection_key(result):
    return tuple(result[k] if result[k] is not None else -1 for k in ('pair_accuracy', 'auroc', 'balanced_accuracy'))


def fit_dense(train, val, x_train, x_val, seed=0, shuffle=False):
    y = np.array([r['outcome'] for r in train])
    if set(y) != {0, 1} or {r['outcome'] for r in val} != {0, 1}:
        raise ValueError('Train and validation require both outcomes')
    if shuffle:
        y = np.random.default_rng(seed).permutation(y)
    w = weights(train)
    best, sweep = None, []
    for layer in range(x_train.shape[1]):
        scaler = StandardScaler().fit(x_train[:, layer], sample_weight=w)
        a, b = scaler.transform(x_train[:, layer]), scaler.transform(x_val[:, layer])
        for c in CS:
            clf = fit_classifier(a, y, w, c, seed)
            measured = metrics(val, clf.decision_function(b))
            sweep.append(dict(layer=layer, C=c, validation=measured))
            key = selection_key(measured)
            if best is None or key > best[0]:
                raw_w = clf.coef_[0] / scaler.scale_
                raw_b = float(clf.intercept_[0] - raw_w @ scaler.mean_)
                best = (key, dict(kind='dense', layer=layer, C=c, w=raw_w, b=raw_b,
                                  shuffle=shuffle, seed=seed))
    return best[1], sweep


def fit_lexical(train, val, condition, seed=0):
    features = FeatureUnion([
        ('tokens', TfidfVectorizer(token_pattern=r'\w+|[^\w\s]', lowercase=False,
                                  ngram_range=(1, 2), max_features=25000)),
        ('characters', TfidfVectorizer(analyzer='char', ngram_range=(3, 5),
                                      lowercase=False, max_features=25000))])
    x = features.fit_transform([prompt(r, condition) for r in train])
    v = features.transform([prompt(r, condition) for r in val])
    y = [r['outcome'] for r in train]; best, sweep = None, []
    for c in CS:
        clf = fit_classifier(x, y, weights(train), c, seed)
        result = metrics(val, clf.decision_function(v)); key = selection_key(result)
        sweep.append(dict(C=c, validation=result))
        if best is None or key > best[0]:
            best = key, dict(kind='lexical', condition=condition, features=features, classifier=clf, C=c)
    return best[1], sweep


def predict(model, rows, x=None):
    if model['kind'] == 'dense':
        if x is None or x.shape[0] != len(rows):
            raise ValueError('Activation/row mismatch')
        return x[:, model['layer']] @ model['w'] + model['b']
    return model['classifier'].decision_function(model['features'].transform([prompt(r, model['condition']) for r in rows]))
