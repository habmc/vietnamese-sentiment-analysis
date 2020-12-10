import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import metrics


def train(x_train, y_train):
    C_OPTIONS = np.logspace(-9, 3, 15)
    text_clf = Pipeline([('cvec', CountVectorizer(ngram_range=(1,3))),
                         ('tfidf', TfidfTransformer(sublinear_tf=False)),
                         ('clf', LinearSVC(dual=False, tol=1e-3, penalty="l2", loss='squared_hinge')),])
    parameters = {
        'clf__C': C_OPTIONS,
    }

    score = 'accuracy'

    print("\nTuning parameters for Linear SVM\n")
    gs_clf = GridSearchCV(text_clf, parameters, cv=5, scoring=score,
                          n_jobs=-1)
    gs_clf.fit(x_train, y_train)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    print("Best score (Grid search): %f" % gs_clf.best_score_)

    clf1 = LinearSVC(
        C=gs_clf.best_params_['clf__C'],
        loss='squared_hinge',
        penalty="l2",
        dual=False, tol=1e-3)

    clf2 = SGDClassifier(alpha=0.0001, l1_ratio=0.1, loss='hinge', penalty="l2")

    print("\nEnsemble Model\n")

    clf3 = RandomForestClassifier(n_estimators=200, max_depth=None,
                                  random_state=100)

    ensemble_clf = VotingClassifier(
        estimators=[('linearSVM', clf1), ('sgd', clf2),
                    ('rdf', clf3),
                    ],
        voting='hard')

    text_clf = Pipeline([('cvec', CountVectorizer(ngram_range=(1,3))),
                         ('tfidf', TfidfTransformer(sublinear_tf=False)),
                         ('clf', ensemble_clf),])

    text_clf.fit(x_train, y_train)
    return text_clf


def predict(text_clf, test_sens):
    return text_clf.predict(test_sens)


if __name__ == "__main__":
    train_df = pd.read_csv("../data/train.csv")
    X_train, y_train = train_df.review, train_df.label
    test_df = pd.read_csv("../data/test.csv")
    X_val, y_val = test_df.review, test_df.label

    text_clf = train(X_train, y_train)
    preds = predict(text_clf, X_val)
    print(metrics.classification_report(y_val, preds))
