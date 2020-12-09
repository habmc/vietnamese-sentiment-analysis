from __future__ import print_function
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import pandas as pd

train_data = pd.read_csv("../data/train.csv")

#Try some models
classifiers = [
            LinearSVC(fit_intercept = True,multi_class='crammer_singer', C=1),
        ]

X_train, X_test, y_train, y_test = train_test_split(train_data.review, train_data.label, test_size=0.3,random_state=42)
validation_data = pd.read_csv("../data/test.csv")
X_val, y_val = validation_data.review, validation_data.label

for classifier in classifiers:
    steps = []
    steps.append(('CountVectorizer', CountVectorizer(ngram_range=(1,5),max_df=0.5, min_df=5)))
    steps.append(('tfidf', TfidfTransformer(use_idf=False, sublinear_tf = True,norm='l2',smooth_idf=True)))
    steps.append(('classifier', classifier))
    clf = Pipeline(steps)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report1 = metrics.classification_report(y_test, y_pred, labels=[1,0], digits=3)

X_train, y_train = train_data.review, train_data.label


#TRAIN OVERFITTING/ERROR ANALYSIS
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_train)
# report2 = metrics.classification_report(y_train, y_pred, labels=[1,0], digits=3)
# print(report2)


#ERROR ANALYSIS
# for id,x, y1, y2 in zip(train_data.id, X_train, y_train, y_pred):
#     if y1 != y2:
#         # CHECK EACH FALSE POSSITIVE/NEGATIVE
#         if y1!=1:#0:
#             print(id,x, y1, y2)

#CROSS VALIDATION
cross_score = cross_val_score(clf, X_train,y_train, cv=5)

#REPORT
print('DATASET LENGTH: %d'%(len(X_train)))
print('TRAINING RESULT \n\n',report1)
# print('TRAIN OVERFITING\n\n',report2)
print("CROSSVALIDATION 5 FOLDS: %0.4f (+/- %0.4f)" % (cross_score.mean(), cross_score.std() * 2))

# Test on validation set
print("VALIDATION RESULT")
print(metrics.classification_report(y_val, clf.predict(X_val)))
