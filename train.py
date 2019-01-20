import process
import numpy as np

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

def lr_cv():
    x, y = process.read_data("train.csv", 1, 1)
    x_train_preprocessed = preprocessing.scale(x)
    y_train = y.T
    best_score = 0.0
    num = 0
    for C in [0.12, 0.14, 0.16]:
        num = num + 1
        lr = LogisticRegression(C=C, random_state=1, solver='saga',
                                multi_class='multinomial', max_iter=1000, penalty='l2')
        scores = cross_val_score(lr, x_train_preprocessed, y_train, cv=3)
        score = scores.mean()
        print("Iteration time:{}th".format(num))
        print("Current score on validation set:{:.9f}".format(score))
        print("Current parameters:{:.4f}".format(C))
        if score > best_score:
            best_score = score
            best_parameters = {"C": C}
    print("Best score on validation set:{:.9f}".format(best_score))
    print("Best parameters:{}".format(best_parameters))

def lr():
    x, y = process.read_data("train.csv", 1, 1)
    x_train_preprocessed = preprocessing.scale(x)
    x_test, y_test = process.read_data("test.csv", 1, 0)
    x_test_preprocessed = preprocessing.scale(x_test)
    y_train = y.T
    lr = LogisticRegression(C=0.05, random_state=1, solver='saga',
                            multi_class='multinomial', max_iter=800)
    lr.fit(x_train_preprocessed, y_train)
    return lr, x_test_preprocessed

def mnb_cv():
    x, y = process.read_data("train.csv", 1, 1)
    y_train = y.T
    best_score = 0.0
    num = 0
    for alpha in [18.0, 18.2, 18.4, 18.6, 18.8]:
        num = num + 1
        mnb = MultinomialNB(alpha=alpha, fit_prior=False)
        scores = cross_val_score(mnb, x, y_train, cv=4)
        score = scores.mean()
        print("Iteration time:{}th".format(num))
        print("Current score on validation set:{:.9f}".format(score))
        print("Current parameters:{:.2f}".format(alpha))
        if score > best_score:
            best_score = score
            best_parameters = {"alpha": alpha}
    print("Best score on validation set:{:.9f}".format(best_score))
    print("Best parameters:{}".format(best_parameters))

def mnb():
    x, y = process.read_data("train.csv", 1, 1)
    x_test, y_test = process.read_data("test.csv", 1, 0)
    y_train = y.T
    mnb = MultinomialNB(alpha=19.6, fit_prior=False)
    mnb.fit(x,y_train)
    return mnb, x_test

def svm_train_cv():
    x, y = process.read_data("train.csv", 1, 1)
    sscaler = StandardScaler()
    sscaler.fit(x)
    x_train_preprocessed = sscaler.transform(x)
    y_train = y.T
    best_score = 0.0
    num = 0
    for coef0 in [0.1, 0.2, 0.3, 0.4]:
        svm_rbf = svm.SVC(C=0.90, cache_size=500, kernel='sigmoid', gamma='scale', coef0=coef0)
        num = num + 1
        scores = cross_val_score(svm_rbf, x_train_preprocessed, y_train, cv=3)
        score = scores.mean()
        print("Iteration time:{}th".format(num))
        print("Current score on validation set:{:.9f}".format(score))
        print("Current parameters:{:.2f}".format(coef0))
        if score > best_score:
            best_score = score
            best_parameters = {"coef0": coef0}
    print("Best score on validation set:{:.9f}".format(best_score))
    print("Best parameters:{}".format(best_parameters))

def svm_train():
    x, y = process.read_data("train.csv", 1, 1)
    print(x)
    print(y)
    x_test, y_test = process.read_data("test.csv", 1, 0)
    sscaler = StandardScaler()
    sscaler.fit(x)
    x_train_preprocessed = sscaler.transform(x)
    x_test_preprocessed = sscaler.transform(x_test)
    y_train = y.T
    svm_rbf = svm.SVC(C=0.90, cache_size=500, kernel='sigmoid', gamma='auto', coef0=0.1)
    svm_rbf.fit(x_train_preprocessed, y_train)
    return svm_rbf,x_test_preprocessed

def predict(model, x_test):
    y_pred = model.predict(x_test)
    index = np.arange(y_pred.shape[0]).reshape(y_pred.shape[0], 1)
    return np.column_stack((index, y_pred))

def main():
    #svm_train_cv()
    #clf, x_test = svm_train()
    #y_pred = svm_predict(clf, x_test)
    #process.write_data(y_pred, "Submission.csv")
    #model, x_test =mnb()
    #lr_cv()
    model, x_test = lr()
    y_pred = predict(model, x_test)
    process.write_data(y_pred, "submission.csv")


if __name__ == '__main__':
    main()