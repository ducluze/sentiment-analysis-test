from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from synthesio_classes import *


if __name__ == '__main__':
    data = Data()
    data.load_dataset()

    preprocess = Preprocesser(data)
    preprocess.regex_all()
    preprocess.remove_stop_words()
    preprocess.stemmed_text()
    preprocess.lemmatized_text()

    X, X_test = preprocess.ngram_vectorization()
    data.split_dataset(X)

    model = Model('logistic_regression', data)
    model.init_model(LogisticRegression(C=0.05))
    model.fit_model()
    model.test_accuracy_model()
    print(model.test_score)
    fpr, tpr, _ = model.roc_curve()