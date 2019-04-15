import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, auc
from sklearn.model_selection import train_test_split, cross_val_score

english_stop_words = stopwords.words('english')

class Data:

    def __init__(self):
        self.train_set = []
        self.test_set = []
        self.clean_train = []
        self.clean_test = []
        self.target = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

    def load_dataset(self):
        '''

        :return:
        '''
        for line in open('data/full_train.txt', 'r'):
            self.train_set.append(line.strip())

        for line in open('data/full_test.txt', 'r'):
            self.test_set.append(line.strip())
        self.create_target()

    def create_target(self):
        '''
        Create target list to do supervised classification
        :return:
        '''
        self.target = [1 if i < 12500 else 0 for i in range(25000)]

    def split_dataset(self, X=None, train_size=0.8):
        '''
        split dataset into train and validation set
        :param X:
        :param train_size:
        :return:
        '''
        if X is None:
            X = self.clean_train
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, self.target, train_size=train_size)


class Preprocesser:

    def __init__(self,data):
        self.data = data

    def regex_html(self):
        '''
        regex to remove html markdown
        :return:
        '''
        RE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        self.data.clean_train = [RE.sub(" ", line) for line in self.data.train_set]
        self.data.clean_test = [RE.sub(" ", line) for line in self.data.test_set]

    def regex_ponctuation(self):
        '''
        regex to remove punctuation sign
        :return:
        '''
        RE = re.compile("[.;:!\'?,\"()\[\]]")
        self.data.clean_train = [RE.sub(" ", line) for line in self.data.train_set]
        self.data.clean_test = [RE.sub(" ", line) for line in self.data.test_set]

    def regex_all(self, keep_ponctuation=False):
        if not keep_ponctuation:
            self.regex_ponctuation()
        self.regex_html()

    def remove_stop_words(self):
        '''
        remove stopwords from the train and test set
        :return:
        '''
        self.data.clean_train = list(map(lambda x: ' '.join([item for item in x.lower().split() if item not in english_stop_words]), self.data.clean_train))
        self.data.clean_test = list(map(lambda x: ' '.join([item for item in x.lower().split() if item not in english_stop_words]), self.data.clean_test))

    def stemmed_text(self):
        '''
        stemmed train and test set
        :return:
        '''
        stemmer = PorterStemmer()
        self.data.clean_train = [' '.join([stemmer.stem(word) for word in review.split()]) for review in self.data.clean_train]
        self.data.clean_test = [' '.join([stemmer.stem(word) for word in review.split()]) for review in self.data.clean_test]

    def lemmatized_text(self):
        '''
        lemmatize train and test set
        :return:
        '''
        lemmatizer = WordNetLemmatizer()
        self.data.clean_train = [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in self.data.clean_train]
        self.data.clean_test = [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in self.data.clean_test]



    def count_vectorization(self):
        '''
        Count vectorize the train and test set
        :return:
        '''
        wc_vectorizer = CountVectorizer(binary=False)
        wc_vectorizer.fit(self.data.clean_train)
        X = wc_vectorizer.transform(self.data.clean_train)
        X_test = wc_vectorizer.transform(self.data.clean_test)
        return X, X_test

    def tfidf_vectorization(self):
        '''
        TFIDF transformation
        :return:
        '''
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(self.data.clean_train)
        X = tfidf_vectorizer.transform(self.data.clean_train)
        X_test = tfidf_vectorizer.transform(self.data.clean_test)
        return X, X_test

    def ngram_vectorization(self):
        '''
        ngram transformation
        :return:
        '''
        ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
        ngram_vectorizer.fit(self.data.clean_train)
        X = ngram_vectorizer.transform(self.data.clean_train)
        X_test = ngram_vectorizer.transform(self.data.clean_test)
        return X, X_test


class Model:
    '''
    Patron pour chaque model que l'on va utiliser au cours du Test.
    Cette classe s'instancie avec un nom ainsi qu'un dataframe d'entrainement préalablement coupé en train et test set.
    '''

    def __init__(self, name, data):
        self.data = data
        self.params = None
        self.name = name
        self.score = None
        self.model = None

    def init_model(self, model):
        '''
        Initialise le parametre avec un Model de format sklearn.
        Ce model n'a pas forcement besoin d'etre fitted à ce point
        :param model: Sklearn model
        :return:
        '''
        self.model = model

    def fit_model(self):
        '''
        Fit le model que l'on a initialisé
        :return:
        '''
        try:
            self.model.fit(self.data.X_train, self.data.y_train)
        except:
            print('You must initialize the model before')

    def predict(self, X=None):
        if X is None:
            X = self.data.X_val
        return self.model.predict(X)

    def test_accuracy_model(self):
        '''
        Score accuracy sur le test set (que le modèle n'a jamais vu)
        :return:
        '''
        self.test_score = accuracy_score(self.data.y_val, self.predict())

    def roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.data.y_val, self.model.predict_proba(self.data.X_val)[:,1])
        return fpr, tpr, thresholds

    def confusion_matrix(self):
        conf = confusion_matrix(self.data.y_val, self.predict())
        return conf

    def cross_val_score(self, model=None, cv=5):
        '''
        Effectue une cross-validation sur un model donné avec une configuration pré-établie
        :param model: Il est possible de passer directement en parametre le model fitted
        :param cv: nombre de crossvalidation
        :return: renvoie un numpy array de taille nomber de cv
        '''
        if model is not None:
            self.model = model
        return cross_val_score(self.model, self.data.X_train, self.data.target, cv=cv)