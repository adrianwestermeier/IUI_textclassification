import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score


def normalize_text(s):
    s = s.lower()

    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)

    # make sure we didn't introduce any double spaces
    s = re.sub('\s+', ' ', s)

    return s


class BayesianTrainer:
    def __init__(self, train_set):
        self.train_set = train_set

    def run_trainer(self):
        sns.set()  # use seaborn plotting style
        test_set = [
            "A 7-year-old boy named Eli saved his baby sister's life when "
            "he jumped into her room through a window to rescue her from "
            "a fire that destroyed his family's home.",
            " Muslim and Jewish paramedics pause to pray together.One of many inspiring moments "
             "in the coronavirus crisis ",
            "Adidas developing plant-based leather material that will be used to make shoes...material made "
             "from mycelium, which is part of fungus. Company produced 15 million pairs of shoes "
             "in 2020 made from recycled plastic waste collected from beaches and coastal regions.",
            "A new cancer therapy simultaneously zaps tumors with imaging-guided laser radiation and stimulates "
             "the anti-cancer immune response. This technology, developed by South Korean scientists, combines "
             "photodynamic therapy with immunotherapy for the treatment of cancer and is the first of its kind.",
            "Japan developing wooden satellites to cut space junk",
            "Two Korean-American women elected to congress for the first time ever."
        ]
        # grab the data
        news = self.train_set
        news['text'] = [normalize_text(s) for s in news['text']]
        # pull the data into vectors
        vectorizer = CountVectorizer()
        texts = news['text']
        x = vectorizer.fit_transform(texts)

        encoder = LabelEncoder()
        y = encoder.fit_transform(news['label'])

        # split into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

        # take a look at the shape of each of these
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)

        nb = MultinomialNB()
        nb.fit(x_train, y_train)

        print('score ', nb.score(x_test, y_test))

        predictions = nb.predict(x_test)
        # encoder.inverse_transform(predictions)

        test_df = pd.Series(test_set)
        el_vec = vectorizer.transform(test_df)
        predictions_custom = nb.predict(el_vec)
        print(encoder.inverse_transform(predictions_custom))

        # plot the confusion matrix
        mat = confusion_matrix(y_test, predictions)
        sns.heatmap(mat.T, square=True, annot=True, fmt="d", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.xlabel("true labels")
        plt.ylabel("predicted label")
        plt.show()
        # plt.savefig('baysian_confmatrix.png')
        plt.close()
        print("The accuracy is {}".format(accuracy_score(y_test, predictions)))

        return 0


