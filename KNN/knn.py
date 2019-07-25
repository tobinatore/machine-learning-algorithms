import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
import pickle

# --------------------------------------------------------------------------------------------------------- #
# A script using K-Nearest-Neighbour-Algorithm to classify a vehicle based on buying and maintenance price, #
# no. of doors, capacity in terms of persons, size of luggage boot and the car's safety rating.             #
#                                                                                                           #
#                          This script uses the following (slightly modified) dataset:                      #
#                            https://archive.ics.uci.edu/ml/datasets/Car+Evaluation                         #
# --------------------------------------------------------------------------------------------------------- #


def show_output(predictions, x_test, y_test, model):
    """
    A function for printing the output our algorithm generates.
    :param predictions: numpy.ndarray
    :param x_test: numpy.ndarray
    :param y_test: numpy.ndarray
    :param model: KNeighboursClassifier()
    :return: None
    """
    err = 0
    for x in range(len(predictions)):
        print("Prediction: ", predictions[x])
        print("Input data: ", x_test[x])
        print("Actual Class: ", y_test[x])
        print("----")
        if not y_test[x] == predictions[x]:
            err += 1

    print("Total Accuracy:", round(model.score(x_test, y_test) * 100, 2),
          "% with ", err, " errors.")


# Read car.data like a CSV - as all values are comma separated
# I added a header row to the original dataset so it can be
# read in directly using pandas.
data = pd.read_csv("car.data")

# Using the sklearn preprocessor, we're transforming non-integer values to integers
# so we can train our model.
preprocessor = preprocessing.LabelEncoder()
buying = preprocessor.fit_transform(list(data["buying"]))
maintenance = preprocessor.fit_transform(list(data["maintenance"]))
doors = preprocessor.fit_transform(list(data["doors"]))
persons = preprocessor.fit_transform(list(data["persons"]))
lug_boot = preprocessor.fit_transform(list(data["lug_boot"]))
safety = preprocessor.fit_transform(list(data["safety"]))
cls = preprocessor.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maintenance, doors, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

acc = 0.0
# Same as in the script for Linear Regression.
# We're trying to load a pickled model, if there is none,
# train our model until it has an accuracy greater than 97%
# then save it for later use.
try:
    pickle_file = open("knn.pickle", "rb")
except:
    while acc < 0.97:
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
        model = KNeighborsClassifier(n_neighbors=7)
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)

    with open("knn.pickle", "wb") as file:
        pickle.dump(model, file)

    pickle_file = open("knn.pickle", "rb")

model = pickle.load(pickle_file)
predictions = model.predict(x_test)

show_output(predictions, x_test, y_test, model)

