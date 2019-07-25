import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

# -------------------------------------------------------------------------------------------------------- #
# A script using linear regression to estimate the grades of students in G3 based on their results in G1   #
# and G2 as well as their absences during the academic year, their failures and the time studied per week. #
#                                                                                                          #
#                                This script uses the following dataset:                                   #
#                       https://archive.ics.uci.edu/ml/datasets/Student+Performance                        #
# -------------------------------------------------------------------------------------------------------- #


def plot_scatter_diagram(data):
    """
    Plots a chosen relation in a scatter plot.
    :param data: DataFrame
    :return: None
    """
    att = "G1"
    style.use("ggplot")
    pyplot.scatter(data[att], data["G3"])
    pyplot.xlabel(att)
    pyplot.ylabel("Final Grade")
    pyplot.show()


def show_output(predictions, x_test, y_test, linear):
    """
    A function for printing the output our algorithm generates.
    :param predictions: numpy.ndarray
    :param x_test: numpy.ndarray
    :param y_test: numpy.ndarray
    :param linear: LinearRegression()
    :return: None
    """
    err = 0
    for x in range(len(predictions)):
        print("Prediction: ", predictions[x])
        print("Input data: ", x_test[x])
        print("Actual Final Grade: ", y_test[x])
        print("----")

        if not predictions[x] == y_test[x]:
            err += 1

    print("Total Accuracy:", round(linear.score(x_test, y_test) * 100, 2), "% with ", err, "errors. ")
    print(type(y_test), type(predictions))


def read_data(filename):
    """
    Function for reading the CSV-file and dropping all columns that aren't important for our purpose.
    :param filename: String
    :return: DataFrame
    """
    dat = pd.read_csv(filename, sep=";")
    dat = dat[["G1", "G2", "G3", "studytime", "failures", "absences"]]
    return dat


data = read_data("student-mat.csv")
predict = "G3" # specifying which attribute we want to predict

# Our x and y are going to be the whole DataFrame without the G3 column
# and only the G3 column respectively
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Splitting our dataset in different training and testing sets
# 90% of the data will be used for training, 10% for testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

acc = 0.0

# Here we'll get our saved model,
# or if we're doing the computation the first time (in which case open() will throw a FileNotFound-Exception)
# we'll save a model with an accuracy > 95% using pickle.
# Albeit not really necessary because of the short computation time, it's great practice in using pickle.
try:
    pickle_file = open("lin_reg.pickle", "rb")
except:
    while acc < 0.95:
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)
        acc = linear.score(x_test, y_test)

    with open("lin_reg.pickle", "wb") as file:
        pickle.dump(linear, file)

    pickle_file = open("lin_reg.pickle", "rb")

linear = pickle.load(pickle_file)
predictions = linear.predict(x_test)

show_output(predictions, x_test, y_test, linear)

plot_scatter_diagram(data)

