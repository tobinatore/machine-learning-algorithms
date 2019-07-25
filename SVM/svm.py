import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

# ---------------------------------------------------------------------------------------------------------- #
# A script using a support-vector-machine to classify tumors as benign or malign, based on about 30 criteria #
# such as size, growth and many more.                                                                        #
#                                                                                                            #
#                                 This script uses the following dataset:                                    #
#                                   Sklearn's own breast cancer dataset                                      #
# ---------------------------------------------------------------------------------------------------------- #


def show_output(prediction, y_test, acc):
    """
    A function for printing the output of our algorithm.
    :param prediction: numpy.ndarray
    :param y_test: numpy.ndarray
    :param acc: numpy.float64
    :return: None
    """
    err = 0
    for x in range(len(prediction)):
        print("Predicted Classification: ", classes[prediction[x]])
        print("Actual Classification: ", classes[y_test[x]])
        print("----")

        if not y_test[x] == prediction[x]:
            err += 1

    print("Total accuracy: ", round(acc * 100, 2), "% with ", err, " errors.")


# Loading the dataset directly from sklearn
cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target

# Splitting the data in 90% training data and 10% test data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

classes = ['malignant', 'benign']

# Defining our classifier as a support-vector-classification
# using a linear kernel and a penalty parameter of 2
classifier = svm.SVC(kernel="linear", C=2)

classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)

acc = metrics.accuracy_score(y_test, prediction)

show_output(prediction, y_test, acc)
