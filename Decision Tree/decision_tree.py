from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from subprocess import call
from sklearn import metrics
import pandas as pd
import os


# ---------------------------------------------------------------------------------------------------------- #
#                   A script using a decision tree to predict the onset of diabetes based                    #
#                                           on diagnostic measures.                                          #
#                                                                                                            #
#                                  This script uses the following dataset:                                   #
#                    https://gist.github.com/chaityacshah/899a95deaf8b1930003ae93944fd17d7                   #
# ---------------------------------------------------------------------------------------------------------- #


def visualize(classifier):
    # Exporting the classifier as dot file, GraphViz is required to make a .png from that .dot file
    export_graphviz(classifier, out_file='tree.dot',
                    rounded=True, proportion=False,
                    precision=2, filled=True,
                    special_characters=True, feature_names=feature_cols, class_names=['0', '1'])

    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    os.remove("tree.dot")
    print("Image of the decision tree has been saved as 'tree.png'!")


col_names = ['pregnancies', 'glucose', 'bp', 'skin_thickness', 'insulin', 'bmi', 'pedigree', 'age', 'label']

# load data
pima = pd.read_csv("pima-indians-diabetes.csv", header=None, names=col_names)

# split data in features and target variable
feature_cols = ['pregnancies', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']

# features:
X = pima[feature_cols]
# target variable:
y = pima.label

# Splitting the data in 70% training data and 30% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Creating the classifier, which will use information gain as attribute selection measure
# and limiting the tree to a maximum depth of 4
classifier = DecisionTreeClassifier(criterion="entropy", max_depth=4)

classifier = classifier.fit(X_train, y_train)

# Predict the response for test data
y_pred = classifier.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)

print("Accuracy: Predicted " + str(acc*100) + "% of cases correctly.")

visualize(classifier)

