# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

categories = ['education', 'finance', 'game', 'social', 'weather']

# Transform data into training and testing datasets
data = pd.read_csv('allReviews.csv')
data.columns = ['reviews', 'category']

data['reviews'] = data['reviews'].str.replace('[^0-9a-zA-Z ]+', ' ')
textLen = [len(a) for a in data.reviews]
newData = pd.DataFrame({'reviews':data.reviews, 'nChars': textLen, 'category':data.category})
newData = newData[newData.nChars >= 50]

rd = np.random.rand(len(newData)) < 0.7
train = newData[rd]
test = newData[-rd]

y_train = train['category']
y_test = test['category']
x_train = train['reviews']
x_test = test['reviews']

# Create tf-idf matrix
vectorizer = TfidfVectorizer(strip_accents = 'ascii', max_features = 30000, decode_error = 'ignore', stop_words = 'english') 
X = vectorizer.fit_transform(x_train)

# Get tuple (number of rows in training data, number of features)
# X.shape

# Build classifier
#model = MultinomialNB().fit(X, y_train)  #Naive Bayes
#model = RandomForestClassifier().fit(X, y_train)  #Random Forest
model = LinearSVC().fit(X, y_train)  #Linear Support Vector Classifier

predictTfidf = vectorizer.transform(x_test)
predicted = model.predict(predictTfidf)


# Printing of confusion matrix
# Reference: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predicted)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=categories,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=categories, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

print("Accuracy: " + str((cnf_matrix[0][0] + cnf_matrix[1][1] + cnf_matrix[2][2] + cnf_matrix[3][3] + cnf_matrix[4][4])/x_test.shape[0]))
