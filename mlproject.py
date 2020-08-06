# importing all the required libraries for this project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import regularizers
import seaborn as sn
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier


## pre-processing of datasets

def readDataSet():
    # used the global keyword so that the vairables could be used throughout the program
    global test_data, train_data, xtest, ytest, xtrain, ytrain
    # test dataset is being stored in test_data and train dataset is being stored in train_data
    test_data = pd.read_csv("poker-hand-testing.data", header=None)
    train_data = pd.read_csv("poker-hand-training.data", header=None)
    # adding column names to noth our datasets.
    col = ['Suit of card #1', 'Rank of card #1', 'Suit of card #2', 'Rank of card #2', 'Suit of card #3',
           'Rank of card #3', 'Suit of card #4', 'Rank of card #4', 'Suit of card #5', 'Rank of card #5', 'PokerHand']
    train_data.columns = col
    test_data.columns = col
    # splitting the input columns and output column in test and train variables.
    xtest = test_data.drop('PokerHand', axis=1)
    ytest = test_data['PokerHand']
    xtrain = train_data.drop('PokerHand', axis=1)
    ytrain = train_data['PokerHand']


# this function used by each model to calculate the testing accuracy
def accuracy(cm):
    de = cm.trace()
    ae = cm.sum()
    return de / ae


# this function shows the shape and the ouput labels with there number of instances of train and test datasets
def displaydataset(ds):
    global datasize, cls
    # ds.shape show the number of rows and columns of dataset.
    datasize = ds.shape

    PH_classes = 10
    cls = {}
    # storing the instances of each poker-hand in cls
    for i in range(PH_classes):
        cls[i] = len(ds[ds.PokerHand == i])
    # plotting a histogram using matplotlib that shows the division of pokerhands (labels)
    poker_hands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.bar(poker_hands, [cls[i] for i in poker_hands], align='center')
    plt.xlabel('Poker hand id')
    plt.ylabel('Number of instances')
    plt.show()


# designing a model for keras neural network having an input layer two hidden layers of 30 and 15 output unit
# and a output layer of 10 because total number of labels(poker hands) are 10.
def designKerasNN():
    global Poker_model
    Poker_model = Sequential()
    Poker_model.add(Dense(50, activation='relu', input_dim=10))
    Poker_model.add(Dense(30, activation='relu'))
    Poker_model.add(Dense(15, activation='relu'))
    Poker_model.add(Dense(10, activation='softmax'))
    # we used categorical_crossentropy as a loss function because the dataset is of multi-class , and we used adam
    # optimizer
    Poker_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# this function is used to draw the loss and accuracy graph of test and validation dataset.
def showAccLoss():
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(Poker_graph.history['accuracy'])
    plt.plot(Poker_graph.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot(Poker_graph.history['loss'])
    plt.plot(Poker_graph.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower left')
    plt.show()


# this function is called to train the keras model using the .fit() method
def trainkeras():
    # calling the function to design and compile the keras neural network
    designKerasNN()

    kytrain = pd.get_dummies(ytrain)

    global Poker_graph
    # now we train the model we pass in the xtrain and ytrain output and input columns we used 20 epochs because it
    # reaches its maximum train accuracy of about 98-99 and we not need to increase the epochs size we divide our
    # train dataset by 80% to 20% using the 20% to use as validation set. we set batchsize=256 as the train dataset
    # is quite large so setting the batch size can train the model faster. verbose is set to 2 so that we can see the
    # training process of each epochs. shuffle is set true to that dataset is trained randomly.
    Poker_graph = Poker_model.fit(xtrain, kytrain, epochs=20, validation_split=0.2, batch_size=256, verbose=2,
                                  shuffle=True)
    # after training we call the function to plot the accuracy and loss graph
    showAccLoss()


# we use the heatmap to graph the predicted data of each training model in this project to see how many predictions
# were made correctly.

def HeatMap(cm, title):
    pk_cm = pd.DataFrame(cm, columns=['Nothing in hand', 'One pair', 'two pair', 'Three of a kind', 'Straight', 'Flush',
                                      'Full house', 'Four of a kind', 'straight flush', 'royal flush'],
                         index=['Nothing in hand', 'One pair', 'two pair', 'Three of a kind', 'Straight', 'Flush',
                                'Full house', 'Four of a kind', 'straight flush', 'royal flush'])
    plt.figure(figsize=(10, 7))
    plt.title(title)
    sn.set(font_scale=1)
    sn.heatmap(pk_cm, annot=True, annot_kws={"size": 11}, fmt='g')
    plt.show()


# using this function testing of keras model is done by .predict() method
def testkeras():
    kytest = pd.get_dummies(ytest)
    # then  pass in the xtest into .predict method and we get the predicted output labels.
    predictions = Poker_model.predict(xtest)
    roundP = predictions
    # then by using the sklearn confusion_matrix we compare the predicted output column of test data set with the
    # original output column of test dataset. it gives us the result in form of a matrix which we plot using the
    # heatmap function above and we then pass cm in the accuracy function above to calculate accuracy
    cm = confusion_matrix(kytest.values.argmax(axis=1), roundP.argmax(axis=1))

    acc[0] = (accuracy(cm) * 100)
    HeatMap(cm, 'keras_heatmap')


# below are the three types of naiive buyes classifier another machine learning model that we use to train and test
# our datasets on to see how much accuracy we can obtain from these classifiers.

# first is the bernoulli naiive bayes classifier.

def bernoulliNBC():
    bernNB = BernoulliNB()
    bernNB.fit(xtrain, ytrain)
    BNBpred = bernNB.predict(xtest)
    cm = confusion_matrix(ytest, BNBpred)

    acc[1] = (accuracy(cm) * 100)
    HeatMap(cm, 'bernoulli_heatmap')


# secoundly we use the multinomial naiive byes classifier.
def multinomialNBC():
    multiNB = MultinomialNB()
    multiNB.fit(xtrain, ytrain)
    MNBpred = multiNB.predict(xtest)
    cm = confusion_matrix(ytest, MNBpred)

    acc[2] = (accuracy(cm) * 100)
    HeatMap(cm, 'multinomial_heatmap')


# thirdly we use the gaussian naiive buyes classifier
def gaussianNBC():
    gausNB = GaussianNB()
    gausNB.fit(xtrain, ytrain)
    GNBpred = gausNB.predict(xtest)
    cm = confusion_matrix(ytest, GNBpred)

    acc[3] = (accuracy(cm) * 100)
    HeatMap(cm, 'gaussian_heatmap')


# our third machine learning training model is logistic regression. i used to sklearn library to import logistic
# regression methods to use the logistic regression algorithms on the training model.
def LogReg():
    # first create the logistic regression model , we set the solver as lbfgs as it handles the multi-class problem
    # we set max-iteration as 100 because even if increased accuracy obtained does not change so at 100 we obtain the
    # max accuracy. multi_class is set to ovr by default which is used for multi-class datasets. and verbose is set
    # to display the working of the training model.
    logreg = LogisticRegression(random_state=0, solver='lbfgs', max_iter=100, multi_class='ovr', verbose=1)
    logreg.fit(xtrain, ytrain)

    ypred = logreg.predict(xtest)
    cm = confusion_matrix(ytest, ypred)

    acc[4] = (accuracy(cm) * 100)
    HeatMap(cm, 'logistic_regression_heatmap')


# then we use the decision tree classifier as an machine learning model to train and test our datasets.
def DTC():

    # creating a decision tree classifier training model setting max depth at 50 this defines the height of the tree.
    # at 50 we achieve the maximum accuracy so there was no need to increase the depth criterion is used to measure
    # the quality of split. we use 'gini'.
    dtc = DecisionTreeClassifier(random_state=0, max_depth=50, criterion='gini')
    dtc = dtc.fit(xtrain, ytrain)

    ypred = dtc.predict(xtest)
    cm = confusion_matrix(ytest, ypred)

    acc[5] = (accuracy(cm) * 100)
    HeatMap(cm, 'decision_tree_classifier_heatmap')


def SVM():
    print("trainig the model....")
    # now we create and svm training model by using linearSVC() method.
    # we use  'ovo' the default decision function shape for our multiclass classification

    #supvec = svm.LinearSVC()
    #supvec.fit(xtrain, ytrain)
    # THIS IS TO JUST SAVE THE TRAINED SVC MODEL FOR FUTURE USE.
    import pickle
    filename = 'svmmodels.sav'
    #pickle.dump(supvec, open(filename, 'wb'))
    supvec = pickle.load(open(filename,'rb'))
    ypred = supvec.predict(xtest)
    cm = confusion_matrix(ytest, ypred)

    acc[6] = (accuracy(cm) * 100)
    HeatMap(cm, 'support_vector_machine_heatmap')


# ----GUI CREATED BY pyQt5 designer and imported and integreted here in this program----------

from PyQt5 import QtCore, QtGui, QtWidgets


# this is the main screen window GUI
class Ui_MainWindow(object):
    def openwindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_ModelWindow()
        self.ui.setupUi(self.window)
        self.window.show()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 385)
        MainWindow.setAutoFillBackground(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 110, 141, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 150, 121, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 190, 161, 20))
        self.label_3.setObjectName("label_3")
        self.trainbutton = QtWidgets.QPushButton(self.centralwidget)
        self.trainbutton.setGeometry(QtCore.QRect(170, 110, 75, 23))
        self.trainbutton.setObjectName("trainbutton")

        self.modelsbutton = QtWidgets.QPushButton(self.centralwidget)
        self.modelsbutton.setGeometry(QtCore.QRect(170, 190, 75, 23))
        self.modelsbutton.setObjectName("modelsbutton")
        self.testbutton = QtWidgets.QPushButton(self.centralwidget)
        self.testbutton.setGeometry(QtCore.QRect(170, 150, 75, 23))
        self.testbutton.setObjectName("testbutton")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(220, 0, 371, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(220, 40, 381, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(560, 70, 81, 40))
        self.label_6.setObjectName("label_6")
        self.p1 = QtWidgets.QLabel(self.centralwidget)
        self.p1.setGeometry(QtCore.QRect(390, 100, 81, 20))
        self.p1.setObjectName("p1")
        self.p2 = QtWidgets.QLabel(self.centralwidget)
        self.p2.setGeometry(QtCore.QRect(390, 130, 71, 20))
        self.p2.setObjectName("p2")
        self.pn2 = QtWidgets.QLabel(self.centralwidget)
        self.pn2.setGeometry(QtCore.QRect(490, 130, 47, 13))
        self.pn2.setObjectName("pn2")
        self.pn1 = QtWidgets.QLabel(self.centralwidget)
        self.pn1.setGeometry(QtCore.QRect(490, 100, 47, 13))
        self.pn1.setObjectName("pn1")
        self.p3 = QtWidgets.QLabel(self.centralwidget)
        self.p3.setGeometry(QtCore.QRect(390, 160, 71, 20))
        self.p3.setObjectName("p3")
        self.pn3 = QtWidgets.QLabel(self.centralwidget)
        self.pn3.setGeometry(QtCore.QRect(490, 160, 47, 13))
        self.pn3.setObjectName("pn3")
        self.p4 = QtWidgets.QLabel(self.centralwidget)
        self.p4.setGeometry(QtCore.QRect(386, 190, 81, 20))
        self.p4.setObjectName("p4")
        self.pn4 = QtWidgets.QLabel(self.centralwidget)
        self.pn4.setGeometry(QtCore.QRect(490, 190, 47, 13))
        self.pn4.setObjectName("pn4")
        self.p5 = QtWidgets.QLabel(self.centralwidget)
        self.p5.setGeometry(QtCore.QRect(390, 220, 71, 16))
        self.p5.setObjectName("p5")
        self.pn5 = QtWidgets.QLabel(self.centralwidget)
        self.pn5.setGeometry(QtCore.QRect(490, 220, 47, 13))
        self.pn5.setObjectName("pn5")
        self.pn7 = QtWidgets.QLabel(self.centralwidget)
        self.pn7.setGeometry(QtCore.QRect(710, 130, 47, 13))
        self.pn7.setObjectName("pn7")
        self.pn9 = QtWidgets.QLabel(self.centralwidget)
        self.pn9.setGeometry(QtCore.QRect(710, 190, 47, 13))
        self.pn9.setObjectName("pn9")
        self.pn10 = QtWidgets.QLabel(self.centralwidget)
        self.pn10.setGeometry(QtCore.QRect(710, 220, 47, 13))
        self.pn10.setObjectName("pn10")
        self.p10 = QtWidgets.QLabel(self.centralwidget)
        self.p10.setGeometry(QtCore.QRect(630, 220, 61, 16))
        self.p10.setObjectName("p10")
        self.p8 = QtWidgets.QLabel(self.centralwidget)
        self.p8.setGeometry(QtCore.QRect(630, 160, 71, 16))
        self.p8.setObjectName("p8")
        self.p9 = QtWidgets.QLabel(self.centralwidget)
        self.p9.setGeometry(QtCore.QRect(630, 190, 71, 16))
        self.p9.setObjectName("p9")
        self.pn8 = QtWidgets.QLabel(self.centralwidget)
        self.pn8.setGeometry(QtCore.QRect(710, 160, 47, 13))
        self.pn8.setObjectName("pn8")
        self.p7 = QtWidgets.QLabel(self.centralwidget)
        self.p7.setGeometry(QtCore.QRect(630, 130, 47, 13))
        self.p7.setObjectName("p7")
        self.p6 = QtWidgets.QLabel(self.centralwidget)
        self.p6.setGeometry(QtCore.QRect(630, 100, 61, 16))
        self.p6.setObjectName("p6")
        self.pn6 = QtWidgets.QLabel(self.centralwidget)
        self.pn6.setGeometry(QtCore.QRect(710, 100, 47, 13))
        self.pn6.setObjectName("pn6")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "1) SHOW TRAIN DATASET"))
        self.label_2.setText(_translate("MainWindow", "2)SHOW TEST DATASET"))
        self.label_3.setText(_translate("MainWindow", "3)SELECT A TRAINING MODEL"))
        self.trainbutton.setText(_translate("MainWindow", "SELECT"))
        # clicking on this button will call the displaydataset() function and will display info of train dataset
        self.trainbutton.clicked.connect(lambda: displaydataset(train_data))
        self.trainbutton.clicked.connect(self.testlabel)
        self.modelsbutton.setText(_translate("MainWindow", "SELECT"))
        # clicking on this button will take you to the model selection window.
        self.modelsbutton.clicked.connect(self.openwindow)

        self.testbutton.setText(_translate("MainWindow", "SELECT"))
        # clicking on this button will call the displaydataset() function and will display info of test dataset
        self.testbutton.clicked.connect(lambda: displaydataset(test_data))
        self.testbutton.clicked.connect(self.testlabel)
        self.label_4.setText(_translate("MainWindow", "POKER HAND PREDICTION"))
        self.label_5.setText(_translate("MainWindow", "PREDICTION OF POKER HAND ON DIFFERENT MACHINE LEARNING MODELS"))
        self.label_6.setText(_translate("MainWindow", "TABLESIZE"))
        self.p1.setText(_translate("MainWindow", "Nothing in hand"))
        self.p2.setText(_translate("MainWindow", "One pair"))
        self.pn2.setText(_translate("MainWindow", "0"))
        self.pn1.setText(_translate("MainWindow", "0"))
        self.p3.setText(_translate("MainWindow", "Two pairs"))
        self.pn3.setText(_translate("MainWindow", "0"))
        self.p4.setText(_translate("MainWindow", "Three of a kind"))
        self.pn4.setText(_translate("MainWindow", "0"))
        self.p5.setText(_translate("MainWindow", "Straight"))
        self.pn5.setText(_translate("MainWindow", "0"))
        self.pn7.setText(_translate("MainWindow", "0"))
        self.pn9.setText(_translate("MainWindow", "0"))
        self.pn10.setText(_translate("MainWindow", "0"))
        self.p10.setText(_translate("MainWindow", "Royal flush"))
        self.p8.setText(_translate("MainWindow", "Four of a kind"))
        self.p9.setText(_translate("MainWindow", "Straight flush"))
        self.pn8.setText(_translate("MainWindow", "0"))
        self.p7.setText(_translate("MainWindow", "Full house"))
        self.p6.setText(_translate("MainWindow", "Flush"))
        self.pn6.setText(_translate("MainWindow", "0"))

    # this function show the info of train/test dataset on to the gui when called.
    def testlabel(self):
        self.label_6.setText(str(datasize))
        self.pn1.setText(str(cls[0]))
        self.pn2.setText(str(cls[1]))
        self.pn3.setText(str(cls[2]))
        self.pn4.setText(str(cls[3]))
        self.pn5.setText(str(cls[4]))
        self.pn6.setText(str(cls[5]))
        self.pn7.setText(str(cls[6]))
        self.pn8.setText(str(cls[7]))
        self.pn9.setText(str(cls[8]))
        self.pn10.setText(str(cls[9]))


# this is model selection window GUI
class Ui_ModelWindow(object):
    def setupUi(self, ModelWindow):
        ModelWindow.setObjectName("ModelWindow")
        ModelWindow.resize(800, 401)
        self.centralwidget = QtWidgets.QWidget(ModelWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(220, 0, 371, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(240, 40, 321, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 120, 151, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(0, 160, 151, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(0, 200, 151, 16))
        self.label_3.setObjectName("label_3")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(0, 240, 151, 16))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(0, 280, 151, 16))
        self.label_7.setObjectName("label_7")
        self.kerastestbutton = QtWidgets.QPushButton(self.centralwidget)
        self.kerastestbutton.setGeometry(QtCore.QRect(150, 120, 75, 23))
        self.kerastestbutton.setObjectName("kerastestbutton")
        self.bnbutton = QtWidgets.QPushButton(self.centralwidget)
        self.bnbutton.setGeometry(QtCore.QRect(150, 160, 75, 23))
        self.bnbutton.setObjectName("bnbutton")
        self.lgbutton = QtWidgets.QPushButton(self.centralwidget)
        self.lgbutton.setGeometry(QtCore.QRect(150, 200, 75, 23))
        self.lgbutton.setObjectName("lgbutton")
        self.treebutton = QtWidgets.QPushButton(self.centralwidget)
        self.treebutton.setGeometry(QtCore.QRect(150, 240, 75, 23))
        self.treebutton.setObjectName("treebutton")
        self.svmbutton = QtWidgets.QPushButton(self.centralwidget)
        self.svmbutton.setGeometry(QtCore.QRect(150, 280, 75, 23))
        self.svmbutton.setObjectName("svmbutton")
        self.kerasacc = QtWidgets.QLabel(self.centralwidget)
        self.kerasacc.setGeometry(QtCore.QRect(470, 120, 111, 21))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.kerasacc.setFont(font)
        self.kerasacc.setObjectName("kerasacc")
        self.macc = QtWidgets.QLabel(self.centralwidget)
        self.macc.setGeometry(QtCore.QRect(460, 160, 111, 21))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.macc.setFont(font)
        self.macc.setObjectName("macc")
        self.bacc = QtWidgets.QLabel(self.centralwidget)
        self.bacc.setGeometry(QtCore.QRect(240, 160, 111, 21))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.bacc.setFont(font)
        self.bacc.setObjectName("bacc")
        self.gacc = QtWidgets.QLabel(self.centralwidget)
        self.gacc.setGeometry(QtCore.QRect(680, 160, 111, 21))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.gacc.setFont(font)
        self.gacc.setObjectName("gacc")
        self.lgacc = QtWidgets.QLabel(self.centralwidget)
        self.lgacc.setGeometry(QtCore.QRect(250, 200, 111, 21))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lgacc.setFont(font)
        self.lgacc.setObjectName("lgacc")
        self.treeacc = QtWidgets.QLabel(self.centralwidget)
        self.treeacc.setGeometry(QtCore.QRect(250, 240, 111, 21))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.treeacc.setFont(font)
        self.treeacc.setObjectName("treeacc")
        self.svmacc = QtWidgets.QLabel(self.centralwidget)
        self.svmacc.setGeometry(QtCore.QRect(250, 280, 111, 21))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.svmacc.setFont(font)
        self.svmacc.setObjectName("svmacc")
        self.kerasacc_2 = QtWidgets.QLabel(self.centralwidget)
        self.kerasacc_2.setGeometry(QtCore.QRect(250, 80, 311, 21))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.kerasacc_2.setFont(font)
        self.kerasacc_2.setObjectName("kerasacc_2")
        self.kerastrainbutton = QtWidgets.QPushButton(self.centralwidget)
        self.kerastrainbutton.setGeometry(QtCore.QRect(240, 120, 75, 23))
        self.kerastrainbutton.setObjectName("kerastrainbutton")
        self.mnbutton = QtWidgets.QPushButton(self.centralwidget)
        self.mnbutton.setGeometry(QtCore.QRect(370, 160, 75, 23))
        self.mnbutton.setObjectName("mnbutton")
        self.gnbutton = QtWidgets.QPushButton(self.centralwidget)
        self.gnbutton.setGeometry(QtCore.QRect(590, 160, 75, 23))
        self.gnbutton.setObjectName("gnbutton")
        self.showacc = QtWidgets.QPushButton(self.centralwidget)
        self.showacc.setGeometry(QtCore.QRect(490, 80, 101, 23))
        self.showacc.setObjectName("showacc")
        ModelWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(ModelWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        ModelWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(ModelWindow)
        self.statusbar.setObjectName("statusbar")
        ModelWindow.setStatusBar(self.statusbar)

        self.retranslateUi(ModelWindow)
        QtCore.QMetaObject.connectSlotsByName(ModelWindow)

    def retranslateUi(self, ModelWindow):
        _translate = QtCore.QCoreApplication.translate
        ModelWindow.setWindowTitle(_translate("ModelWindow", "MainWindow"))
        self.label_4.setText(_translate("ModelWindow", "POKER HAND PREDICTION"))
        self.label_5.setText(_translate("ModelWindow", "SELECT A MODEL FROM THE LIST TO TRAIN AND TEST"))
        self.label.setText(_translate("ModelWindow", "1)KERAS NEURAL NETWORK"))
        self.label_2.setText(_translate("ModelWindow", "2) NAIIVE BUYES CLASSIFIERS"))
        self.label_3.setText(_translate("ModelWindow", "3)LOGISTIC REGRESSION"))
        self.label_6.setText(_translate("ModelWindow", "4) DECISION TREE CLASSIFER"))
        self.label_7.setText(_translate("ModelWindow", "5)SVM"))
        self.kerastestbutton.setText(_translate("ModelWindow", "TRAIN"))
        # by clicking on this button we call the trainkeras function
        self.kerastestbutton.clicked.connect(trainkeras)
        self.bnbutton.setText(_translate("ModelWindow", "TRAIN/TEST"))
        # by clicking on this button we call the bernoulliNBC function
        self.bnbutton.clicked.connect(bernoulliNBC)
        self.lgbutton.setText(_translate("ModelWindow", "TRAIN/TEST"))
        # by clicking on this button we call the LogReg function
        self.lgbutton.clicked.connect(LogReg)
        self.treebutton.setText(_translate("ModelWindow", "TRAIN/TEST"))
        # by clicking on this button we call the DTC function
        self.treebutton.clicked.connect(DTC)
        self.svmbutton.setText(_translate("ModelWindow", "TRAIN/TEST"))
        # by clicking on this button we call the SVM function
        self.svmbutton.clicked.connect(SVM)
        self.kerasacc.setText(_translate("ModelWindow", "ACCURACY"))
        self.macc.setText(_translate("ModelWindow", "ACCURACY(MN)"))
        self.bacc.setText(_translate("ModelWindow", "ACCURACY(BN)"))
        self.gacc.setText(_translate("ModelWindow", "ACCURACY(GN)"))
        self.lgacc.setText(_translate("ModelWindow", "ACCURACY"))
        self.treeacc.setText(_translate("ModelWindow", "ACCURACY"))
        self.svmacc.setText(_translate("ModelWindow", "ACCURACY"))
        self.kerasacc_2.setText(_translate("ModelWindow", "COMPARING ALL THE ACCURACIES"))
        self.kerastrainbutton.setText(_translate("ModelWindow", "TEST"))
        # by clicking on this button we call the testkeras function
        self.kerastrainbutton.clicked.connect(testkeras)
        self.mnbutton.setText(_translate("ModelWindow", "TRAIN/TEST"))
        # by clicking on this button we call the multinomialNBC function
        self.mnbutton.clicked.connect(multinomialNBC)
        self.gnbutton.setText(_translate("ModelWindow", "TRAIN/TEST"))
        # by clicking on this button we call the gaussianNBC function
        self.gnbutton.clicked.connect(gaussianNBC)
        self.showacc.setText(_translate("ModelWindow", "SHOW ACCURACY"))
        # by clicking on this button we call the accuracies  function
        self.showacc.clicked.connect(self.accuracies)

    # this accuracies function will show the accuracies of each model after training and testing them. by default there are set to 0
    def accuracies(self):
        print(acc)
        self.kerasacc.setText(str(acc[0]))
        self.bacc.setText(str(acc[1]))
        self.macc.setText(str(acc[2]))
        self.gacc.setText(str(acc[3]))
        self.lgacc.setText(str(acc[4]))
        self.treeacc.setText(str(acc[5]))
        self.svmacc.setText(str(acc[6]))


acc = [0, 0, 0, 0, 0, 0, 0]
print("importing libraries...")

print("imported")
print("reading the train and test datasets and splitting them xtrain,ytrain and xtest and ytest respectively")
# importing and pre-processing the data set by calling readDataSet() function
readDataSet()

# program starts here by running the main GUI window.
if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
