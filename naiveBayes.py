import util
import classificationMethod
import math
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1  # this is the smoothing parameter, ** use it in your train method **
        # Look at this flag to decide whether to choose k automatically ** use this in your train method **
        self.automaticTuning = False
        self.probabilities = dict()

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # this could be useful for your code later...
        self.features = trainingData[0].keys()

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels,
                          validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        # go thru the training labels
        # convert trainingLabels to dict so we can make a counter object from it
        num_of_labels = 10
        num_of_rows = 28
        num_of_columns = 28
        # need to set the probability arr to 0.001 because when calculating join probability
        # we dont want a 0 value canceling out our whole prediction.
        self.probability_arr = np.full((num_of_labels, num_of_rows,
                                        num_of_columns),  0.001, dtype=np.float)
        label_proportions = {}
        for label in self.legalLabels:
            label_proportions[label] = util.Counter()
            # assigns a counter to all legal labels

        # for every thing in the training data increment respective label when seen in training data
        for t in trainingLabels:
            label_proportions[t].incrementAll([t], 1)
        print(label_proportions)
        # gets us the amount of that label there is

        # print(label_proportions.__getitem__(2).totalCount())
        '''
        This loop just fixes the array to fit the training data, problem was that
        we had 0.001 at every index, when 1/#of datapoints was added it wouldnt sum to 1,
        difference is honestly negligble will probably take out
        '''
        for i in self.legalLabels:
            label = trainingLabels[i]
            temp1 = label_proportions.__getitem__(label).totalCount()
            for key in trainingData[i].keys():
                temp2 = trainingData[label].__getitem__(key)
                if(temp2 == 1):
                    self.probability_arr[label, key[0], key[1]] = 0
        '''
        This for loop iterates through the training data and sets the
        probability array for each label
        '''
        for i in range(len(trainingData)):
            label = trainingLabels[i]
            temp1 = label_proportions.__getitem__(label).totalCount()
            for key in trainingData[i].keys():
                temp2 = trainingData[i].__getitem__(key)
                if(temp2 == 1):
                    self.probability_arr[label, key[0], key[1]] += (1.0/temp1)

            # our feature[label] stores the probability matrix for a given label
            # we add the pixel value which is either 0 or 1, times 1/# of that label from training,
            # now we'll have the count of each label in the training data so we have our P(Y) and P(Not Y)
            # now we still need the P(y | x) so we need to go through training data and find the probability
            # for each feature given its label
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        # Log posteriors are stored for later data analysis (autograder).
        self.posteriors = []
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
        """
        logJoint = util.Counter()

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)
        """
        featuresOdds = []

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return featuresOdds
