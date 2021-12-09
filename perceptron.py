# Perceptron implementation
import util
PRINT = True


class PerceptronClassifier:
    """
    Perceptron classifier.
    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            # this is the data-structure you should use
            self.weights[label] = util.Counter()
            # each number has a counter for the weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.
        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector a values).
        """
        # self.features just stores the keys of the image,
        self.features = trainingData[0].keys()  # could be useful later
        # basicFeatureExtractor method in dataClassifier.py breaks the image down
        # into 1s and 0s, self.features is the pixels that have values,
        # self.weights stores the pixels along with their weights

        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            for i in range(len(trainingData)):
                "*** YOUR CODE HERE ***"
                # training data stores the keys and values of the image
                guess = self.classify([trainingData[i]])[0]
                if guess != trainingLabels[i]:  # training
                    self.weights[guess] -= trainingData[i]
                    self.weights[trainingLabels[i]] += trainingData[i]

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.
        Recall that a datum is a util.counter... 
        """
        guesses = []
        # data is a list object
        # datum is a util.Counter object
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns a list of the 100 features with the greatest difference in weights:
                         w_label1 - w_label2
        """
        featuresOdds = []

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return featuresOdds
