#!/usr/bin/python
import math
import random
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict
from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 1: binary classification
############################################################

############################################################
# Problem 1a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    pass
    # ### START CODE HERE ###
    FeatureVector = {}
    for word in x.split():
        if word not in FeatureVector:
            FeatureVector[word] = 1
        else:
            FeatureVector[word] += 1
    return FeatureVector
    # ### END CODE HERE ###


############################################################
# Problem 1b: stochastic gradient descent

T = TypeVar("T")


def learnPredictor(
    trainExamples: List[Tuple[T, int]],
    validationExamples: List[Tuple[T, int]],
    featureExtractor: Callable[[T], FeatureVector],
    numEpochs: int,
    eta: float,
) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    """
    weights = {}  # feature => weight
    # ### START CODE HERE ###

    # ------------- Hinge Loss Start -------------
    def gradientLoss(w, i):
        x, y = trainExamples[i]
        phix_grad = featureExtractor(x)
        if dotProduct(w, phix_grad) * y > 1:
            y = 0
        return phix_grad, y
    # ------------- Hinge Loss End -------------

    def stochasticGradientDescent(gradientLoss,n,numEpochs,eta):
        w = {}
        for t in range(numEpochs):
            for i in range(n):
                gradient,y = gradientLoss(w, i)
                increment(w,-eta*-y,gradient)
        return w

    weights = stochasticGradientDescent(gradientLoss, len(trainExamples),numEpochs,eta)
    # ### END CODE HERE ###
    return weights


############################################################
# Problem 1c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        phi = None
        y = None
        # ### START CODE HERE ###
        phi = {k: v * random.random() for k, v in weights.items()}
        y = 1 if dotProduct(phi, weights) > 0 else -1
        # ### END CODE HERE ###
        return (phi, y)

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 1d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x):
        pass
        # ### START CODE HERE ###
        FeatureVector = {}
        b = x.replace(" ","")
        for n_gram in [b[i:i+n] for i in range(len(b)-n+1)]:
            if n_gram not in FeatureVector:
                FeatureVector[n_gram] = 1
            else:
                FeatureVector[n_gram] += 1
        return FeatureVector

        # ### END CODE HERE ###

    return extract


############################################################
# Problem 1e:
#
# Helper function to test 1e.
#
# To run this function, run the command from termial with `n` replaced
#
# $ python -c "from solution import *; testValuesOfN(n)"
#


def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be submitted.
    """
    trainExamples = readExamples("polarity.train")
    validationExamples = readExamples("polarity.dev")
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(
        trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01
    )
    outputWeights(weights, "weights")
    outputErrorAnalysis(
        validationExamples, featureExtractor, weights, "error-analysis"
    )  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    validationError = evaluatePredictor(
        validationExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    print(
        (
            "Official: train error = %s, validation error = %s"
            % (trainError, validationError)
        )
    )


############################################################
# Problem 2b: K-means
############################################################


def kmeans(
    examples: List[Dict[str, float]], K: int, maxEpochs: int
) -> Tuple[List, List, float]:
    """
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    """
    # ### START CODE HERE ###
    zlist = [None] * len(examples)
    total_loss = 0
    prev_loss = float("inf")  # stores an infinite value that acts as an unbounded upper value for comparison
    centroids = {}

    #random_sample = random.sample(range(0, len(examples) - 1), K)  # get K random indexes from examples

    for c in range(K):
        centroids[c] = examples[c]

    cacheExamples = {}
    for i in range(len(examples)):
        cacheExamples[i] = dotProduct(examples[i], examples[i])


    for t in range(maxEpochs):
        if prev_loss == total_loss:
            break

        prev_loss == total_loss
        total_loss = 0

        cacheCentroids = {}
        for j in range(K):
            cacheCentroids[j] = dotProduct(centroids[j], centroids[j])

        # set assignments
        for i in range(len(examples)):
            example = examples[i]
            running_min = float("inf")
            exampleDot = cacheExamples[i]
            for j in range(K):
                centroid = centroids[j]
                centroidDot = cacheCentroids[j]
                dist = abs(exampleDot - 2 * dotProduct(example, centroid) + centroidDot)
                if dist < running_min:
                    running_min = dist
                    zlist[i] = j
            total_loss += running_min

        # update clusters
        for j in range(K):
            running_sum = {}
            count = 0.0
            for i in range(len(examples)):
                if zlist[i] == j:
                    increment(running_sum, 1, examples[i])
                    count += 1
            if count == 0.0:
                continue
            centroids[j] = {k: v / count for k, v in running_sum.items()}

    cen_list = []
    for v in centroids.items():
        cen_list.append(v)

    return cen_list, zlist, total_loss
    # ### END CODE HERE ###
