from __future__ import division
import numpy as np
import scipy.io.wavfile as wav

from features import mfcc


class TestingNetwork:

    layerCount = 0
    shape = None
    weights = []

    def __init__(self, layerSize, weights):

        self.layerCount = len(layerSize) - 1
        self.shape = layerSize

        self._layerInput = []
        self._layerOutput = []
        self.weights = weights

    def forwardProc(self, input):

        InCases = input.shape[0]

        self._layerInput = []
        self._layerOutput = []

        for index in range(self.layerCount):
            if index == 0:
                layerInput = self.weights[0].dot(
                    np.vstack([input.T, np.ones([1, InCases])]))
            else:
                layerInput = self.weights[index].dot(
                    np.vstack([self._layerOutput[-1], np.ones([1, InCases])]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(self.sgm(layerInput))

        return self._layerOutput[-1].T

    def sgm(self, x, Derivative=False):
        if not Derivative:
            return 1 / (1+np.exp(-x))
        else:
            out = self.sgm(x)
            return out*(1-out)


def testInit():
    # Setup Neural Network
    f1 = open("network/vowel_network_words.npy", "r+b")
    # weights = np.load(f1)
    weights = np.load('network/vowel_network_words.npy',
                      allow_pickle=True, fix_imports=True, encoding='latin1')
    testNet = TestingNetwork((260, 25, 25, 5), weights)
    return testNet


def extractFeature(soundfile):
    # Get MFCC Feature Array
    (rate, sig) = wav.read(soundfile)
    duration = len(sig)/rate
    mfcc_feat = mfcc(sig, rate, winlen=duration/20, winstep=duration/20)
    print("MFCC Feature Length: " + str(len(mfcc_feat)))
    s = mfcc_feat[:20]
    st = []
    for elem in s:
        st.extend(elem)
    st /= np.max(np.abs(st), axis=0)
    inputArray = np.array([st])
    return inputArray


def feedToNetwork(inputArray, testNet):
    # Input MFCC Array to Network
    outputArray = testNet.forwardProc(inputArray)

    # if the maximum value in the output is less than
    # the threshold the system does not recognize the sound
    # the user spoke
    print(outputArray[0])
    indexMax = outputArray.argmax(axis=1)[0]

    print(outputArray)

    # Mapping each index to their corresponding meaning
    outStr = None

 #   if(outputArray.argmax(axis=1)[0][0] < 4 or outputArray.argmax(axis=1)[0][1] < 4):
 #       outStr = "None"
    if indexMax == 0:
        outStr = "Bed is Down"
    elif indexMax == 1:
        outStr = "Eat position apply"

    elif indexMax == 2:
        outStr = "Sleep position apply"
    elif indexMax == 3:
        outStr = "Bed is Up"
    # elif indexMax == 4:
    #    outStr = "Detected: Orange"

    print(outStr)
    return outStr


if __name__ == "__main__":

    testNet = testInit()
    inputArray = extractFeature("test_files/test.wav")
    feedToNetwork(inputArray, testNet)
