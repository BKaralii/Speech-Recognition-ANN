Speech Recognition ANN Implementation
=====================================

An implementation of Speech Recognition using Artificial Neural Networks. 

Language Used: Python  -  with Python3 updated 

You need numpy and scipy for this to work.

Words Recognized: "Down", "Eat", "Sleep", "Up"

#How to add new words

Optional 1 
----------------------------------------------------------------------------------------------------------------------------------------------------------------
1. Record your new word in Audacity or any audio processing software. Set the sampling rate to 44100Hz then export into a .wav file. It would be better to record a lot of samples from different speakers to improve accuracy.

2. Put the wav files into the training_sets directory. Rename your wav files to the word you want to add + -sample_index (ex: hello-1.wav,hello-2.wav). In this way, the feature extractor later can iterate within the files easily.

Optional 2
------------------------------------------------------------------------------------------------------------------------------------------------------------------
1. You can record the sounds sequentially by pressing the training voice recording button.

2. You can start saying the words with commands that will appear on the console.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------

3. In the featureExtractor.py, append your new word to the words array.

4. Run the main.py. Numpy files with Mel Cepstrum Coefficients will be generated in the mfccData folder.

5. In anntrainer.py, go to the main function, open another file instance: Ex. f6 = open("mfccData/hello_mfcc.npy").

6. Load the npy file by using np.load() then concatenate it in the inputArray

7. You have to edit the Neural network target outputs, so if I'm going to add the new_word, I'll need to edit the results as follows

```
t1 = np.array([[1,0,0,0,0,0] for _ in range(len(inputArray1))]) #Down
t2 = np.array([[0,1,0,0,0,0] for _ in range(len(inputArray2))]) #Eat
t3 = np.array([[0,0,1,0,0,0] for _ in range(len(inputArray3))]) #Sleep
t4 = np.array([[0,0,0,1,0,0] for _ in range(len(inputArray4))]) #Up
t5 = np.array([[0,0,0,0,1,0] for _ in range(len(inputArray5))]) #New_Word

target = np.concatenate([t1,t2,t3,t4,t5])
```

<dl>
    <dt>if</dt>
    <dd>don't use training voice recording button then run anntrainer.py. <br/>This could take a lot of time to compute. Grab a coffee while you wait =)</dd>
    <dt>else</dt>
    <dd>You can start drinking coffee immediately after recording the voice training (:</dd>
</dl>

#Running the speech recognizer  </br>
Just run main.py! =)  </br>

