
from threading import Thread
from record import record_to_file
from features import mfcc
from anntrainer import *
from featureExtractor import *
from anntester_single import *
import scipy.io.wavfile as wav

from tkinter import *


class Application(Frame):

    def createWidgets(self):
        self.button_image = PhotoImage(file="button.gif")
        self.RECORD = Button(self, image=self.button_image, width="100",
                             height="100", command=self.record_buttonpress)
        self.RECORD.pack()
        self.TEXTBOX = Text(self, height="1", width="30")
        self.TEXTBOX.pack()

        # for train record button
        self.button_image2 = PhotoImage(file="button.gif")
        self.RECORD2 = Button(self, image=self.button_image2, width="100",
                              height="100", command=self.record2_buttonpress)
        self.RECORD2.pack()
        self.TEXTBOX2 = Text(self, height="1", width="30")
        self.TEXTBOX2.pack()

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()
        self.TEXTBOX.insert(INSERT, "Press to record")
        self.TEXTBOX.tag_config(
            "recording", foreground="red", justify="center")
        self.TEXTBOX.tag_config(
            "success", foreground="red", justify="center")
        self.TEXTBOX.configure(state="disabled")
        self.TEXTBOX2.insert(INSERT, "Press to record for train dataset")
        self.TEXTBOX2.tag_config(
            "recording", foreground="red", justify="center")
        self.TEXTBOX2.tag_config(
            "success", foreground="darkgreen", justify="center")
        self.TEXTBOX2.configure(state="disabled")

    def record_buttonpress(self):  # TEST BUTTON
        recorder_thread = Thread(
            target=record_and_test, args=(self.TEXTBOX, self.RECORD))
        recorder_thread.start()

    def record2_buttonpress(self):  # TRAIN BUTTON
        recorder_thread = Thread(
            target=record_and_train, args=(self.TEXTBOX2, self.RECORD2))
        recorder_thread.start()


def record_and_train(textbox, button):

    # button on
    button.configure(state="disabled")
    textbox.configure(state="normal")
    textbox.delete("1.0", END)
    textbox.insert(INSERT, "Recording")
    textbox.tag_add("recording", "1.0", END)
    textbox.configure(state="disabled")

    words = ['down', 'eat', 'sleep', 'up']
    for i in range(len(words)):  # WORD loop
        for j in range(10):  # REPEAT loop
            # record to train
            record_to_file("training_sets/" +
                           words[i] + "-" + str(j+1) + ".wav")

            print("repeat", words[i])
        if(len(words) != i+1):
            print("next word", words[i+1])
        else:
            print("finish")

    # Button of
    textbox.configure(state="normal")
    textbox.delete("1.0", END)
    textbox.tag_remove("recording", "1.0")
    textbox.tag_add("success", "1.0", END)
    textbox.configure(state="disabled")
    button.configure(state="normal")

    # MFCC calculation and TRAIN
    mfcc_apply()
    print("train start..")
    TRAIN()


def record_and_test(textbox, button, filename="test_files/test.wav"):
    # Buton on
    button.configure(state="disabled")
    textbox.configure(state="normal")
    textbox.delete("1.0", END)
    textbox.insert(INSERT, "Recording")
    textbox.tag_add("recording", "1.0", END)
    textbox.configure(state="disabled")

    # Record to test
    record_to_file(filename)

    # Feed into ANN
    testNet = testInit()
    inputArray = extractFeature(filename)
    print(len(inputArray))
    outStr = feedToNetwork(inputArray, testNet)

    # Button of
    textbox.configure(state="normal")
    textbox.delete("1.0", END)
    textbox.tag_remove("recording", "1.0")
    textbox.insert(INSERT, outStr)
    textbox.tag_add("success", "1.0", END)
    textbox.configure(state="disabled")
    button.configure(state="normal")


if __name__ == '__main__':

    # Display GUI
    root = Tk()
    app = Application(master=root)
    app.mainloop()
    # root.destroy()
