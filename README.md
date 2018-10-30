# Backpropergation in python
Simple Backpropergation Neural Network to learn to recoginse numbers 0 - 9 in a very limited grid. 
Based upon my 1992 undergrad Project; yes, neural networks arn't new! Just thought I'd see what it 
was like re-implementing it in python. It's not meant to be the best example of a backpropergation
neural net but just a play thing.

Numbers are represented in a 5x7 pxiel grid. Each pixel is either 0/1.
Each number is shown below and this is the training set. The ASCII under
the text "Number = N" displays the current status of the output neurons
and a measure of their level of activation.

Activation runs from 0 - 1
```
 >  0.85 '*'
 >  0.55 '^'
 >  0.30 '-'
 <= 0.30 '.'
```
There are ten characters representing the ten output neurons activation and
the target output starts 0 (left most). Using this you can watch the network
learn.
```
Number = 0          Number = 1          Number = 2
*.........          .*........          ..*.......
  #                   #                 ####
 # #                 ##                     #
#   #               # #                    #
#   #                 #                   #
#   #                 #                  #
 # #                  #                 #
  #                 #####                ####

Number = 3          Number = 4          Number = 5
...*......          ....*.....          .....*....
 ###                   #                 ####
#   #                 ##                #
    #                # #                #
 ####               #  #                ####
    #               #####                   #
#   #                  #                #   #
 ###                   #                 ###

Number = 6          Number = 7          Number = 8
......*...          .......*..          ........*.
 ###                #####                ###
#   #                   #               #   #
#                      #                #   #
####                  #                  ###
#   #                #                  #   #
#   #               #                   #   #
 ###                #                    ###

Number = 9
.........*
 ###
#   #
#   #
 ####
    #
    #
    #
```
A prompt for [y|Y] after all 10 grid numbers have been presented and errors
backpropergated. Press return to carry on with the next 10 presentations or
enter [y|Y] to end.

To run use python 3 and type:
```
> python number_recognition.py
```

There is also a Qt version that can be invoked with:
```
> python qt_number_recognition.py
```
This will display the following:

![Alt text](images/qt_number_recognition.png?raw=true "Qt Number Recognition")

Each press of the "Run learning iteration" will present the next number in the cycle and propergate the data through the hidden and output layers. It will then work out the errors and backpropergate those by adjusting the weights (connections) between neurons. You can see the level of activation for the given current input in the hidden layer and output layer by the level of Red in the neuron i.e. more red more activation, ranging from 0 - 1. The display at the bottom shows the history of activation for the presented numbers i.e. last time it was presented we had these activations in the output layer. You can see from above that 6 and 8 are still in progress of finding unique features where as the rest are pretty well learnt. 

One last feature is that you can interactively turn on/off "pixels" and instantly see the effect on the hidden and output layers. Once you have taught the network you can see how much noise you can introduce before the activation of that number changes to something else. You can test to see which features in the input the network is picking up upon.

Enjoy! :)
