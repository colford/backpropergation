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
> python number-recognition.py
```
