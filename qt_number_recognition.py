##############################################################################
# Qt frontend to load the weights of the backpropergation neural network.
# Allows the user to train the network and test it at any point.
##############################################################################

import sys
import number_recognition as nr
from itertools import cycle
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel,
                             QGridLayout, QVBoxLayout, QHBoxLayout,
                             QPushButton, QGroupBox, QProgressBar)


def handler(msg_type, msg_log_context, msg_string):
    ''' Supress warning messages
    '''
    pass


def activation_indicator(height=20):
    ''' Return an activation indicator widget.
    '''
    bar = QProgressBar()
    bar.setOrientation(QtCore.Qt.Vertical)
    bar.setRange(0, 100)
    bar.setFixedHeight(height)
    bar.setFixedWidth(10)
    bar.setTextVisible(False)
    bar.setStyleSheet('''
        QProgressBar::chunk {
            background-color: red;
            width: 1;
            height: 1;
        }
    ''')
    return bar


# Supress startup Warnings.
QtCore.qInstallMessageHandler(handler)


class QtNumberRecognition(QWidget):
    ''' Qt UI frontend to train a neural network to patern match numbers
        drawn in a limited grid using the backpropergation algorithm.
    '''

    def __init__(self):
        super(QtNumberRecognition, self).__init__()
        self.title = 'Number Recognition with Backpropergation'
        self.input = []
        self.history_output = {}
        self.hidden = []
        self.output = []
        self.iterations = 0
        self.train_values = cycle(list(nr.training_set.keys()))
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.createModeLayout()
        self.createInputLayout()
        self.createOutputLayout()
        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.controlGroupBox)
        windowLayout.addWidget(self.inputGroupBox)
        windowLayout.addWidget(self.outputGroupBox)
        self.setLayout(windowLayout)
        self.show()

    def createModeLayout(self):
        self.controlGroupBox = QGroupBox("Control")
        vlayout = QVBoxLayout()
        hlayout = QHBoxLayout()
        iterations = QLabel('Iterations:')
        self.num_iterations = QLabel('%d' % self.iterations)
        run_iteration = QPushButton("Run learning iternation")
        run_iteration.clicked.connect(self.backpropergation)
        hlayout.addWidget(iterations)
        hlayout.addWidget(self.num_iterations)
        vlayout.addLayout(hlayout)
        vlayout.addWidget(run_iteration)
        self.controlGroupBox.setLayout(vlayout)

    def createInputLayout(self):
        self.inputGroupBox = QGroupBox("Input/Hidden/Output")
        hlayout = QHBoxLayout()
        input_layout = QGridLayout()
        hidden_layout = QGridLayout()
        output_layout = QGridLayout()
        input_layout.setSpacing(0)
        hidden_layout.setSpacing(0)
        output_layout.setSpacing(0)
        for y in range(0, nr.pixels_per_pattern["height"]):
            for x in range(0, nr.pixels_per_pattern["width"]):
                but = QPushButton('')
                self.input.append(but)
                input_layout.addWidget(but, y, x)
        for y in range(1, nr.hidden_layer_size):
            bar = activation_indicator(10)
            hidden_layout.addWidget(bar, y, 0)
            self.hidden.append(bar)
        for y in range(0, nr.output_layer_size):
            bar = activation_indicator(10)
            output_layout.addWidget(bar, y, 0)
            self.output.append(bar)
        hlayout.addLayout(input_layout)
        hlayout.addLayout(hidden_layout)
        hlayout.addLayout(output_layout)
        self.inputGroupBox.setLayout(hlayout)

    def createOutputLayout(self):
        self.outputGroupBox = QGroupBox("Output Activation History")
        layout = QGridLayout()
        layout.setSpacing(1)
        for x in range(0, nr.patterns_to_train):
            lab = QLabel('%d' % x)
            lab.setFixedWidth(20)
            layout.addWidget(lab, x, 0)
            self.history_output[x] = []
            for y in range(0, nr.patterns_to_train):
                bar = activation_indicator()
                layout.addWidget(bar, x, y + 1)
                self.history_output[x].append(bar)
        self.outputGroupBox.setLayout(layout)

    def present(self, item):
        nr.present(item)
        for inx in range(0, len(self.input)):
            if nr.input_neurons[inx + 1] == 1:
                self.input[inx].setStyleSheet("background-color: grey")
            else:
                self.input[inx].setStyleSheet("")

    def propergate(self):
        nr.propergate()
        for idx in range(1, nr.hidden_layer_size):
            self.hidden[idx - 1].setValue(
                int(nr.hidden_neurons[idx] * 100))
        for idx in range(0, nr.output_layer_size):
            self.output[idx].setValue(
                int(nr.output_neurons[idx] * 100))

    def store(self, item):
        nr.store(item)
        for idx in range(0, len(self.history_output[item])):
            self.history_output[item][idx].setValue(
                int(nr.current_status_of_learning[item][idx] * 100))

    def backpropergation(self):
        self.iterations += 1
        self.num_iterations.setText('%d' % self.iterations)
        for item in self.train_values:
            self.present(item)
            self.propergate()
            nr.error(item)
            nr.adjust()
            self.store(item)
            break


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = QtNumberRecognition()
    result = app.exec_()
    sys.exit(result)
