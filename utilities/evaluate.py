from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def evaluate(prediction, target):
    """
    This function evaluates the model after training on training set and testing on testing set.
    Plots of the confusion matrices are also shown.
    """
    print 'Evaluating Predictions...'
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(target[0:len(prediction)], prediction)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    class_names = ['Fatal', 'Serious', 'Slight']
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.show()