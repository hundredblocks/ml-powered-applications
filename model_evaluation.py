from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def get_confusion_matrix_plot(predicted_y, true_y,
                              classes=None,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.winter):
    """
    Inspired by scikit-learn example
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    :param predicted_y:
    :param true_y:
    :param classes:
    :param normalize:
    :param title:
    :param cmap:
    :return: plot for the confusion matrix
    """
    if classes is None:
        classes = ['Answered', 'Unanswered']

    cm = confusion_matrix(true_y, predicted_y)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in np.itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    return plt
