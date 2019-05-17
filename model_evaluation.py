from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    brier_score_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import numpy as np


def get_confusion_matrix_plot(
    predicted_y,
    true_y,
    classes=None,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.winter,
):
    """
    Inspired by sklearn example
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    :param predicted_y: model's predicted values
    :param true_y:  true value of the labels
    :param classes: names of both classes
    :param normalize: should we normalize the plot
    :param title: plot title
    :param cmap: colormap to use
    :return: plot for the confusion matrix
    """
    if classes is None:
        classes = ["Answered", "Unanswered"]

    cm = confusion_matrix(true_y, predicted_y)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    for i, j in np.itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] < thresh else "black",
            fontsize=40,
        )

    plt.tight_layout()
    plt.ylabel("True label", fontsize=30)
    plt.xlabel("Predicted label", fontsize=30)

    return plt


def get_roc_plot(predicted_proba_y, true_y):
    """
    Inspired by sklearn example
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    :param predicted_proba_y: the predicted probabilities of our model for each example
    :param true_y: the true value of the label
    :return:roc plot
    """
    fpr, tpr, thresholds = roc_curve(true_y, predicted_proba_y)
    roc_auc = auc(fpr, tpr)
    plt.plot(
        fpr, tpr, lw=1, alpha=0.3, label="ROC curve (AUC = %0.2f)" % roc_auc
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=2,
        color="r",
        label="Chance",
        alpha=0.8,
    )
    return plt


def get_calibration_plot(predicted_proba_y, true_y):
    """
    Inspired by sklearn example
    https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    :param predicted_proba_y: the predicted probabilities of our model for each example
    :param true_y: the true value of the label
    :return: calibration plot
    """

    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    clf_score = brier_score_loss(
        true_y, predicted_proba_y, pos_label=true_y.max()
    )
    print("\tBrier: %1.3f" % clf_score)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        true_y, predicted_proba_y, n_bins=10
    )

    ax1.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label="Brier score: (%1.3f)" % clf_score,
    )

    ax2.hist(predicted_proba_y, range=(0, 1), bins=10, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title("Calibration plot  (reliability curve)")

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    return plt


def get_metrics(predicted_y, true_y):
    """
    Get standard metrics for binary classification
    :param predicted_y: model's predicted values
    :param true_y:  true value of the labels
    :return:
    """
    # true positives / (true positives+false positives)
    precision = precision_score(
        true_y, predicted_y, pos_label=None, average="weighted"
    )
    # true positives / (true positives + false negatives)
    recall = recall_score(
        true_y, predicted_y, pos_label=None, average="weighted"
    )

    # harmonic mean of precision and recall
    f1 = f1_score(true_y, predicted_y, pos_label=None, average="weighted")

    # true positives + true negatives/ total
    accuracy = accuracy_score(true_y, predicted_y)
    return accuracy, precision, recall, f1
