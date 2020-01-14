import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_embeddings(embeddings, sent_labels):
    """
    Plot embeddings, colored by sentence label
    :param embeddings: two dimensional embeddings
    :param sent_labels: labels to display
    """
    fig = plt.figure(figsize=(16, 10))
    color_map = {True: "#1f77b4", False: "#ff7f0e"}
    plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=[color_map[x] for x in sent_labels],
        s=40,
        alpha=0.4,
    )

    handles = [
        Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ["#1f77b4", "#ff7f0e"]
    ]
    labels = ["answered", "unanswered"]
    plt.legend(handles, labels)

    plt.gca().set_aspect("equal", "box")
    plt.gca().set_xlabel("x")
    plt.gca().set_ylabel("y")
