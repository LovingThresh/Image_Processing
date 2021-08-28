import seaborn as sns
import matplotlib.pyplot as plt


def plot_heatmap(predict_array):
    if predict_array.ndim == 4:
        sns.heatmap(predict_array[:, :, :, 0].reshape(227, 227))
    else:
        sns.heatmap(predict_array.reshape(227, 227))
    plt.show()
