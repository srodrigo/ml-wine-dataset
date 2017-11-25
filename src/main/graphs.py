import seaborn as sns
import matplotlib.pyplot as plt


def boxplot(data, figsize=(25, 10), file_name=None):
    plt.figure(figsize=figsize)
    sns.boxplot(data=data)
    plt.show()
    if file_name:
        plt.savefig(file_name)


def hist(data, bins=30, figsize=(16, 12), file_name=None):
    data.hist(bins=bins, figsize=figsize)
    plt.show()
    if file_name:
        plt.savefig(file_name)


def hist_by(
        data,
        by,
        by_values,
        bins=30,
        figsize=(16, 12),
        fontsize=20,
        folder_name=None):

    hist_groups = data.groupby(by).hist(
        bins=bins,
        figsize=figsize)
    for value, hist in zip(by_values, hist_groups):
        fig = hist[0, 0].get_figure()
        plt.title('%s: %d' % (by, value), fontsize=fontsize)
        fig.suptitle('%s: %d' % (by, value), fontsize=fontsize)
        plt.show()
        if folder_name:
            fig.savefig('%shist-%s-%d.png' % (folder_name, by, value))
        plt.close()


def pairplot(data, file_name=None):
    figure = sns.pairplot(data=data, diag_kind='kde')
    plt.show()
    figure.savefig(file_name)
