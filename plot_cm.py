'''
Function to plot confusion matrix
'''

#%%Import packages
import numpy as np
import matplotlib.pylab as plt
from mlxtend.plotting import plot_confusion_matrix



#%%create plot

def plot_cm(cm, class_dict, y_test):
    fig, ax = plot_confusion_matrix(
        cm, 
        class_names=class_dict.values(),)

    #ax.set_xticklabels([''] + list(np.unique(y_test)))
    #ax.set_yticklabels([''] + list(np.unique(y_test)))
    plt.show()