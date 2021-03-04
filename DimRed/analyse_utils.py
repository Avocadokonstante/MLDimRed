import matplotlib.pyplot as plt
import os
import seaborn as sns
import csv
import numpy as np
from matplotlib.colors import ListedColormap



#-----------------------ERGEBNISSE SCHREIBEN----------------------------
def make_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def write_epoche_results(history, title, directory):
    make_directory(directory)

    with open('{}/{}.csv'.format(directory, title), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(map(lambda x: [x], history.history['mse']))
        csvwriter.writerow(map(lambda x: [x], history.history['val_mse']))
        csvwriter.writerow(map(lambda x: [x], history.history['mae']))
        csvwriter.writerow(map(lambda x: [x], history.history['val_mae']))
        csvwriter.writerow(map(lambda x: [x], history.history['r2_metric']))
        csvwriter.writerow(map(lambda x: [x], history.history['val_r2_metric']))

def write_dimension_results(errors, title, directory):
    make_directory(directory)

    with open('{}/{}.csv'.format(directory, title), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(map(lambda x: [x], errors['mse']))
        csvwriter.writerow(map(lambda x: [x], errors['val_mse']))
        csvwriter.writerow(map(lambda x: [x], errors['test_mse']))

        csvwriter.writerow(map(lambda x: [x], errors['mae']))
        csvwriter.writerow(map(lambda x: [x], errors['val_mae']))
        csvwriter.writerow(map(lambda x: [x], errors['test_mae']))

        csvwriter.writerow(map(lambda x: [x], errors['r2_metric']))
        csvwriter.writerow(map(lambda x: [x], errors['val_r2_metric']))
        csvwriter.writerow(map(lambda x: [x], errors['test_r2']))

def write_dimension_results_pca(errors, title, directory):
    make_directory(directory)

    with open('{}/{}.csv'.format(directory, title), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(map(lambda x: [x], errors['mse']))
        csvwriter.writerow(map(lambda x: [x], errors['test_mse']))

        csvwriter.writerow(map(lambda x: [x], errors['mae']))
        csvwriter.writerow(map(lambda x: [x], errors['test_mae']))

        csvwriter.writerow(map(lambda x: [x], errors['r2_metric']))
        csvwriter.writerow(map(lambda x: [x], errors['test_r2_metric']))





#-------------------PLOTS---------------------
def pca_error_plot(dimensions, error1, error2, error3, verror1, verror2, verror3, directory):
    make_directory(directory)

    paper_rc = {'lines.linewidth': 2.5}
    sns.set_theme(font_scale=1.2, palette=sns.color_palette("colorblind").as_hex(), rc=paper_rc)
    sns.set_style('white')

    plt.plot(range(1, dimensions+1), error1, '*-')
    plt.plot(range(1, dimensions+1), verror1, '.-')

    plt.plot(range(1, dimensions+1), error2, '*-')
    plt.plot(range(1, dimensions+1), verror2, '.-')

    plt.plot(range(1, dimensions+1), error3, '*-')
    plt.plot(range(1, dimensions+1), verror3, '.-')

    plt.title('PCA mit verschiedenen Dimensionen')
    plt.ylabel('Fehler')
    plt.xlabel('Dimension')
    plt.legend(['mse_train', 'mse_test', 'mae_train', 'mae_test', 'r2_train', 'r2_test'], loc='upper left')
    plt.grid(True)
    plt.savefig('{}/pca.pdf'.format(directory))
    plt.savefig('{}/pca.png'.format(directory))
    plt.close()

def pca_error_plot_lines(dimensions, error1, error2, error3, verror1, verror2, verror3, directory):
    make_directory(directory)

    paper_rc = {'lines.linewidth': 1}
    sns.set_theme(font_scale=1.2, palette=sns.color_palette("colorblind").as_hex(), rc=paper_rc)
    sns.set_style('white')

    plt.plot(range(1, dimensions+1), error1)
    plt.plot(range(1, dimensions+1), verror1)

    plt.plot(range(1, dimensions+1), error2)
    plt.plot(range(1, dimensions+1), verror2)

    plt.plot(range(1, dimensions+1), error3)
    plt.plot(range(1, dimensions+1), verror3)

    plt.title('PCA mit verschiedenen Dimensionen')
    plt.ylabel('Fehler')
    plt.xlabel('Dimension')
    plt.legend(['mse_train', 'mse_test', 'mae_train', 'mae_test', 'r2_train', 'r2_test'], loc='lower right')
    plt.grid(True)
    plt.savefig('{}/pca.pdf'.format(directory))
    plt.savefig('{}/pca.png'.format(directory))
    plt.close()

def auto_error_plot(history, metric, error, title, directory):
    sns.set()
    sns.set_context('paper',
                    font_scale=2)  # this makes the font and scatterpoints much smaller, hence the need for size adjustemnts
    sns.set_style('white')
    make_directory(directory)
    plt.plot(history.history[metric])
    val = 'val_{}'.format(metric)
    plt.plot(history.history[val])
    plt.title(title)
    plt.ylabel(error)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid(True)
    plt.savefig('{}auto_{}_{}.png'.format(directory, error, title))
    plt.savefig('{}auto_{}_{}.pdf'.format(directory, error, title))
    plt.close()

def autoencoder_history_lines(history, error, title, directory):
    #sns.set()
    make_directory(directory)
    paper_rc = {'lines.linewidth': 2.0}
    sns.set_theme(font_scale=1.2, palette=sns.color_palette("colorblind").as_hex(), rc = paper_rc)
    sns.set_style('white')
    #sns.set_palette()

    plt.rc_context(paper_rc)
    plt.ylim([0,1])
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.plot(history.history['r2_metric'])
    plt.plot(history.history['val_r2_metric'])
    plt.title('Vergleich unterschiedlicher Fehlermaße')
    plt.ylabel('Fehler')
    plt.xlabel('Epochen')
    plt.legend(['mse_train', 'mse_val', 'mae_train', 'mae_val', 'r2_train', 'r2_val'], loc='upper right')
    plt.grid(True)

    plt.savefig('{}/auto_{}_{}.png'.format(directory, error, title),bbox_inches = "tight", pad_inches=0.2)
    plt.savefig('{}/auto_{}_{}.pdf'.format(directory, error, title), bbox_inches = "tight", pad_inches=0.2)
    plt.close()

def autoencoder_history(history, error, title, directory):
    paper_rc = {'lines.linewidth': 2.5}
    sns.set_theme(font_scale=1.1, palette=sns.color_palette("colorblind").as_hex(), rc=paper_rc)
    sns.set_style('white')

    plt.rc_context(paper_rc)
    make_directory(directory)
    plt.ylim([0,1])
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.plot(history.history['r2_metric'])
    plt.plot(history.history['val_r2_metric'])

    plt.ylabel('Fehler')
    plt.xlabel('Epochen')
    plt.legend(['mse_train', 'mse_val', 'mae_train', 'mae_val', 'r2_train', 'r2_val'], loc='upper left')
    plt.grid(True)

    plt.savefig('{}/auto_{}_{}.png'.format(directory, error, title), bbox_inches="tight", pad_inches=0.2)
    plt.savefig('{}/auto_{}_{}.pdf'.format(directory, error, title), bbox_inches="tight", pad_inches=0.2)
    plt.close()




#---------------------LEARNING CURVE------------------------
def plot_learning_curve(error_name, training_sizes, error):
    plt.ylabel(error_name)
    plt.xlabel('Training Sizes')
    plt.title('Learning Curve')
    plt.gcf().subplots_adjust(left=0.15)
    plt.plot(training_sizes, error)
    plt.grid(True)
    plt.savefig('learningcurve_{}.png'.format(error_name))
    plt.close()




#------------------LATENT DIMENSION------------------------------
def plot_latent_dim(dimensions, errors, title, directory):
    paper_rc = {'lines.linewidth': 2.0}
    sns.set_theme(font_scale=1.1, palette=sns.color_palette("colorblind").as_hex(), rc=paper_rc)
    sns.set_style('white')

    plt.rc_context(paper_rc)
    make_directory(directory)
    #plt.ylabel('Metriken')
    plt.xlabel('Reduzierte Dimension')
    plt.title('Vergleich der Fehlermetriken bei \n unterschiedlichen reduzierten Dimensionen')
    plt.gcf().subplots_adjust(left=0.15)

    plt.plot(range(1,dimensions), errors['mse'], '*-')
    plt.plot(range(1,dimensions), errors['val_mse'], '.-', )
    plt.plot(range(1, dimensions), errors['test_mse'], 'o-')

    plt.plot(range(1,dimensions), errors['mae'], '*-')
    plt.plot(range(1,dimensions), errors['val_mae'], '.-')
    plt.plot(range(1, dimensions), errors['test_mae'], 'o-')

    plt.plot(range(1,dimensions), errors['r2_metric'], '*-')
    plt.plot(range(1,dimensions), errors['val_r2_metric'], '.-')
    plt.plot(range(1, dimensions), errors['test_r2'], 'o-')

    plt.legend(['mse_train', 'mse_val', 'mse_test' ,'mae_train', 'mae_val', 'mae_test', 'r2_train', 'r2_val', 'r2_test'], loc='upper left')
    plt.grid(True)
    plt.savefig('{}/latent_dim_{}.png'.format(directory, title))
    plt.savefig('{}/latent_dim_{}.pdf'.format(directory, title))
    plt.close()




#----------------------HISTOGRAM------------------------
def plot_histogram(directory, filename, prediction, gap):
    paper_rc = {'lines.linewidth': 2.5}
    sns.set_theme(font_scale=1.25, palette=sns.color_palette("colorblind").as_hex(), rc=paper_rc)
    sns.set_style('white')

    h_pred = prediction
    bottom, top = calculate_linespace(prediction)
    bins = np.linspace(bottom, 0.04, gap)
    plt.hist(h_pred, bins=bins, alpha=0.3)  # the blue one
    plt.title('RMSD Werteverteilung')
    plt.xlabel('RMSD in nm')
    plt.ylabel('Häufigkeit')

    pdf = 'pdf'
    png = 'png'
    plt.tight_layout()
    plt.savefig(directory + '/' + filename + '.' + pdf, format=pdf)
    plt.savefig(directory + '/' + filename + '.' + png, format=png)

def calculate_linespace(axis):
    smallest_value = np.amin(axis, axis=None, out=None)
    largest_value = np.amax(axis, axis=None, out=None)
    return smallest_value, largest_value