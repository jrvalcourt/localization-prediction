import keras
import os
import pickle
import gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
import sklearn.metrics

pickle_dir = 'data/'
x_data_filename     = os.path.join(pickle_dir, 'X.p.gz')
y_data_filename     = os.path.join(pickle_dir, 'Y.p.gz')
names_data_filename = os.path.join(pickle_dir, 'names.p.gz')
features_filename   = os.path.join(pickle_dir, 'features.p.gz')
x_test_filename     = os.path.join(pickle_dir, 'x_test.p.gz')
y_test_filename     = os.path.join(pickle_dir, 'y_test.p.gz')
x_train_filename    = os.path.join(pickle_dir, 'x_train.p.gz')
y_train_filename    = os.path.join(pickle_dir, 'y_train.p.gz')

# load data
x_test   = pickle.load(gzip.open(x_test_filename))
y_test   = pickle.load(gzip.open(y_test_filename))
x_train  = pickle.load(gzip.open(x_train_filename))
y_train  = pickle.load(gzip.open(y_train_filename))
features = pickle.load(gzip.open(features_filename))

model = keras.models.load_model('model.h5')

def make_roc_curves(roc_dir, model, xs, ys):
    if not os.path.exists(roc_dir):
        os.mkdir(roc_dir)
    predictions = model.predict(xs)
    for ii in range(features.shape[0]):
        print(features[ii])
        guess = predictions[:,ii]
        truth = ys[:,ii]
        tpr = []
        fpr = []
        for thresh in np.arange(0, 1, step=0.01):
            t_guess = guess > thresh
            tpr.append(np.sum(np.logical_and(truth, t_guess)) / np.sum(truth))
            fpr.append(np.sum(np.logical_and(np.logical_not(truth), t_guess)) / np.sum(np.logical_not(truth)))
        
        auc = sklearn.metrics.roc_auc_score(truth, guess)
        print(auc)

        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0,1], [0,1], 'k--')
        plt.text(0.65, 0.1, 'AUC = {:0.2f}'.format(auc))
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title(features[ii].decode('utf-8'))
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.tight_layout()
        plt.savefig(os.path.join(roc_dir, features[ii].decode('utf-8') + '.png'), dpi=300)
        plt.close()

make_roc_curves('roc_curves_test',  model, x_test,  y_test)
make_roc_curves('roc_curves_train', model, x_train, y_train)
