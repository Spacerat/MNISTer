import pickle 
import logging
import os
import scipy.misc
from sklearn.datasets import fetch_mldata
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from flask_caching import Cache

model_path = os.path.join('mnister','lib','model.pkl')
logger = logging.getLogger(__name__)
cache = Cache(config={'CACHE_TYPE': 'simple'})

class MNISTException(Exception):
    pass

def read_input_file(f):
    """ Read the input image. Raise an exception if it is not a 28x28 image file. """
    try:
        image_data = scipy.misc.imread(f, flatten=True)
    except OSError:
        raise  MNISTException("Invalid image file")
    if image_data.shape != (28, 28):
        raise MNISTException("Image size must be 28x28")
    return image_data

def classify_file(f):
    """ Run the image file through our support vector machine to get a class """
    image_data = read_input_file(f)
    vector = image_data.reshape((28*28,))
    clf = get_classifier()
    return int(clf.predict([vector])[0])


def get_data():
    """ Get the MNIST data, split into two subsets. """

    # We just load the MNIST data using scikit-learn's dataset downloader
    mnist = fetch_mldata('MNIST original', data_home='mnist_cache')

    # The MNIST dataset specifies that only the first 60000 are training samples.
    X_train = mnist.data[:60000]
    y_train = mnist.target[:60000]

    # We split the dataset into two subsets:
    # - a small subset for hyperparameter optimisation
    # - a larger subset for training the final model
    return train_test_split(X_train, y_train, train_size=0.05, test_size=0.2, stratify=y_train)

def make_classifier():
    """ Build the actual ML model """
    x_hyperopt, x_fulltrain, y_hyperopt, y_fulltrain = get_data()

    # We'll use a support vector machine with a polynomial kernel. These seem to get reasonable 
    # results in literature without being too complicated to set up.
    classifier = SVC(kernel='poly', cache_size=2048)

    # We'll use GridSearchCV to pick the best C and degree, by using cross-validation to test
    # each combination.
    param_grid = {
        'C': [1e-8, 1e-4, 1, 1e4, 1e8],
        'degree': [3, 2]
    }
    optimiser = GridSearchCV(classifier, param_grid, verbose=5, n_jobs=4, refit=True)
    optimiser.fit(x_hyperopt, y_hyperopt)

    # GridSearchCV updates the classifier's parameters, now we just have to train it with more
    # data and save the model.
    classifier = optimiser.best_estimator_
    classifier.fit(x_fulltrain, y_fulltrain)

    # Save the model using pickle
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f, -1)
    return classifier

def set_app(app):
    """ Set up the cache system """
    cache.init_app(app)

@cache.cached(timeout=600, key_prefix='model')
def get_classifier():
    """ Get the model: either from memory (the cache), disk, or rebuild it if it's missing """
    if os.path.exists(model_path):
        logger.info("Loading pickled classifier...")
        with open(model_path, 'rb') as f:
            classifier = pickle.load(f)
        logger.info("Classifier loaded")
    else:
        logger.warning("Classifier not present, rebuilding...")
        classifier = make_classifier()
        logger.info("Rebuild complete")
    return classifier
