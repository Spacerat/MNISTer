# MNISTer

A simple web service for classifying MNIST digits with reasonable-ish accuracy. Only tested with Python 3.

## Setup
First install the required libraries (Scikit-Learn, SciPy, Flask, Flask-Caching)

	$ pip install -r requirements.txt

## Running
### With the Flask CLI
Then there are two ways to run the server. The first is to use the [flask CLI](http://flask.pocoo.org/docs/0.12/cli/). Open a terminal in the directory of `server.py` and run the following commands:

    $ export FLASK_APP=server.py
    $ export FLASK_DEBUG=1 
    $ flask init
    $ flask run
    
The command `flask init` is a custom command which builds and saves the machine learning model if it not already located at `mnister/lib/model.pkl` (which I have included in the repo). If you choose to run this, be aware that it will take a while.

### Manually

You can run the server directly with python:

    $ python server.py

If you do this when the model file is not present, `flask init` will be run on the first request and it will run much slower because it won't be able to use multiple cores.

## Using the server

Flask will give you the URL of the server, e.g. `http://127.0.0.1:5000`. If you navigate to that and use the HTML form to upload an image, you will get a HTML response. For a JSON response, send a request to the endpoint with the `Accept: application/json` header.

## Comments

The classification algorithm used by this project is optimised for simplicity and training time above accuracy, so the error rate is around 5% (compared to values < 1% found in literature). It is a support vector machine with a polynomial kernel, and grid-search with cross validation is used to choose the polynomial degree and the value of `C`. It is trained with a random subset of the MNIST training set.

If you rebuild the model, the MNIST data will be downloaded directly from mldata.org via scikit-learn's built in function `fetch_mldata('MNIST original')` and stored in a folder called `mnist_cache`.

The model file `model.pkl` itself is simply a pickled scikit-learn classifier object stored on the disk. After it has been loaded it gets stored in a cache in RAM. 