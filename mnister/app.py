from flask import Flask, render_template, url_for, jsonify, request 
from mnister.lib import mnist

app = Flask(__name__)
mnist.set_app(app)


def render_response(result=None, error=None):
    """ Render a server response, as HTML unless json is requested """
    if request_wants_json():
        out = {}
        if result:
            out['class'] = result
        if error:
            out['error'] = result
            return jsonify(out)
    else:
        return render_template('index.html', numbers=make_num_urls(), error=error, result=result)


def request_wants_json():
    """ Return True if the request is for JSON """ 
    # http://flask.pocoo.org/snippets/45/
    best = request.accept_mimetypes \
        .best_match(['application/json', 'text/html'])
    return best == 'application/json' and \
        request.accept_mimetypes[best] > \
        request.accept_mimetypes['text/html']

def make_num_urls():
    """ Return some static image URLs... """
    nums = ['3', '1', '4', '1_2', '5', '9']
    return [url_for('static', filename='{}.png'.format(x)) for x in nums]

@app.route("/")
def index():
    """ Render the homepage """
    return render_response()

@app.route("/mnist/classify", methods=['POST'])
def classify():
    """ Get the POSTed image file, send it to mnist for classification """
    if 'image' not in request.files or not request.files['image'].filename:
        raise mnist.MNISTException('Missing image')

    image_class = mnist.classify_file(request.files['image'])
    return render_response(result=image_class)

@app.errorhandler(mnist.MNISTException)
def handle_exception(error):
    """ Send a nice error response for invalid input """
    return render_response(error=str(error)), 404

@app.before_first_request
def ensure_setup():
    """ Ensure that mnist initialised if `flask init` doesn't get run """
    mnist.get_classifier()

@app.cli.command()
def init():
    """ Set up the `flask init` CLI command """
    mnist.get_classifier()
