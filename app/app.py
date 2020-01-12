# Common python package imports.
from flask import Flask, jsonify, request, render_template
import time
from fastai.vision import *

# Initialize the app and set a secret_key.
app = Flask(__name__)
app.secret_key = 'something_secret'

# Load the pickled model.
defaults.device = torch.device('cpu')
MODEL = pickle.load(open('dice.pkl', 'rb'))
path = '.'
learn = load_learner(path, file='dice.pkl')


@app.route('/')
def docs():
    print("Root called", file=sys.stderr)
    return render_template('docs.html')


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return jsonify(prediction)


@app.route('/upload')
def upload():
    return render_template('image.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        print(f)
        print("type of f is:")
        print(type(f))

        img_bytes = f
        print(img_bytes.content_type)
        print(img_bytes.content_length)
        print(img_bytes, file=sys.stderr)
        print('Attempt to open image', file=sys.stderr)

        bytes = f.read()
        print("bytes")
        print(bytes)
        print(type(bytes))

        stream = img_bytes.stream
        print('stream type is:')
        print(type(stream))
        print(stream)
        img = open_image(BytesIO(bytes))

        pred_class, pred_idx, outputs = learn.predict(img)
        print(pred_class, file=sys.stderr)
        print(pred_class)
        print(outputs)
        print(pred_idx)
        return str(pred_class)
        #return jsonify(outputs)


@app.route('/classify', methods=["POST"])
async def classify(request):
    print("Hello world", file=sys.stderr)
    print("Classify called")
    print(request.json, file=sys.stderr)


@app.route("/uploadStar", methods=["POST"])
async def upload_star():
    print("upload_star called", file=sys.stderr)
    data = await request.form()
    print(data)

    # Get image as bytes
    img_bytes = await (data['file'].read())
    bytes = await(data["file"].read())
    print(bytes)

    img_ = open_image(BytesIO(img_bytes))
    img = open_image(BytesIO(bytes))

    prediction = learn.predict(img_)[0]
    pred_class, pred_idx, outputs = learn.predict(img)

    print(prediction)
    print(pred_class)
    print(outputs)

    return jsonify(prediction)


@app.route("/form")
def form():
    print("Form called")
    return render_template('form.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
