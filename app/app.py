# Common python package imports.
from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
import sys
from io import BytesIO
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


@app.route('/a')
def a():
    print(path)
    time.sleep(5)
    print(path)
    img = open_image('C:/Users/Jason/Pictures/d12.jpg')
    pred_class, pred_idx, outputs = learn.predict(img)
    print(pred_class, file=sys.stderr)
    print(pred_class)
    return render_template('docs.html')


@app.route('/classify', methods=["POST"])
async def classify(request):
    print("Hello world", file=sys.stderr)
    print("Classify called")
    print(request.json, file=sys.stderr)

    # Get Request Form data
    data = await request.form()
    # request.form()
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


@app.route("/form2")
def form2():
    print("Form 2 called")
    return render_template('form2.html')


@app.route('/api', methods=['GET'])
def api():
    # Handle empty requests.
    print('Request received!', file=sys.stderr)
    #     print(request.json, file=sys.stderr)
    #     print('args above!', file=sys.stderr)
    #     print('request.json: ' + request.json, file=sys.stderr)
    #     if not request.json:
    #         return jsonify({'error': 'no request received'})
    #
    #     # Parse request args into feature array for prediction.
    x_list, missing_data = parse_args(request.json)
    x_array = np.array([x_list])

    # Predict on x_array and return JSON response.
    estimate = int(MODEL.predict(x_array)[0])
    response = dict(ESTIMATE=estimate, MISSING_DATA=missing_data)

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
