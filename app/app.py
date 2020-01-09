# app/app.py

# Common python package imports.
from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
import sys
from werkzeug.utils import secure_filename

# Initialize the app and set a secret_key.
app = Flask(__name__)
app.secret_key = 'something_secret'

# Load the pickled model.
MODEL = pickle.load(open('dice.pkl', 'rb'))


@app.route('/')
def docs():
    print("Root called", file=sys.stderr)
    return render_template('docs.html')


@app.route('/upload')
def upload():
    return render_template('image.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return 'file uploaded successfully'


@app.route('/classify', methods=["POST"])
def classify():
    print("Hello world", file=sys.stderr)
    print("Classify called")
    print(request.json, file=sys.stderr)

@app.route("/form2")
def form2(request):
    return HTMLResponse(
        """
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)

@app.route("/form")
def form():
    print("Form called")
    return render_template('form.html')


@app.route("/uploadStar", methods=["POST"])
async def upload_star():
    print("upload_star called", file=sys.stderr)
    data = await request.form()
    print(data)
    bytes = await (data["file"].read())
    print(bytes)
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    print("predict image from bytes", file=sys.stderr)
    print(bytes, file=sys.stderr)

    losses = img.predict(cat_learner)

    pred_class, pred_idx, outputs = MODEL.predict(img)
    print("predicted class is: " + pred_class, file=sys.stderr)

    return JSONResponse({
        "predictions": sorted(
            zip(cat_learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })


@app.route('/api', methods=['GET'])
def api():
    # Handle empty requests.
    print('Request recieved!', file=sys.stderr)
    print(request.json, file=sys.stderr)
    print('args above!', file=sys.stderr)
    print('request.json: ' + request.json, file=sys.stderr)
    if not request.json:
        return jsonify({'error': 'no request received'})

    # Parse request args into feature array for prediction.
    x_list, missing_data = parse_args(request.json)
    x_array = np.array([x_list])

    # Predict on x_array and return JSON response.
    estimate = int(MODEL.predict(x_array)[0])
    response = dict(ESTIMATE=estimate, MISSING_DATA=missing_data)

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
