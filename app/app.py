# Common python package imports.
from flask import Flask, jsonify, request, render_template
from fastai.vision import *

# Initialize the app and set a secret_key.
app = Flask(__name__)
app.secret_key = 'something_secret'

# Load the pickled model.
defaults.device = torch.device('cpu')
path = '.'
learn = load_learner(path, file='dice.pkl')


@app.route('/')
def docs():
    return render_template('docs.html')


@app.route('/upload')
def upload():
    return render_template('image.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        img_bytes = f.read()
        img = open_image(BytesIO(img_bytes))

        pred_class, pred_idx, outputs = learn.predict(img)
        print('Returning: ' + str(pred_class), file=sys.stderr)
        print('Index: ' + str(pred_idx))
        print('Outputs: ' + str(outputs))
        return str(pred_class)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
