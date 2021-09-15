import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('pipemodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [ x for x in request.form.values()]
    temp = int_features[0:13]
    int_feature = [int(float(x)) for x in temp]
    final_features = [np.array(int_feature)]
    prediction = model.predict(final_features)
    print(int_features)
    print(final_features)
    print(prediction)
    string="Status: {}".format(prediction)
    return render_template('index.html', prediction_text=string)


if __name__ == "__main__":
    app.run(debug=True)