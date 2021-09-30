from flask import Flask,request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pk1','rb'))


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        '#input_data = (req, 0, 100.00, 159, 6107, 0, 0, 0, 0, 0, 1, 0)'
        input_data=(int(request.form.get("PackageID")), int(request.form.get("Category")),
                    float(request.form.get("TotalPrice")), int(request.form.get("District")),
                    int(request.form.get("Taluka")), int(request.form.get("Shipper")),
                    int(request.form.get("Source")), int(request.form.get("State_GJ")),
                    int(request.form.get("State_MH")), int(request.form.get("State_MP")),
                    int(request.form.get("State_RJ")), int(request.form.get("State_UP")))
        input_data_array = np.asarray(input_data)
        input_data_reshaped = input_data_array.reshape(1, -1)
        prediction = model.predict(input_data_reshaped)

        if prediction == 1:
            return render_template('index.html',pred='The Order has chance it will be returned: {}'.format(prediction))
        else:
            return render_template('index.html',pred='The Order is safe to process {}'.format(prediction))
    except:
        return render_template('index.html', pred='Invalid Input')

if __name__ == '__main__':
    app.run(debug=True)
