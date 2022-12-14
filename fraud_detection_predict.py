# # Load the model
import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'fraud_model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)



app = Flask('fraud_detection_service')

@app.route('/fraud_detection_predict', methods = ['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0,1]
    fraud = y_pred >= 0.5

    result = {
        'fraud_probability': float(y_pred),
        'fraud': bool(fraud)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port = 9696)