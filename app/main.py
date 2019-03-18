from flask import Flask, request, jsonify
from google.oauth2 import service_account
import numpy as np
from google.cloud import storage
import pickle


APP = Flask(__name__)


def download_from_cloud_storage(credentials):
    client = storage.Client(project=credentials.project_id,
                            credentials=credentials)
    bucket = client.get_bucket('search-ranking-state')
    blob = bucket.get_blob('model_predict.h5')
    blob.download_to_filename('/models/model_predict.h5')
    blob = bucket.get_blob('model_predict_arch.json')
    blob.download_to_filename('/models/model_predict_arch.json')


@APP.before_first_request
def load_model():
    global model
    with open('/models/lte_fe34789_20190318101552947002.pkl', 'rb') as f:
        model = pickle.load(f)


@APP.route("/prediction", methods=['POST'])
def prediction():
    if request.method == 'POST':
        prediction_data = request.json
        print(prediction_data)
        result = np.argmax(model.predict(np.random.random(4).reshape(1, -1)))
    return jsonify({'result': int(result)})


if __name__ == "__main__":
    credenials = service_account.Credentials.from_service_account_file(
        '/secrets/private-key.json')
    download_from_cloud_storage(credenials)
    APP.run(host='0.0.0.0', port=5000, debug=True)
