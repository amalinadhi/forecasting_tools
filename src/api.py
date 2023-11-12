import pandas as pd
import utils
import subprocess
from json import loads

from fastapi import FastAPI
import uvicorn


# Load results
CONFIG_FILE = utils.config_load()
MODEL = utils.pickle_load(CONFIG_FILE['best_model_path'])
COLS = CONFIG_FILE["features_col"]

# FastAPI
app = FastAPI()

# First
@app.get('/')
def home():
    return {'message': 'Hello World'}

@app.post('/train')
def train():
    # Execute the training data
    result = subprocess.run(["sh", "train.sh"], capture_output=True, text=True)
    
    # Get the messages
    res_msg = {"message": None, "error": None}
    try:
        result.check_returncode()
        res_msg["message"] = "Modelling success. The best model is saved"
        return res_msg
    except subprocess.CalledProcessError as e:
        res_msg["error"] = e
        return res_msg

@app.post('/predict')
def predict():
    return_msg = {'results': None, 'error': None}
    try:
        # Load data dulu
        X_test_clean = utils.pickle_load(CONFIG_FILE['X_test_clean_path'])
        
        # Lakukan prediksi
        y_test_pred = pd.DataFrame(MODEL.predict(X_test_clean[COLS]),
                                index=X_test_clean.index)
        
        # Concat dataframe
        data_test_pred = pd.concat((X_test_clean, y_test_pred), axis=1)
        data_test_pred = data_test_pred.rename(columns={0: 'Weekly_Sales'})
        data_test_pred['Date'] = data_test_pred['Date'].astype(str)
        
        # Convert to json
        res = data_test_pred.to_json(orient="split")
        res = loads(res)

        return_msg["results"] = res
    except Exception as e:
        return_msg["error"] = e        

    return return_msg


# Run
if __name__ == '__main__':
    uvicorn.run('api:app',
                host = '127.0.0.1',
                port = 8000)