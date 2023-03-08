from fastapi import FastAPI
from flask import request
from fastapi import FastAPI, File, UploadFile


app = FastAPI()

#app.state.model = load_model('function that will load our model: either local, psc, or mlflow')



@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        with open(file.filename, 'wb') as f:
            while contents := file.file.read(1024 * 1024):
                f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}



"""
@app.get('/predict')
def predict("data"):

    code here where we get data as input and then have to tiurn it into a dataframe that can be fed to our model
    stored in X_pred
    then use already created function that will preprocess the input
    and then use the model that we got wih load model fucntion and apply it to our X
    return it un the format we want

    model = app.state.model
    assert model is not None

    X_processed_audio = preprocess_features(X_pred)
    y_pred = model.predict(X_processed_audio)

    return {'format':'as we like'}
"""

@app.get('/')
def root():
    return {"British":"50%",
            "American": "20%",
            "Australian": "10%",
            "Canadian": "20%"}
