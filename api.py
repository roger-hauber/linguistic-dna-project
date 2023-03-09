from fastapi import FastAPI
from flask import request
from fastapi import FastAPI, File, UploadFile
from preproc import *
import io


app = FastAPI()

app.state.model = tf.keras.models.load_model('my_model.h5')


@app.post("/uploadfile")
async def create_upload_file(wav: bytes = File(...)):

    model = app.state.model


    res_arr = preprocess(io.BytesIO(wav))
    print(type(res_arr[0]))
    print(res_arr.shape)
    res_lst = list(res_arr)
    print(res_lst[0][0])
    print(type(res_lst[0][0]))
    #print(res_lst)
    resp_dict = dict(resp=float(res_lst[0][0]))


    return resp_dict




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

    hard_code your dictionary function:


    return {'format':'as we like'}
"""

@app.get('/')
def root():
    return {"British":"50%",
            "American": "20%",
            "Australian": "10%",
            "Canadian": "20%"}
