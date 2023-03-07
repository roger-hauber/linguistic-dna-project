from fastapi import FastAPI

app = FastAPI()
"""
app.state.model = load_model('function that will load our model: either local, psc, or mlflow')

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
    return {"Our":"API"}
