from tensorflow import keras

model_path='/Users/frido/linguistic-dna-project/models/copied_best_CNN.h5'
model = keras.models.load_model(model_path)

metrics = model.evaluate(
        x=X_test,
        y=y_test_cat,
        batch_size=16,
        verbose=0,
        # callbacks=None,
        return_dict=True)

print(metrics['accuracy'])