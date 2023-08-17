import joblib
import numpy as np
import pandas as pd

model = joblib.load('logistic_model')

def iris(sepal_length,	sepal_width, petal_length,petal_width):
    data = np.array(['setosa', 'versicolor', 'virginica'])
    x = np.array([sepal_length,	sepal_width, petal_length,petal_width	])
    x = x.reshape((1,4))
    x = pd.DataFrame(x, columns=['sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'])
    flor =  model.predict(x)[0]
    return data[flor]

print(iris(5.9, 9.0, 5.1, 1.8))
