from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from azureml.core.run import Run
import os
import numpy as np
import joblib

os.makedirs('./outputs', exist_ok=True)   # to save model in the outputs folder so it automatically get uploaded

X, y = load_diabetes(return_X_y=True)

run = Run.get_context()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for alpha in np.arange(0.0, 1.0, 0.05):
    reg = Ridge(alpha=alpha)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    mse = mean_squared_error(preds, y_test)
    run.log('alpha', alpha)
    run.log('mse', mse)
    model_file_name = 'ridge_{0:.2f}.pkl'.format(alpha)
    with open(model_file_name, 'wb') as file:
        joblib.dump(value=reg, filename=os.path.join('./outputs/', model_file_name))

    print('alpha is {0:.2f}, and mse is {1:0.2f}'.format(alpha, mse))
