import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
data = pd.read_csv('heart.csv')
y = data["target"]
X = data.drop('target',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)
pipe = Pipeline([('scaler', StandardScaler()), ('Logistic Regression', LogisticRegression())])
pipe.fit(X_train, y_train)
pickle.dump(pipe, open('pipemodel.pkl','wb'))