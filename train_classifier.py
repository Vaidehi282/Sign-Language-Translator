import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


data_dict = pickle.load(open('data.pickle', 'rb'))

X = np.asarray(data_dict['data'])
y = np.asarray(data_dict['labels'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
accuracy = accuracy_score(y_predict, y_test)
print(f"{accuracy*100}% of samples predicted correctly")

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)



