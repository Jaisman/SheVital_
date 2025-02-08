import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('data/Training.csv')
train = train.drop("Unnamed: 133", axis=1, errors='ignore')
le = LabelEncoder()
train["prognosis"] = le.fit_transform(train["prognosis"])
x = train.drop("prognosis", axis=1)
y = train["prognosis"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=42)
model = LogisticRegression(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred) * 100)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
