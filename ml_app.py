from json import load, dumps

with open('categories.json', 'r') as file:
    estabilishments_and_categories: dict = load(file)


# import pandas as pd

# df = pd.read_csv('Nubank.csv')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer


categorie_data = list(estabilishments_and_categories.values())
estabilishments_data = list(estabilishments_and_categories.keys())

print(categorie_data, '\n')
print(estabilishments_data, '\n')

mlb = MultiLabelBinarizer()
binary_categories = mlb.fit_transform(categorie_data)

vactorizer = CountVectorizer()
X = vactorizer.fit_transform(estabilishments_data)

model = OneVsRestClassifier(LinearSVC())
model.fit(X, binary_categories)

# estabilishments_test = list(df['TRANSACTION'].values)
estabilishments_test = ['99app', 'Cinemark WestPlaza']
X_test = vactorizer.transform(estabilishments_test)
binary_predict = model.predict(X_test)


predictable_categories = mlb.inverse_transform(binary_predict)
print(f'The predictable categorias for estabilishments: {estabilishments_test} are: {predictable_categories}')