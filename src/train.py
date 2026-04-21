from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

data = load_iris("C:\\Users\\ASUS\\OneDrive\\Desktop\\IRIS.csv")
X, y = data.data, data.target

model = DecisionTreeClassifier()
model.fit(X, y)

print("Model trained successfully")