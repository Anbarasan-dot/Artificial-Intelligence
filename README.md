<!DOCTYPE html>
<html>
<head>
    <title>My Technology Blog</title>
</head>
<body>

    <h1>Welcome to My Blog</h1>

    <p>This is my technology blog.</p>

    <a href="https://www.google.com"
       title="Visit Google">
       Click Here
    </a>

    <h2 id="about">About Us</h2>
    <p>Welcome to my website.</p>

    <a href="#about">Go to About</a>

</body>
</html>    
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([35, 40, 50, 60, 65])

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[6]])

print("Predicted Score:", round(prediction[0], 2))
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X = iris.data
y = iris.target

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)

prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])

print("Predicted Class:",
      iris.target_names[prediction[0]])
      import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

x_train = x_train / 255.0
x_test = x_test / 255.0

model = models.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)

loss, accuracy = model.evaluate(x_test, y_test)

print("Accuracy:", accuracy)
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

X = iris.data
y = iris.target

model = DecisionTreeClassifier()
model.fit(X, y)

prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])

print("Predicted Class:",
      iris.target_names[prediction[0]])
      from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

texts = [
    "I love this product",
    "This is wonderful",
    "I hate this product",
    "This is terrible"
]

labels = [
    "Positive",
    "Positive",
    "Negative",
    "Negative"
]

model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression()
)

model.fit(texts, labels)

result = model.predict(["This product is wonderful"])

print("Sentiment:", result[0])
import random

def fitness(x):
    return x * x

population = [random.randint(0, 31) for i in range(10)]

for generation in range(30):

    population.sort(key=fitness, reverse=True)

    parents = population[:5]
    new_population = parents[:]

    while len(new_population) < 10:

        p1, p2 = random.sample(parents, 2)

        child = (p1 + p2) // 2

        if random.random() < 0.2:
            child = random.randint(0, 31)

        new_population.append(child)

    population = new_population

best = max(population, key=fitness)

print("Best Value:", best)
print("Maximum Fitness:", fitness(best))
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

data = load_breast_cancer()

X = data.data
y = data.target

model = SVC(kernel='linear')

model.fit(X, y)

prediction = model.predict([X[0]])

print("Predicted Class:",
      data.target_names[prediction[0]])
      from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()

X = iris.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)

X_reduced = pca.fit_transform(X_scaled)

print("Original Shape:", X.shape)
print("Reduced Shape:", X_reduced.shape)

print("Variance Retained:",
      round(sum(pca.explained_variance_ratio_), 3))
      import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.DataFrame({
    'Movie_A': [5, 4, 1, None],
    'Movie_B': [4, 5, 2, None],
    'Movie_C': [None, 2, 5, 4],
    'Movie_D': [2, None, 4, 5]
}, index=['User1', 'User2', 'User3', 'User4'])

similarity = cosine_similarity(ratings.fillna(0))

sim = pd.Series(similarity[0], index=ratings.index)
sim = sim.drop('User1')

unseen = ratings.columns[ratings.loc['User1'].isna()]

scores = {}

for movie in unseen:
    known = ratings[movie].dropna()
    weights = sim.loc[known.index]

    scores[movie] = (known * weights).sum() / weights.sum()

print("Recommended Movie:", max(scores, key=scores.get))
