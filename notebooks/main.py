import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Incarcare set de date
df = pd.read_csv("../data/cleaned_dataset.csv")
df = df.dropna().copy()

features = ["Danceability",
            "Energy",
            "Loudness",
            "Speechiness",
            "Acousticness",
            "Instrumentalness",
            "Liveness",
            "Valence",
            "Tempo"]
x = df[features]

print("Set incarcat")
print(df.shape)
print(df.head())

#target clasificare stream
threshold = df["Stream"].quantile(0.75)
df["hit"] = (df["Stream"] >= threshold).astype(int)
y = df["hit"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

pred = model.predict(x_test)
print("accuracy:", accuracy_score(y_test, pred))

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.show()