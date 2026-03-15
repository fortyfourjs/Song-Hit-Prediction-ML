import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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

#target variable
threshold = df["Stream"].quantile(0.75)
df["hit"] = (df["Stream"] >= threshold).astype(int)
y = df["hit"]

print("Hit threshold:", threshold)
print(df["hit"].value_counts())

#train/test split 80/20
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

#Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

pred_log = model.predict(x_test)
print("logistic regression accuracy:", accuracy_score(y_test, pred_log))

#Random forest
rf_model = RandomForestClassifier(n_estimators=42)
rf_model.fit(x_train, y_train)
pred_rf = rf_model.predict(x_test)
print("Random Forest accuracy:", accuracy_score(y_test, pred_rf))
#importanta proprietati
importance = rf_model.feature_importances_
indices = np.argsort(importance)

plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importance[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title("Feature importances")
plt.show()
#Visualization
fig, axes = plt.subplots(1,2, figsize = (16,6))

sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=axes[0])
axes[0].set_title("Correlation heatmap")

sns.histplot(df["Stream"], bins=50, ax=axes[1], log_scale=True)
axes[1].set_title("Stream distribution")

plt.tight_layout()
plt.show()

print("\nComparatie lr vs rf:")
print("Logistic regression:", accuracy_score(y_test, pred_log))
print("Random Forest:", accuracy_score(y_test, pred_rf))