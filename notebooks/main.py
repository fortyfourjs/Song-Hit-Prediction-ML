import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



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
sns.countplot(x=df["hit"])
plt.title("hit vs non hit distribution")
plt.show()

print("Hit threshold:", threshold)
print(df["hit"].value_counts())


#train/test split 80/20
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

pred_log = model.predict(x_test)
print("logistic regression accuracy:", accuracy_score(y_test, pred_log))
print("\n === LR classification report ===")
print(classification_report(y_test, pred_log))

#Random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
pred_rf = rf_model.predict(x_test)
print("Random Forest accuracy:", accuracy_score(y_test, pred_rf))
cm = confusion_matrix(y_test, pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix Random Forest")
plt.show()

print("\n === RF classification report ===")
print(classification_report(y_test, pred_rf))

#KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
pred_knn = knn.predict(x_test)
print("KNN accuracy:", accuracy_score(y_test, pred_knn))

#SVM
svm = SVC()
svm.fit(x_train, y_train)
pred_svm = svm.predict(x_test)
print("SVM accuracy:", accuracy_score(y_test, pred_svm))

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

coef = model.coef_[0]
plt.figure(figsize=(8,6))
plt.barh(features, coef)
plt.title("Coeficienti Logistic Regression")
plt.xlabel("Impact on hit probability")
plt.show()

print("\nComparatie lr vs rf vs knn vs svm:")
print(f"Logistic regression:{accuracy_score(y_test, pred_log):.3f}")
print(f"Random Forest:{accuracy_score(y_test, pred_rf):.3f}")
print(f"KNN:{accuracy_score(y_test, pred_knn):.3f}")
print(f"SVM:{accuracy_score(y_test, pred_svm):.3f}")