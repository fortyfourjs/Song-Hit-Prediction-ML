from turtledemo.chaos import plot

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
from sklearn.model_selection import cross_val_score #cross validation
from sklearn.metrics import RocCurveDisplay #roc



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
print("\n === LR classification report ===")
print(classification_report(y_test, pred_log))

#Random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
pred_rf = rf_model.predict(x_test)
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
cm_knn = confusion_matrix(y_test, pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot()
plt.title("Confusion Matrix KNN")
plt.show()
print(classification_report(y_test, pred_knn))


#SVM
svm = SVC()
svm.fit(x_train, y_train)
pred_svm = svm.predict(x_test)
cm_svm = confusion_matrix(y_test, pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
disp_svm.plot()
plt.title("Confusion Matrix SVM")
plt.show()
print(classification_report(y_test, pred_svm))
#cross validation 5 times
for name, model_cv in [("LR", model), ("RF", rf_model), ("KNN", knn), ("SVM", svm)]:
    scores = cross_val_score(model_cv, x_train, y_train, cv=5, scoring="accuracy")
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

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

#ROC Curve
fig, ax = plt.subplots(figsize=(8,6))
for name, clf, pred in [
    ("Logistic Regression", model, pred_log),
    ("Random Forest", rf_model, pred_rf),
    ("KNN", knn, pred_knn),
    ("SVM", svm, pred_svm)
]:
    RocCurveDisplay.from_estimator(clf, x_test, y_test, ax=ax, name=name)
plt.title("ROC Curve - Comparatie modele")
plt.show()

rezultate = {
    "Logistic Regression": accuracy_score(y_test, pred_log),
    "Random Forest": accuracy_score(y_test, pred_rf),
    "KNN": accuracy_score(y_test, pred_knn),
    "SVM": accuracy_score(y_test, pred_svm)
}
plt.figure(figsize=(8,6))
plt.barh(list(rezultate.keys()), list(rezultate.values()), color="steelblue")
plt.xlim(0.5, 1.0)
plt.xlabel("Precizie")
plt.title("Comparatie modele - Precizie")
plt.tight_layout()
plt.show()


print("\nComparatie lr vs rf vs knn vs svm:")
print(f"Logistic regression:{accuracy_score(y_test, pred_log):.3f}")
print(f"Random Forest:{accuracy_score(y_test, pred_rf):.3f}")
print(f"KNN:{accuracy_score(y_test, pred_knn):.3f}")
print(f"SVM:{accuracy_score(y_test, pred_svm):.3f}")
