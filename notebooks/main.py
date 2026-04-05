
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.metrics import RocCurveDisplay #roc
from sklearn.svm import LinearSVC



#Incarcare set de date
df = pd.read_csv("../data/spotify_dataset.csv")
df = df.dropna().copy()
df = df.drop_duplicates(subset=["track_id"])
print(f"Shape dupa deduplicare: {df.shape}")

features = ["danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "duration_ms",
            "explicit",
            "key",
            "mode",
            "time_signature"]
x = df[features]

print("Set incarcat")
print(df.shape)
print(df.head())

#target variable
threshold = df["popularity"].quantile(0.75)
df["hit"] = (df["popularity"] >= threshold).astype(int)
y = df["hit"]
sns.countplot(x=df["hit"])
plt.title("hit vs non hit distribution")
plt.show()

print("Hit threshold:", threshold)
print(df["hit"].value_counts())

#train/test split 80/20 + scalare
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#Logistic Regression
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(x_train, y_train)

pred_log = model.predict(x_test)
print("\n === LR classification report ===")
print(classification_report(y_test, pred_log))

#Random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
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
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(x_train, y_train)
pred_knn = knn.predict(x_test)
cm_knn = confusion_matrix(y_test, pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot()
plt.title("Confusion Matrix KNN")
plt.show()
print(classification_report(y_test, pred_knn))


#SVM
svm = LinearSVC(class_weight="balanced", max_iter=2000) #linearsvc timesave, svc too slow
svm.fit(x_train, y_train)
pred_svm = svm.predict(x_test)
cm_svm = confusion_matrix(y_test, pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
disp_svm.plot()
plt.title("Confusion Matrix SVM")
plt.show()
print(classification_report(y_test, pred_svm))

pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
pipe_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1))
])
pipe_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
])
pipe_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearSVC(class_weight="balanced", max_iter=2000))
])
#Cross validation pe x(date initiale)
print("\n===Cross Validation(5fold)===")
for name, pipe in [("LR", pipe_lr), ("RF",pipe_rf), ("KNN", pipe_knn), ("SVM", pipe_svm)]:
    scores = cross_val_score(pipe, x, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")
#importanta proprietati random forest
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

sns.histplot(df["popularity"], bins=50, ax=axes[1], log_scale=True)
axes[1].set_title("Popularity distribution")

plt.tight_layout()
plt.show()
#coeficienti logistic regression
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
    # ("SVM", svm, pred_svm) incompatibil roc
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
