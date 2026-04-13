import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
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
from xgboost import XGBClassifier
from scipy.stats import randint
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_auc_score, matthews_corrcoef
os.makedirs("../plots", exist_ok=True)



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
plt.savefig("../plots/hit_non_hit_distribution.pdf", dpi=300, bbox_inches="tight")
plt.show()

print("Hit threshold:", threshold)
print(df["hit"].value_counts())

#train/test split 80/20 + scalare
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
ratio = neg / pos
print(f"scale_pos_weight este: {ratio:.2f}")


#Logistic Regression
lr_param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs", "liblinear"],
    "penalty": ["l2"]
}
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    lr_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)
lr_grid.fit(x_train, y_train)
print("Best LR Params:", lr_grid.best_params_)

best_lr = lr_grid.best_estimator_
pred_log = best_lr.predict(x_test)

print("\n === LR classification report ===")
print(classification_report(y_test, pred_log))


#Random Forest
rf_param_grid = {
    "n_estimators": [100, 200, 300], #add 300 for proper grid
    "max_depth": [None, 10, 20, 30], #add 30 for proper grid
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4] #add 4 for proper grid
}
rf_grid = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=3, #change 3 to 5 for 5 folds, 3 is performance only goes from 540 fits to 108
    scoring="f1",
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(x_train, y_train)
print("Best RF Params:", rf_grid.best_params_)
print("Best RF F1 scores:", rf_grid.best_score_)

best_rf = rf_grid.best_estimator_
pred_rf = best_rf.predict(x_test)

cm = confusion_matrix(y_test, pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix Random Forest")
plt.savefig("../plots/cm_random_forest.pdf", dpi=300, bbox_inches="tight")
plt.show()
print("\n === RF classification report ===")
print(classification_report(y_test, pred_rf))

#KNN
knn_param_grid = {
    "n_neighbors": [3, 5, 7, 11, 15, 21],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean","manhattan"]
}
knn_grid = GridSearchCV(
    KNeighborsClassifier(n_jobs=-1),
    knn_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)
knn_grid.fit(x_train, y_train)
print("Best KNN params:", knn_grid.best_params_)

best_knn = knn_grid.best_estimator_
pred_knn = best_knn.predict(x_test)

cm_knn = confusion_matrix(y_test, pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot()
plt.title("Confusion Matrix KNN")
plt.savefig("../plots/cm_KNN.pdf", dpi=300, bbox_inches="tight")
plt.show()
print("\n === KNN classification report ===")
print(classification_report(y_test, pred_knn))


#SVM
svm_param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "max_iter": [2000]
}
svm_grid = GridSearchCV(
    LinearSVC(class_weight="balanced"),
    svm_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)
svm_grid.fit(x_train, y_train)
print("Best SVM params:", svm_grid.best_params_)

best_svm = svm_grid.best_estimator_
pred_svm = best_svm.predict(x_test)

cm_svm = confusion_matrix(y_test, pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
disp_svm.plot()
plt.title("Confusion Matrix SVM")
plt.savefig("../plots/cm_SVM.pdf", dpi=300, bbox_inches="tight")
plt.show()
print("\n === SVM classification report ===")
print(classification_report(y_test, pred_svm))

#XGBoost
xgb_param_dist = {
    "n_estimators": randint(100, 400),
    "max_depth": randint(3, 10),
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]

}
xgb_rand = RandomizedSearchCV(
    XGBClassifier(scale_pos_weight=ratio, eval_metric="logloss", random_state=42, n_jobs=-1),
    xgb_param_dist,
    n_iter=15, #change to 30 for proper grid
    cv=3, #change to 5, goes from 150 fits to 45
    scoring="f1",
    n_jobs=-1,
    verbose=1,
    random_state=42
)
xgb_rand.fit(x_train, y_train)
print("Best XGBoost params:", xgb_rand.best_params_)

best_xgb = xgb_rand.best_estimator_
pred_xgb =best_xgb.predict(x_test)

cm_xgb = confusion_matrix(y_test, pred_xgb)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb)
disp_xgb.plot()
plt.title("Confusion Matrix XGBoost")
plt.savefig("../plots/cm_XGBoost.pdf", dpi=300, bbox_inches="tight")
plt.show()

print("\n === XGBoost classification report ===")
print(classification_report(y_test, pred_xgb))

pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", best_lr)
])
pipe_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("model", best_rf)
])
pipe_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("model", best_knn)
])
pipe_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("model", best_svm)
])
pipe_xgb = Pipeline([
    ("scaler", StandardScaler()),
    ("model", best_xgb)
])

#Cross validation pe x(date initiale)
print("\n===Cross Validation(5fold)===")
for name, pipe in [("LR", pipe_lr), ("RF",pipe_rf), ("KNN", pipe_knn), ("SVM", pipe_svm), ("XGB", pipe_xgb)]:
    results = cross_validate(pipe, x, y, cv=5, scoring=["f1", "accuracy"], n_jobs=-1)
    f1 = results["test_f1"].mean()
    accuracy = results["test_accuracy"].mean()
    print(f"{name}: Accuracy={accuracy:.3f} F1={f1:.3f}")

#importanta proprietati random forest
importance = best_rf.feature_importances_
indices = np.argsort(importance)

plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importance[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title("Feature importances")
plt.savefig("../plots/feature_importance.pdf", dpi=300, bbox_inches="tight")
plt.show()

#Visualization
fig, axes = plt.subplots(1,2, figsize = (16,6))

sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=axes[0])
axes[0].set_title("Correlation heatmap")

sns.histplot(df["popularity"], bins=50, ax=axes[1], log_scale=True)
axes[1].set_title("Popularity distribution")

plt.tight_layout()
plt.savefig("../plots/correlation_popularity_distribution.pdf", dpi=300, bbox_inches="tight")
plt.show()

#coeficienti logistic regression
coef = best_lr.coef_[0]
plt.figure(figsize=(8,6))
plt.barh(features, coef)
plt.title("Coeficienti Logistic Regression")
plt.xlabel("Impact on hit probability")
plt.savefig("../plots/coef_logistic_regression.pdf", dpi=300, bbox_inches="tight")
plt.show()

#ROC Curve
fig, ax = plt.subplots(figsize=(8,6))
for name, clf, pred in [
    ("Logistic Regression", best_lr, pred_log),
    ("Random Forest", best_rf, pred_rf),
    ("KNN", best_knn, pred_knn),
    # ("SVM", svm, pred_svm) incompatibil roc
    ("XGB", best_xgb, pred_xgb)
]:
    RocCurveDisplay.from_estimator(clf, x_test, y_test, ax=ax, name=name)
plt.title("ROC Curve - Comparatie modele")
plt.savefig("../plots/ROC_curve.pdf", dpi=300, bbox_inches="tight")
plt.show()

#Threshold tuning curve
y_proba = best_xgb.predict_proba(x_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precision[:-1]*recall[:-1]) / (precision[:-1]+recall[:-1] + 1e-9)
best_thresh = thresholds[np.argmax(f1_scores)]

plt.figure(figsize=(8,6))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.plot(thresholds, f1_scores, label="F1")
plt.axvline(best_thresh, color="red", linestyle="--", label=f"Best: {best_thresh:.2f}")
plt.xlabel("Threshold")
plt.title("Threshold Tuning - XGBoost")
plt.legend()
plt.savefig("../plots/threshold_tuning.pdf", bbox_inches="tight")
plt.show()

#learning curve RF + XGboost

def plot_learning_curve(estimator, title, x, y):
    train_sizes, train_scores, val_scores=learning_curve(estimator, x, y, cv=5, scoring="f1",
    train_sizes=np.linspace(0.1, 1.0, 8),n_jobs=-1)

    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train F1")
    plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation F1")
    plt.fill_between(train_sizes,
                     train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
    plt.fill_between(train_sizes,
                     val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
    plt.title(f"Learning curve - {title}")
    plt.xlabel("Training samples")
    plt.ylabel("F1 score")
    plt.legend()
    plt.savefig(f"../plots/learning_curve_{title}pdf", bbox_inches="tight")
    plt.show()

plot_learning_curve(best_rf, "Random Forest", x, y)
plot_learning_curve(best_xgb, "XGBoost", x, y)

#AUC + MCC
print("\n=== AUC ===")
for name, clf in [("LR", best_lr), ("RF", best_rf), ("KNN", best_knn), ("XGB", best_xgb)]:
    proba = clf.predic_proba(x_test)[:, 1]
    print(f"{name} AUC: {roc_auc_score(y_test, proba):.3f}")

print("\n=== Matthews Correlation coefficient ===")
for name, pred in [("LR", pred_log), ("RF", pred_rf), ("KNN", pred_knn), ("SVM", pred_svm), ("XGB", pred_xgb)]:
    print(f"{name} MCC: {matthews_corrcoef(y_test, pred):.3f}")


rezultate = {
    "Logistic Regression": accuracy_score(y_test, pred_log),
    "Random Forest": accuracy_score(y_test, pred_rf),
    "KNN": accuracy_score(y_test, pred_knn),
    "SVM": accuracy_score(y_test, pred_svm),
    "XGBoost": accuracy_score(y_test, pred_xgb)
}

plt.figure(figsize=(8,6))
plt.barh(list(rezultate.keys()), list(rezultate.values()), color="steelblue")
plt.xlim(0.5, 1.0)
plt.xlabel("Precizie")
plt.title("Comparatie modele - Precizie")
plt.tight_layout()
plt.savefig("../plots/Comparatie_modele_precizie.pdf", dpi=300, bbox_inches="tight")
plt.show()


print("\nComparatie lr vs rf vs knn vs svm vs XGBoost:")
print(f"Logistic regression:{accuracy_score(y_test, pred_log):.3f}")
print(f"Random Forest:{accuracy_score(y_test, pred_rf):.3f}")
print(f"KNN:{accuracy_score(y_test, pred_knn):.3f}")
print(f"SVM:{accuracy_score(y_test, pred_svm):.3f}")
print(f"XGBoost:{accuracy_score(y_test, pred_xgb):.3f}")