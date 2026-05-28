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
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
os.makedirs("../figuri_clasificare", exist_ok=True)

#Incarcare set de date
df = pd.read_csv("../data/spotify_dataset.csv")
df = df.dropna().copy()
df = df.drop_duplicates(subset=["track_id"])
print(f"Dimensiune dupa eliminarea duplicatelor: {df.shape}")

caracteristici = ["danceability",
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
x = df[caracteristici]

print("Set incarcat")
print(df.shape)
print(df.head())

#variabila target
threshold = df["popularity"].quantile(0.75)
df["hit"] = (df["popularity"] >= threshold).astype(int)
y = df["hit"]

sns.countplot(x=df["hit"])
plt.title("Distributie hituri vs non hituri")
plt.savefig("../figuri_clasificare/distributie_hit_non_hit.png", dpi=300, bbox_inches="tight")
plt.show()

print("Prag popularitate(top 25%):", threshold)
print(df["hit"].value_counts())

#harta corelatie + disttributia popularitatii
fig, axes = plt.subplots(1,2, figsize = (16,6))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=axes[0])
axes[0].set_title("Harta de corelatie")
sns.histplot(df["popularity"], bins=50, ax=axes[1], log_scale=True)
axes[1].set_title("Distributia popularitatii")
plt.tight_layout()
plt.savefig("../figuri_clasificare/corelatie_distributie_popularitate.png", dpi=300, bbox_inches="tight")
plt.show()

#train/test 80/20 split fix rs=42+ scalare
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
raport = neg / pos
print(f"Raport clase: {raport:.3f}")


#Regresie logistica
rl_normal = LogisticRegression(max_iter=1000, class_weight="balanced")
rl_normal.fit(x_train, y_train)
pred_rl_normal = rl_normal.predict(x_test)
print("\n== Regresie Logistica(varianta normala) ===")
print(classification_report(y_test, pred_rl_normal))

rl_parametri = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs", "liblinear"]
}
rl_cautare = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    rl_parametri,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)
rl_cautare.fit(x_train, y_train)
print("Parametri optimi RL:", rl_cautare.best_params_)
best_rl = rl_cautare.best_estimator_
pred_rl_optimizat = best_rl.predict(x_test)

print("\n === Regresie Logistica(varianta optimizata) ===")
print(classification_report(y_test, pred_rl_optimizat))

#matrice confuzie regresie logistica
cm_rl = confusion_matrix(y_test, pred_rl_optimizat)
disp_rl = ConfusionMatrixDisplay(confusion_matrix=cm_rl)
disp_rl.plot()
plt.title("Matrice de confuzie - Regresie Logistica")
plt.savefig("../figuri_clasificare/matrice_confuzie_regresie_logistica.png",dpi=300, bbox_inches="tight")
plt.show()

#coeficienti regresie logistica
coef = best_rl.coef_[0]
plt.figure(figsize=(8,6))
plt.barh(caracteristici, coef)
plt.title("Coeficienti Regresie Logistica")
plt.xlabel("Impact probabilitate hit")
plt.savefig("../figuri_clasificare/coeficienti_regresie_logistica.png", dpi=300, bbox_inches="tight")
plt.show()


#Random Forest
rf_normal = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
rf_normal.fit(x_train, y_train)
pred_rf_normal = rf_normal.predict(x_test)
print("\n===Random Forest(varianta normala) ===")
print(classification_report(y_test, pred_rf_normal))

rf_parametri = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
rf_cautare = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
    rf_parametri,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)
rf_cautare.fit(x_train, y_train)
print("Parametri optimi RF:", rf_cautare.best_params_)
print("Cel mai bun scor F1:", rf_cautare.best_score_)
best_rf = rf_cautare.best_estimator_
pred_rf_optimizat = best_rf.predict(x_test)

print("\n === Random Forest(varianta optimizata) ===")
print(classification_report(y_test, pred_rf_optimizat))

#matrice confuzie random forest
cm_rf = confusion_matrix(y_test, pred_rf_optimizat)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp_rf.plot()
plt.title("Matrice de confuzie - Random Forest")
plt.savefig("../figuri_clasificare/matrice_confuzie_random_forest.png", dpi=300, bbox_inches="tight")
plt.show()

#importanta caracteristicilor - random forest
importanta = best_rf.feature_importances_
indici = np.argsort(importanta)
plt.figure(figsize=(8,6))
plt.barh(range(len(indici)), importanta[indici], align='center')
plt.yticks(range(len(indici)), [caracteristici[i] for i in indici])
plt.title("Importanta caracteristicilor - Random Forest")
plt.savefig("../figuri_clasificare/importanta_caracteristici.png", dpi=300, bbox_inches="tight")
plt.show()


#KNN
knn_normal = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_normal.fit(x_train, y_train)
pred_knn_normal = knn_normal.predict(x_test)
print("\n=== KNN(varianta normala ===")
print(classification_report(y_test, pred_knn_normal))

knn_parametri = {
    "n_neighbors": [3, 5, 7, 11, 15, 21],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean","manhattan"]
}
knn_cautare = GridSearchCV(
    KNeighborsClassifier(n_jobs=-1),
    knn_parametri,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)
knn_cautare.fit(x_train, y_train)
print("Parametri optimi KNN:", knn_cautare.best_params_)
best_knn = knn_cautare.best_estimator_
pred_knn_optimizat = best_knn.predict(x_test)

print("\n === KNN(varianta optimizata) ===")
print(classification_report(y_test, pred_knn_optimizat))

#matrice confuzie knn
cm_knn = confusion_matrix(y_test, pred_knn_optimizat)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot()
plt.title("Matrice de confuzie - KNN")
plt.savefig("../figuri_clasificare/matrice_confuzie_KNN.png", dpi=300, bbox_inches="tight")
plt.show()


#SVM
svm_normal = LinearSVC(class_weight="balanced", max_iter=2000)
svm_normal.fit(x_train, y_train)
pred_svm_normal = svm_normal.predict(x_test)
print("\n=== SVM(varianta normala) ===")
print(classification_report(y_test, pred_svm_normal))

svm_parametri = {
    "C": [0.01, 0.1, 1, 10],
    "max_iter": [2000]
}
svm_cautare = GridSearchCV(
    LinearSVC(class_weight="balanced"),
    svm_parametri,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)
svm_cautare.fit(x_train, y_train)
print("Parametri optimi SVM:", svm_cautare.best_params_)
best_svm = svm_cautare.best_estimator_
pred_svm_optimizat = best_svm.predict(x_test)

print("\n === SVM(varianta optimizata) ===")
print(classification_report(y_test, pred_svm_optimizat))

#matrice confuzie svm
cm_svm = confusion_matrix(y_test, pred_svm_optimizat)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
disp_svm.plot()
plt.title("Matrice de confuzie - SVM")
plt.savefig("../figuri_clasificare/matrice_confuzie_SVM.png", dpi=300, bbox_inches="tight")
plt.show()


#XGBoost
xgb_normal = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                         scale_pos_weight=raport, eval_metric="logloss",random_state=42, n_jobs=-1)
xgb_normal.fit(x_train, y_train)
pred_xgb_normal = xgb_normal.predict(x_test)
print("\n=== XGB(varianta normala) ===")
print(classification_report(y_test, pred_xgb_normal))

xgb_parametri = {
    "n_estimators": randint(100, 400),
    "max_depth": randint(3, 10),
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]

}
xgb_rand = RandomizedSearchCV(
    XGBClassifier(scale_pos_weight=raport, eval_metric="logloss", random_state=42, n_jobs=-1),
    xgb_parametri,
    n_iter=30,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
    random_state=42
)
xgb_rand.fit(x_train, y_train)
print("Parametri optimi XGB:", xgb_rand.best_params_)
best_xgb = xgb_rand.best_estimator_
pred_xgb_optimizat =best_xgb.predict(x_test)

print("\n === XGBoost(varianta optimizata) ===")
print(classification_report(y_test, pred_xgb_optimizat))

#matrice confuzie XGB
cm_xgb = confusion_matrix(y_test, pred_xgb_optimizat)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb)
disp_xgb.plot()
plt.title("Matrice de confuzie - XGBoost")
plt.savefig("../figuri_clasificare/matrice_confuzie_XGBoost.png", dpi=300, bbox_inches="tight")
plt.show()

modele = ["RL", "RF", "KNN", "SVM", "XGB"]

metrici_normal = {nume: {"acc": [], "f1": [], "prec": [], "rec": []}
                  for nume in modele}
metrici_optimizat = {nume: {"acc": [], "f1": [], "prec": [], "rec": [], "mcc": [], "auc": [] if nume != "SVM" else None}
                     for nume in modele}

pipe_normal = {
    "RL": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "RF": Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1))]),
    "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))]),
    "SVM": Pipeline([("scaler", StandardScaler()), ("model", LinearSVC(class_weight="balanced", max_iter=2000))]),
    "XGB": Pipeline([("scaler", StandardScaler()), ("model", XGBClassifier(scale_pos_weight=raport, eval_metric="logloss", n_jobs=-1))])
}
pipe_optimizat = {
    "RL": Pipeline([("scaler", StandardScaler()), ("model", best_rl)]),
    "RF": Pipeline([("scaler", StandardScaler()), ("model", best_rf)]),
    "KNN": Pipeline([("scaler", StandardScaler()), ("model", best_knn)]),
    "SVM": Pipeline([("scaler", StandardScaler()), ("model", best_svm)]),
    "XGB": Pipeline([("scaler", StandardScaler()), ("model", best_xgb)])
}

print("\n --- 10 rulari ---")
for rs in range(10):
    x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(x, y, test_size=0.2, random_state=rs)

    for nume in modele:
        #varianta normala
        pipe_normal[nume].fit(x_train_r, y_train_r)
        pred = pipe_normal[nume].predict(x_test_r)
        metrici_normal[nume]["acc"].append(accuracy_score(y_test_r, pred))
        metrici_normal[nume]["f1"].append(f1_score(y_test_r, pred))
        metrici_normal[nume]["prec"].append(precision_score(y_test_r, pred))
        metrici_normal[nume]["rec"].append(recall_score(y_test_r, pred))

        #varianta optimizata
        pipe_optimizat[nume].fit(x_train_r, y_train_r)
        pred = pipe_optimizat[nume].predict(x_test_r)
        metrici_optimizat[nume]["acc"].append(accuracy_score(y_test_r, pred))
        metrici_optimizat[nume]["f1"].append(f1_score(y_test_r, pred))
        metrici_optimizat[nume]["prec"].append(precision_score(y_test_r, pred))
        metrici_optimizat[nume]["rec"].append(recall_score(y_test_r, pred))
        metrici_optimizat[nume]["mcc"].append(matthews_corrcoef(y_test_r, pred))
        if nume != "SVM":
            prob = pipe_optimizat[nume].predict_proba(x_test_r)[:, 1]
            metrici_optimizat[nume]["auc"].append(roc_auc_score(y_test_r, prob))

    print(f"rulare {rs+1}/{10}")

#Tabel metrici modele optimzate
print(f"\n{'Model':<6} {'Accuracy':>18} {'Precision':>18} {'Recall':>18}"
      f"{'F1':>18} {'AUC':>18} {'MCC':>18}")
print("-"*110)

for nume in modele:
    m = metrici_optimizat[nume]
    auc = (f"{np.mean(m['auc']):>8.3f}+- {np.std(m['auc']):.3f}" if m["auc"] else "---")
    print(f"{nume:<6}"
          f" {np.mean(m['acc']):>8.3f}+- {np.std(m['acc']):.3f}"
          f" {np.mean(m['prec']):>8.3f}+- {np.std(m['prec']):.3f}"
          f" {np.mean(m['rec']):>8.3f}+- {np.std(m['rec']):.3f}"
          f" {np.mean(m['f1']):>8.3f}+- {np.std(m['f1']):.3f}"
          f" {auc}"
          f" {np.mean(m['mcc']):.3f}+- {np.std(m['mcc']):.3f}")

#tabel comparatie varianta normala vs optimizata
print(f"\n{'Model':<6} {'Acc normal':>18} {'Acc optim':>18}"
      f"{'F1 normal':>18} {'F1 optim':>18}")
print("-"*82)
for nume in modele:
    normal = metrici_normal[nume]
    optim = metrici_optimizat[nume]
    print(f"{nume:<6}"
          f" {np.mean(normal['acc']):>8.3f}+- {np.std(normal['acc']):.3f}"
          f" {np.mean(optim['acc']):>8.3f}+- {np.std(optim['acc']):.3f}"
          f" {np.mean(normal['f1']):>8.3f}+- {np.std(normal['f1']):.3f}"
          f" {np.mean(optim['f1']):>8.3f}+- {np.std(optim['f1']):.3f}")

#Validare incrucisata
print("\n===Validare incrucisata(5fold)===")
for nume, pipe in pipe_optimizat.items():
    rezultate = cross_validate(pipe, x, y, cv=5, scoring=["f1", "accuracy"], n_jobs=-1)
    print(f"{nume}: Accuracy={rezultate['test_accuracy'].mean():.3f} F1={rezultate['test_f1'].mean():.3f}")

#Curba ROC
fig, ax = plt.subplots(figsize=(8,6))
for name, clf, pred in [
    ("Regresie Logistica", best_rl, pred_rl_optimizat),
    ("Random Forest", best_rf, pred_rf_optimizat),
    ("KNN", best_knn, pred_knn_optimizat),
    ("XGB", best_xgb, pred_xgb_optimizat),
]:
    RocCurveDisplay.from_estimator(clf, x_test, y_test, ax=ax, name=name)
plt.title("Curba ROC - Comparatie modele")
plt.savefig("../figuri_clasificare/curba_ROC.png", dpi=300, bbox_inches="tight")
plt.show()

#Optimiare prag decizie XGB
probabilitati_xgb = best_xgb.predict_proba(x_test)[:, 1]
precizie, rechemare, prag_decizie = precision_recall_curve(y_test, probabilitati_xgb)
scor_f1 = 2 * (precizie[:-1]*rechemare[:-1]) / (precizie[:-1]+rechemare[:-1] + 1e-9)
prag_decizie_optim = prag_decizie[np.argmax(scor_f1)]
print(f"\nPrag de decizie optim XGBoost: {prag_decizie_optim:.3f}")

plt.figure(figsize=(8,6))
plt.plot(prag_decizie, precizie[:-1], label="Precizie")
plt.plot(prag_decizie, rechemare[:-1], label="Rechemare")
plt.plot(prag_decizie, scor_f1, label="F1")
plt.axvline(prag_decizie_optim, color="red", linestyle="--", label=f"Prag de decizie optim: {prag_decizie_optim:.3f}")
plt.xlabel("Prag de decizie")
plt.title("Optimizare prag de decizie - XGBoost")
plt.legend()
plt.savefig("../figuri_clasificare/optimizare_prag_decizie.png", dpi=300, bbox_inches="tight")
plt.show()

#curbe invatare RF, XGBoost
def plot_curba_invatare(estimator, title, x, y):
    train_sizes, train_scores, val_scores=learning_curve(estimator, x, y, cv=5, scoring="f1",
    train_sizes=np.linspace(0.1, 1.0, 8),n_jobs=-1)
    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label="F1 antrenare")
    plt.plot(train_sizes, val_scores.mean(axis=1), label="F1 validare")
    plt.fill_between(train_sizes,
                     train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
    plt.fill_between(train_sizes,
                     val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
    plt.title(f"Curba invatare - {title}")
    plt.xlabel("Nr exemple antrenare")
    plt.ylabel("Scor F1")
    plt.legend()
    plt.savefig(f"../figuri_clasificare/curba_invatare_{title}.png",dpi=300, bbox_inches="tight")
    plt.show()

plot_curba_invatare(best_rf, "Random Forest", x, y)
plot_curba_invatare(best_xgb, "XGBoost", x, y)

#grafic comparatie acuratete medie
label = ["Reg.Logistica", "Random Forest", "KNN", "SVM", "XGBoost"]
acuratete_medii = [np.mean(metrici_optimizat[nume]["acc"]) for nume in modele]
acuratete_deviatii = [np.std(metrici_optimizat[nume]["acc"]) for nume in modele]

plt.figure(figsize=(8,6))
plt.barh(modele, acuratete_medii, xerr=acuratete_deviatii,
         color="steelblue", capsize=5)
plt.xlim(0.3, 1.0)
plt.xlabel("Acuratete medie(10 rulari)")
plt.title("Comparatie modele - Acuratete medie +- deviatie standard")
plt.tight_layout()
plt.savefig("../figuri_clasificare/comparatie_modele_acuratete.png", dpi=300, bbox_inches="tight")
plt.show()

#grafic comparatie F1 normal vs optimizat
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
f1_normal_medii = [np.mean(metrici_normal[nume]["f1"]) for nume in modele]
f1_normal_deviatii = [np.std(metrici_normal[nume]["f1"]) for nume in modele]
f1_optim_medii = [np.mean(metrici_optimizat[nume]["f1"]) for nume in modele]
f1_optim_deviatii = [np.std(metrici_optimizat[nume]["f1"]) for nume in modele]

axes[0].barh(modele, f1_normal_medii, xerr=f1_normal_deviatii, color="steelblue", capsize=5)
axes[0].set_xlim(0.0, 1.0)
axes[0].set_xlabel("Scor F1 mediu")
axes[0].set_title("F1 mediu - varianta normala +- deviatie")

axes[1].barh(modele, f1_optim_medii, xerr=f1_optim_deviatii, color="seagreen", capsize=5)
axes[1].set_xlim(0.0, 1.0)
axes[1].set_xlabel("Scor F1 mediu")
axes[1].set_title("F1 mediu - varianta optimizata +- deviatie")

plt.tight_layout()
plt.savefig("../figuri_clasificare/comparatie_f1.png", dpi=300, bbox_inches="tight")
plt.show()

#boxplot distributie f1
fig, axes = plt.subplots(1,2, figsize=(16, 6))
axes[0].boxplot([metrici_normal[nume]["f1"] for nume in modele], tick_labels=modele, vert=False)
axes[0].set_xlabel("Scor F1")
axes[0].set_title("Distributia F1 - varianta normala")

axes[1].boxplot([metrici_optimizat[nume]["f1"] for nume in modele], tick_labels=modele, vert=False)
axes[1].set_xlabel("Scor F1")
axes[1].set_title("Distributia F1 - varianta optimizata")

plt.tight_layout()
plt.savefig("../figuri_clasificare/boxplot_f1.png", dpi=300, bbox_inches="tight")
plt.show()
