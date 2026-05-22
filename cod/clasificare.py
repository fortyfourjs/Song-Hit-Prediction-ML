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
os.makedirs("../figuri_clasificare", exist_ok=True)
N_RULARI = 10

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
plt.savefig("../figuri_clasificare/distributie_hit_non_hit.pdf", dpi=300, bbox_inches="tight")
plt.show()

print("Prag popularitate(top 25%):", threshold)
print(df["hit"].value_counts())

#train/test 80/20 + scalare
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
raport = neg / pos
print(f"Raport clase: {raport:.2f}")


#Regresie logistica
rl_normal = LogisticRegression(max_iter=1000, class_weight="balanced")
rl_normal.fit(x_train, y_train)
pred_rl_normal = rl_normal.predict(x_test)
print("\n== Regresie Logistica(varianta normala) ===")
print(classification_report(y_test, pred_rl_normal))

rl_parametri = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs", "liblinear"],
    "penalty": ["l2"]
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
print("Paremtri optimi RL:", rl_cautare.best_params_)

best_rl = rl_cautare.best_estimator_
pred_rl_optimizat = best_rl.predict(x_test)

print("\n === Regresie Logistica(varianta optimizata) ===")
print(classification_report(y_test, pred_rl_optimizat))


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

cm = confusion_matrix(y_test, pred_rf_optimizat)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Matrice de confuzie - Random Forest")
plt.savefig("../figuri_clasificare/matrice_confuzie_random_forest.pdf", dpi=300, bbox_inches="tight")
plt.show()
print("\n === Random Forest(varianta optimizata) ===")
print(classification_report(y_test, pred_rf_optimizat))

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

cm_knn = confusion_matrix(y_test, pred_knn_optimizat)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot()
plt.title("Matrice de confuzie - KNN")
plt.savefig("../figuri_clasificare/matrice_confuzie_KNN.pdf", dpi=300, bbox_inches="tight")
plt.show()
print("\n === KNN(varianta optimizata) ===")
print(classification_report(y_test, pred_knn_optimizat))


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

cm_svm = confusion_matrix(y_test, pred_svm_optimizat)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
disp_svm.plot()
plt.title("Matrice de confuzie - SVM")
plt.savefig("../figuri_clasificare/matrice_confuzie_SVM.pdf", dpi=300, bbox_inches="tight")
plt.show()
print("\n === SVM(varianta optimizata) ===")
print(classification_report(y_test, pred_svm_optimizat))


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

cm_xgb = confusion_matrix(y_test, pred_xgb_optimizat)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb)
disp_xgb.plot()
plt.title("Matrice de confuzie - XGBoost")
plt.savefig("../figuri_clasificare/matrice_confuzie_XGBoost.pdf", dpi=300, bbox_inches="tight")
plt.show()

print("\n === XGBoost(varianta optimizata) ===")
print(classification_report(y_test, pred_xgb_optimizat))

pipe_rl = Pipeline([
    ("scaler", StandardScaler()),
    ("model", best_rl)
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

scor_acuratete_normal = {"RL": [], "RF": [], "KNN": [], "SVM": [], "XGB": []}
scor_f1_normal = {"RL": [], "RF": [], "KNN": [], "SVM": [], "XGB": []}
scor_acuratete_optimizat = {"RL": [], "RF": [], "KNN": [], "SVM": [], "XGB": []}
scor_f1_optimizat = {"RL": [], "RF": [], "KNN": [], "SVM": [], "XGB": []}
scor_auc_optimizat = {"RL": [], "RF": [], "KNN": [], "XGB": []}
scor_mcc_optimizat = {"RL": [], "RF": [], "KNN": [], "SVM": [], "XGB": []}

pipe_rl_normal = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
pipe_rf_normal = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100, class_weight="balanced"))
])
pipe_knn_normal = Pipeline([
    ("scaler", StandardScaler()),
     ("model", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
])
pipe_svm_normal = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearSVC(class_weight="balanced", max_iter=2000))
])
pipe_xgb_normal = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBClassifier(scale_pos_weight=raport, eval_metric="logloss", n_jobs=-1))
])
print("\n --- 10 rulari ---")
for rs in range(N_RULARI):
    x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(x, y, test_size=0.2, random_state=rs)

    for nume, pipe in [("RL", pipe_rl_normal), ("RF", pipe_rf_normal), ("KNN", pipe_knn_normal), ("SVM", pipe_svm_normal), ("XGB", pipe_xgb_normal)]:
        pipe.fit(x_train_r, y_train_r)
        pred = pipe.predict(x_test_r)
        scor_acuratete_normal[nume].append(accuracy_score(y_test_r, pred))
        scor_f1_normal[nume].append(f1_score(y_test_r, pred))
    for nume, pipe in [("RL", pipe_rl), ("RF", pipe_rf), ("KNN", pipe_knn), ("SVM", pipe_svm), ("XGB", pipe_xgb)]:
        pipe.fit(x_train_r, y_train_r)
        pred = pipe.predict(x_test_r)
        scor_acuratete_optimizat[nume].append(accuracy_score(y_test_r, pred))
        scor_f1_optimizat[nume].append(f1_score(y_test_r, pred))

    for nume, pipe in [("RL", pipe_rl), ("RF", pipe_rf), ("KNN", pipe_knn), ("XGB", pipe_xgb)]:
        pipe.fit(x_train_r, y_train_r)
        prob = pipe.predict_proba(x_test_r)[:, 1]
        scor_auc_optimizat[nume].append(roc_auc_score(y_test_r, prob))

    for nume, pipe in [("RL", pipe_rl), ("RF", pipe_rf), ("KNN", pipe_knn), ("SVM", pipe_svm), ("XGB", pipe_xgb)]:
        pred = pipe.predict(x_test_r)
        scor_mcc_optimizat[nume].append(matthews_corrcoef(y_test_r, pred))

    print(f"rulari {rs+1}/{N_RULARI}")

print("\n --- AUC mediu(10 rulari)---")
for nume in ["RL", "RF", "KNN", "XGB"]:
    print(f"{nume}: {np.mean(scor_auc_optimizat[nume]):.3f}"
          f"+-{np.std(scor_auc_optimizat[nume]):.3f}")
print("\n --- MCC mediu(10 rulari) ---")
for nume in ["RL", "RF", "KNN", "SVM", "XGB"]:
    print(f"{nume}: {np.mean(scor_mcc_optimizat[nume]):.3f}"
          f"+-{np.std(scor_mcc_optimizat[nume]):.3f}")

#Validare incrucisata pe x(date initiale)
print("\n===Validare incrucisata(5fold)===")
for nume, pipe in [("RL", pipe_rl), ("RF",pipe_rf), ("KNN", pipe_knn), ("SVM", pipe_svm), ("XGB", pipe_xgb)]:
    rezultate = cross_validate(pipe, x, y, cv=5, scoring=["f1", "accuracy"], n_jobs=-1)
    f1 = rezultate["test_f1"].mean()
    acuratete = rezultate["test_accuracy"].mean()
    print(f"{nume}: Acuratete={acuratete:.3f} F1={f1:.3f}")

#importanta caracteristicilor random forest
importanta = best_rf.feature_importances_
indici = np.argsort(importanta)

plt.figure(figsize=(8,6))
plt.barh(range(len(indici)), importanta[indici], align='center')
plt.yticks(range(len(indici)), [caracteristici[i] for i in indici])
plt.title("Importanta caracteristicilor - Random Forest")
plt.savefig("../figuri_clasificare/importanta_caracteristici.pdf", dpi=300, bbox_inches="tight")
plt.show()

#figuri
fig, axes = plt.subplots(1,2, figsize = (16,6))

sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=axes[0])
axes[0].set_title("Harta de corelatie")

sns.histplot(df["popularity"], bins=50, ax=axes[1], log_scale=True)
axes[1].set_title("Distributia popularitatii")

plt.tight_layout()
plt.savefig("../figuri_clasificare/corelatie_distributie_popularitate.pdf", dpi=300, bbox_inches="tight")
plt.show()

#coeficienti regresie logistica
coef = best_rl.coef_[0]
plt.figure(figsize=(8,6))
plt.barh(caracteristici, coef)
plt.title("Coeficienti Regresie Logistica")
plt.xlabel("Impact probabilitate hit")
plt.savefig("../figuri_clasificare/coeficienti_regresie_logistica.pdf", dpi=300, bbox_inches="tight")
plt.show()

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
plt.savefig("../figuri_clasificare/curba_ROC.pdf", dpi=300, bbox_inches="tight")
plt.show()

#Optimiare prag decizie XGB
probabilitati_xgb = best_xgb.predict_proba(x_test)[:, 1]
precizie, rechemare, prag_decizie = precision_recall_curve(y_test, probabilitati_xgb)
scor_f1 = 2 * (precizie[:-1]*rechemare[:-1]) / (precizie[:-1]+rechemare[:-1] + 1e-9)
prag_decizie_optim = prag_decizie[np.argmax(scor_f1)]

plt.figure(figsize=(8,6))
plt.plot(prag_decizie, precizie[:-1], label="Precizie")
plt.plot(prag_decizie, rechemare[:-1], label="RFechemare")
plt.plot(prag_decizie, scor_f1, label="F1")
plt.axvline(prag_decizie_optim, color="red", linestyle="--", label=f"Prag de decizie optim: {prag_decizie_optim:.2f}")
plt.xlabel("Prag de decizie")
plt.title("Optimizare prag de decizie - XGBoost")
plt.legend()
plt.savefig("../figuri_clasificare/optimizare_prag_decizie.pdf", bbox_inches="tight")
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
    plt.savefig(f"../figuri_clasificare/curba_invatare_{title}.pdf", bbox_inches="tight")
    plt.show()

plot_curba_invatare(best_rf, "Random Forest", x, y)
plot_curba_invatare(best_xgb, "XGBoost", x, y)

#AUC + MCC
print("\n=== Scor AUC(1 rulare)===")
for nume, clf in [("RL", best_rl), ("RF", best_rf), ("KNN", best_knn), ("XGB", best_xgb)]:
    probabilitati = clf.predict_proba(x_test)[:, 1]
    print(f"{nume} AUC: {roc_auc_score(y_test, probabilitati):.3f}")

print("\n=== Coeficient de corelatie Matthews(MCC)[1 rulare] ===")
for nume, pred in [("RL", pred_rl_optimizat), ("RF", pred_rf_optimizat), ("KNN", pred_knn_optimizat), ("SVM", pred_svm_optimizat), ("XGB", pred_xgb_optimizat)]:
    print(f"{nume} MCC: {matthews_corrcoef(y_test, pred):.3f}")

#comparatie acuratete medii
rezultate = {
    "Regresie Logistica": np.mean(scor_acuratete_optimizat["RL"]),
    "Random Forest": np.mean(scor_acuratete_optimizat["RF"]),
    "KNN": np.mean(scor_acuratete_optimizat["KNN"]),
    "SVM": np.mean(scor_acuratete_optimizat["SVM"]),
    "XGBoost": np.mean(scor_acuratete_optimizat["XGB"])
}
acuratete_medii = list(rezultate.values())
acuratete_deviatii = [np.std(scor_acuratete_optimizat[nume]) for nume in ["RL", "RF", "KNN", "SVM", "XGB"]]
plt.barh(list(rezultate.keys()), acuratete_medii, xerr=acuratete_medii,
         color="steelblue", capsize=5)

plt.figure(figsize=(8,6))
plt.barh(list(rezultate.keys()), list(rezultate.values()), color="steelblue")
plt.xlim(0.5, 1.0)
plt.xlabel("Acuratete")
plt.title("Comparatie modele - Acuratete")
plt.tight_layout()
plt.savefig("../figuri_clasificare/comparatie_modele_acuratete.pdf", dpi=300, bbox_inches="tight")
plt.show()


print("\nComparatie acuratete(1 rulare) RL vs RF vs KNN vs SVM vs XGBoost:")
print(f"Regresie Logistica:{accuracy_score(y_test, pred_rl_optimizat):.3f}")
print(f"Random Forest:{accuracy_score(y_test, pred_rf_optimizat):.3f}")
print(f"KNN:{accuracy_score(y_test, pred_knn_optimizat):.3f}")
print(f"SVM:{accuracy_score(y_test, pred_svm_optimizat):.3f}")
print(f"XGBoost:{accuracy_score(y_test, pred_xgb_optimizat):.3f}")

#tabel comparatie base vs tuned
print(f"\n{'Model':<20} {'Acuratete normala':>10} {'Acuratete optimizata':>10} {'F1 normal':>10} {'F1 optimizat':>10}")
print("-" * 60)

model = ["RL", "RF", "KNN", "SVM", "XGB"]
pred_normale = [pred_rl_normal, pred_rf_normal, pred_knn_normal, pred_svm_normal, pred_xgb_normal]
pred_optimizate = [pred_rl_optimizat, pred_rf_optimizat, pred_knn_optimizat, pred_svm_optimizat, pred_xgb_optimizat]

for nume, normal, optimizat in zip(model, pred_normale, pred_optimizate):
    print(f"{nume:<20}"
          f"{accuracy_score(y_test, normal):>10.3f}"
          f"{accuracy_score(y_test, optimizat):>10.3f}"
          f"{f1_score(y_test, normal):>10.3f}"
          f"{f1_score(y_test, optimizat):>10.3f}")
print(f"\n{'Model':<8}{'Acuratete normala':>10}{'Acuratete optimizata':>9}{'Scor F1 normal':>10}{'Scor F1 optimizat':>9}")
print("-" * 60)

for nume in model:
    acuratete_normala_medie = np.mean(scor_acuratete_normal[nume])
    acuratete_normala_deviatie = np.std(scor_acuratete_normal[nume])
    acuratete_optimizata_medie = np.mean(scor_acuratete_optimizat[nume])
    acuratete_optimizata_deviatie = np.std(scor_acuratete_optimizat[nume])
    f1_normal_medie = np.mean(scor_f1_normal[nume])
    f1_normal_deviatie = np.std(scor_f1_normal[nume])
    f1_optimizat_medie = np.mean(scor_f1_optimizat[nume])
    f1_optimizat_deviatie = np.std(scor_f1_optimizat[nume])

    print(f"{nume:<8}"
          f"{acuratete_normala_medie:>10.3f} +- {acuratete_normala_deviatie:>10.3f}"
          f"{acuratete_optimizata_medie:>10.3f} +- {acuratete_optimizata_deviatie:>10.3f}"
          f"{f1_normal_medie:>10.3f} +- {f1_normal_deviatie:>10.3f}"
          f"{f1_optimizat_medie:>10.3f} +- {f1_optimizat_deviatie:>10.3f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

f1_normal_medii = [np.mean(scor_f1_normal[nume]) for nume in model]
f1_normal_deviatii = [np.std(scor_f1_normal[nume]) for nume in model]
f1_optimizat_medii = [np.mean(scor_f1_optimizat[nume]) for nume in model]
f1_optimizat_deviatii = [np.std(scor_f1_optimizat[nume]) for nume in model]

axes[0].barh(model, f1_normal_medii, xerr=f1_normal_deviatii, color="steelblue", capsize=5)
axes[0].set_xlim(0.3, 1.0)
axes[0].set_xlabel("Scor F1 mediu")
axes[0].set_title("Scor F1 mediu(varianta normala) +- deviatie")

axes[1].barh(model, f1_optimizat_medii, xerr=f1_optimizat_deviatii, color="steelblue", capsize=5)
axes[1].set_xlim(0.3, 1.0)
axes[1].set_xlabel("Scor F1 mediu")
axes[1].set_title("Scor F1 mediu(varianta optimizata) +- deviatie")

plt.tight_layout()
plt.savefig("../figuri_clasificare/comparatie_f1.pdf", dpi=300, bbox_inches="tight")
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].boxplot([scor_f1_normal[nume] for nume in model],
                 tick_labels=model, vert=False)
axes[0].set_xlabel("Scor F1")
axes[0].set_title("Distributia F1(varianta normala)")

axes[1].boxplot([scor_f1_optimizat[nume] for nume in model],
                tick_labels=model, vert=False)
axes[1].set_xlabel("Scor F1")
axes[1].set_title("Distributia F1(varianta optimizata)")
plt.tight_layout()
plt.savefig("../figuri_clasificare/boxplot_f1.pdf", dpi=300, bbox_inches="tight")
plt.show()
