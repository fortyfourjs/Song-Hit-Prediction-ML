import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline


os.makedirs("../figuri_regresie", exist_ok=True)


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
y = df["popularity"]

print("Set incarcat")
print(df.shape)
print(df.head())

#split fix rs=42 pentru hiperparaemtri
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

modele = ["RL", "Ridge", "RF", "KNN", "XGB"]

pipe_normal = {
    "RL": Pipeline([("scaler", StandardScaler()),("model", LinearRegression())]),
    "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
    "RF": Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))]),
    "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=5, n_jobs=-1))]),
    "XGB": Pipeline([("scaler", StandardScaler()), ("model", XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, eval_metric="rmse", n_jobs=-1))])
}

#Ridge
ridge_parametri = {
    "alpha": [0.01, 0.1, 1.0, 10, 100]
}
ridge_cautare = GridSearchCV(Ridge(),
                             ridge_parametri,
                             cv=5,
                             scoring="neg_root_mean_squared_error",
                             n_jobs=-1,
                             verbose=1)

ridge_cautare.fit(x_train, y_train)
print("Parametri optimi Ridge:", ridge_cautare.best_params_)
best_ridge = ridge_cautare.best_estimator_

#Random Forest
rf_parametri = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
rf_cautare = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                          rf_parametri,
                          cv=5,
                          scoring="neg_root_mean_squared_error",
                          n_jobs=-1,
                          verbose=1)
rf_cautare.fit(x_train, y_train)
print("Parametri optimi RF:", rf_cautare.best_params_)
best_rf = rf_cautare.best_estimator_

#KNN
knn_parametri = {
    "n_neighbors": [3, 5, 7, 11, 15, 21],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}
knn_cautare = GridSearchCV(KNeighborsRegressor(n_jobs=-1),
                           knn_parametri,
                           cv=5,
                           scoring="neg_root_mean_squared_error",
                           n_jobs=-1,
                           verbose=1)
knn_cautare.fit(x_train, y_train)
print("Parametri optimi KNN:", knn_cautare.best_params_)
best_knn = knn_cautare.best_estimator_

#XGBoost
xgb_parametri = {
    "n_estimators": randint(100, 400),
    "max_depth": randint(3, 10),
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}
xgb_rand = RandomizedSearchCV(XGBRegressor(eval_metric="rmse", random_state=42, n_jobs=-1),
                              xgb_parametri,
                              n_iter=30,
                              cv=5,
                              scoring="neg_root_mean_squared_error",
                              n_jobs=-1,
                              verbose=1,
                              random_state=42)
xgb_rand.fit(x_train, y_train)
print("Parametri optimi XGB:", xgb_rand.best_params_)
best_xgb = xgb_rand.best_estimator_

pipe_optimizat = {
    "RL": Pipeline([("scaler", StandardScaler()),("model", LinearRegression())]),
    "Ridge": Pipeline([("scaler", StandardScaler()), ("model", best_ridge)]),
    "RF": Pipeline([("scaler", StandardScaler()), ("model", best_rf)]),
    "KNN": Pipeline([("scaler", StandardScaler()), ("model", best_knn)]),
    "XGB": Pipeline([("scaler", StandardScaler()), ("model", best_xgb)])
}

metrici_normal = {
    nume: {"mae": [], "rmse": [], "r2": []}
    for nume in modele
}
metrici_optimizat = {
    nume: {"mae": [], "rmse": [], "r2": []}
    for nume in modele
}
print("\n--- 10 rulari ---")
for rs in range(10):
    x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(x, y, test_size=0.2, random_state=rs)


    for nume in modele:
        #varianta normala
        pipe_normal[nume].fit(x_train_r, y_train_r)
        pred = pipe_normal[nume].predict(x_test_r)

        metrici_normal[nume]["mae"].append(mean_absolute_error(y_test_r, pred))
        metrici_normal[nume]["rmse"].append(np.sqrt(mean_squared_error(y_test_r, pred)))
        metrici_normal[nume]["r2"].append(r2_score(y_test_r, pred))

        #varianta optimizata
        pipe_optimizat[nume].fit(x_train_r, y_train_r)
        pred = pipe_optimizat[nume].predict(x_test_r)

        metrici_optimizat[nume]["mae"].append(mean_absolute_error(y_test_r, pred))
        metrici_optimizat[nume]["rmse"].append(np.sqrt(mean_squared_error(y_test_r, pred)))
        metrici_optimizat[nume]["r2"].append(r2_score(y_test_r, pred))

    print(f"rulare {rs+1}/{10}")

#tabel metrici modele optimizate
print(f"\n{'Model':<8} {'MAE':>18} {'RMSE':>18} {'R2':>18}")
print("-" * 70)

for nume in modele:
    m = metrici_optimizat[nume]
    print(
        f"{nume:<8}"
        f" {np.mean(m['mae']):>8.3f} +- {np.std(m['mae']):.3f}"
        f" {np.mean(m['rmse']):>8.3f} +- {np.std(m['rmse']):.3f}"
        f" {np.mean(m['r2']):>8.3f} +- {np.std(m['r2']):.3f}")

#tabel comparatie varianta normala vs optimizata
print(
    f"\n{'Model':<8}"
    f"{'MAE normal':>18} {'MAE optim':>18}"
    f"{'RMSE normal':>18} {'RMSE optim':>18}"
    f"{'R2 normal':>18} {'R2 optim':>18}"
)
print("-"*130)
for nume in modele:
    normal = metrici_normal[nume]
    optim = metrici_optimizat[nume]
    print(
        f"{nume:<8}"
        f" {np.mean(normal['mae']):>8.3f} +- {np.std(normal['mae']):.3f}"
        f" {np.mean(optim['mae']):>8.3f} +- {np.std(optim['mae']):.3f}"
        f" {np.mean(normal['rmse']):>8.3f} +- {np.std(normal['rmse']):.3f}"
        f" {np.mean(optim['rmse']):>8.3f} +- {np.std(optim['rmse']):.3f}"
        f" {np.mean(normal['r2']):>8.3f} +- {np.std(normal['r2']):.3f}"
        f" {np.mean(optim['r2']):>8.3f} +- {np.std(optim['r2']):.3f}"
    )

#validare incrucisata modele optimizate
print("\n=== Validare incrucisata regresie(5fold)===")
for nume, pipe in pipe_optimizat.items():
    rezultate_cv = cross_validate(pipe, x, y, cv=5, scoring={
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2"
    },n_jobs=-1)
    mae_cv = -rezultate_cv["test_mae"]
    rmse_cv = -rezultate_cv["test_rmse"]
    r2_cv = rezultate_cv["test_r2"]

    print(
        f"{nume}: "
        f"MAE={mae_cv.mean():.3f} (+- {mae_cv.std():.3f}) "
        f"RMSE={rmse_cv.mean():.3f} (+- {rmse_cv.std():.3f}) "
        f"R2={r2_cv.mean():.3f} (+- {r2_cv.std():.3f})"
    )

#figura comparatie metrici medie modele optimizate
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

mae_medii = [np.mean(metrici_optimizat[nume]["mae"]) for nume in modele]
mae_deviatii = [np.std(metrici_optimizat[nume]["mae"]) for nume in modele]
rmse_medii = [np.mean(metrici_optimizat[nume]["rmse"]) for nume in modele]
rmse_deviatii = [np.std(metrici_optimizat[nume]["rmse"]) for nume in modele]
r2_medii = [np.mean(metrici_optimizat[nume]["r2"]) for nume in modele]
r2_deviatii = [np.std(metrici_optimizat[nume]["r2"]) for nume in modele]

axes[0].barh(modele, mae_medii, xerr=mae_deviatii, color="steelblue", capsize=5)
axes[0].set_xlabel("MAE mediu(10 rulari)")
axes[0].set_title("Comparatie modele - MAE")

axes[1].barh(modele, rmse_medii, xerr=rmse_deviatii, color="seagreen", capsize=5)
axes[1].set_xlabel("RMSE mediu(10 rulari)")
axes[1].set_title("Comparatie modele - RMSE")

axes[2].barh(modele, r2_medii, xerr=r2_deviatii, color="coral", capsize=5)
axes[2].set_xlabel("R2 mediu(10 rulari)")
axes[2].set_title("Comparatie modele - R2")

plt.tight_layout()
plt.savefig("../figuri_regresie/comparatie_metrici_10rulari.png", dpi=300, bbox_inches="tight")
plt.show()

#grafic comparatie R2 normal vs optimizat
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

r2_normal_medii = [np.mean(metrici_normal[nume]["r2"]) for nume in modele]
r2_normal_deviatii = [np.std(metrici_normal[nume]["r2"]) for nume in modele]
r2_optim_medii = [np.mean(metrici_optimizat[nume]["r2"]) for nume in modele]
r2_optim_deviatii = [np.std(metrici_optimizat[nume]["r2"]) for nume in modele]

axes[0].barh(modele, r2_normal_medii, xerr=r2_normal_deviatii, color="steelblue", capsize=5)
axes[0].set_xlabel("R2 mediu")
axes[0].set_title("R2 mediu - varianta normala +- deviatie")
axes[0].axvline(0, color="red", linestyle="--")

axes[1].barh(modele, r2_optim_medii, xerr=r2_optim_deviatii, color="seagreen", capsize=5)
axes[1].set_xlabel("R2 mediu")
axes[1].set_title("R2 mediu - varianta optimizata +- deviatie")
axes[1].axvline(0, color="red", linestyle="--")

plt.tight_layout()
plt.savefig("../figuri_regresie/comparatie_r2_normal_vs_optimizat.png", dpi=300, bbox_inches="tight")
plt.show()

#boxplot distributie r2

fig, axes = plt.subplots(1,2, figsize=(16, 6))

axes[0].boxplot([metrici_normal[nume]["r2"] for nume in modele], tick_labels=modele, vert=False)
axes[0].set_xlabel("R2")
axes[0].set_title("Distributia R2 - varianta normala")
axes[0].axvline(0, color="red", linestyle="--")

axes[1].boxplot([metrici_optimizat[nume]["r2"] for nume in modele], tick_labels=modele, vert=False)
axes[1].set_xlabel("R2")
axes[1].set_title("Distributia R2 - varianta optimizata")
axes[1].axvline(0, color="red", linestyle="--")

plt.tight_layout()
plt.savefig("../figuri_regresie/boxplot_r2.png", dpi=300, bbox_inches="tight")
plt.show()

#figuri XGB optimizat, split fix, scaler
x_train_fig, x_test_fig, y_train_fig, y_test_fig = train_test_split(x, y, test_size=0.2, random_state=42)

scaler_fig = StandardScaler()
x_train_fig = scaler_fig.fit_transform(x_train_fig)
x_test_fig = scaler_fig.transform(x_test_fig)

model_xgb_fig = XGBRegressor(**xgb_rand.best_params_, eval_metric="rmse", random_state=42, n_jobs=-1)
model_xgb_fig.fit(x_train_fig, y_train_fig)
pred_xgb_fig = model_xgb_fig.predict(x_test_fig)

#valori reale vs prezise xgboost
plt.figure(figsize=(8, 6))
plt.scatter(y_test_fig, pred_xgb_fig, alpha=0.3, s=10, color="steelblue")
plt.plot([0, 100], [0, 100], color="red", linestyle="--", label="Predictie perfecta")
plt.xlabel("Popularitate reala")
plt.ylabel("Popularitate prezisa")
plt.title("Valori reale vs valori prezise - XGBoost")
plt.legend()
plt.tight_layout()
plt.savefig("../figuri_regresie/valori_reale_vs_prezise_xgb.png", dpi=300, bbox_inches="tight")
plt.show()

#reziduuri xgboost
reziduuri = y_test_fig - pred_xgb_fig
plt.figure(figsize=(8, 6))
plt.scatter(pred_xgb_fig, reziduuri, alpha=0.3, s=10, color="seagreen")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Popularitate prezisa")
plt.ylabel("Reziduuri")
plt.title("Grafic reziduuri - XGBoost")
plt.tight_layout()
plt.savefig("../figuri_regresie/reziduuri_xgb.png", dpi=300, bbox_inches="tight")
plt.show()

#distributia reziduurilor XGBoost
plt.figure(figsize=(8, 5))
plt.hist(reziduuri, bins=50, color="mediumpurple", edgecolor="white")
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Reziduuri")
plt.ylabel("Numar")
plt.title("Distributia reziduurilor - XGBoost")
plt.tight_layout()
plt.savefig("../figuri_regresie/distributie_reziduuri_xgb.png", dpi=300, bbox_inches="tight")
plt.show()

#importanta caracteristicilor xgboost
importanta = model_xgb_fig.feature_importances_
indici = np.argsort(importanta)

plt.figure(figsize=(8, 6))
plt.barh(range(len(indici)), importanta[indici], color="darkorange")
plt.yticks(range(len(indici)), [caracteristici[i] for i in indici])
plt.title("Importanta caracteristicilor - XGBoost")
plt.tight_layout()
plt.savefig("../figuri_regresie/importanta_caracteristici_xgb.png", dpi=300, bbox_inches="tight")
plt.show()


