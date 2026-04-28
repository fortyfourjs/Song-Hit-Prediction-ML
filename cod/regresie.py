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
from sklearn.model_selection import RandomizedSearchCV
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#fct evaluare modele
def evaluare(nume, y_test, predictie):
    mae = mean_absolute_error(y_test, predictie)
    rmse = np.sqrt(mean_squared_error(y_test, predictie))
    r2 = r2_score(y_test, predictie)
    print(f"\n=== {nume} ===")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")
    return{"Model": nume, "MAE":mae, "RMSE":rmse, "R2":r2}

#Regresie liniara
rl = LinearRegression()
rl.fit(x_train, y_train)
pred_rl = rl.predict(x_test)
rezultate = [evaluare("Linear Regression", y_test, pred_rl)]

#Ridge
ridge_normal = Ridge(alpha=1.0)
ridge_normal.fit(x_train, y_train)
pred_ridge_normal = ridge_normal.predict(x_test)
rezultate.append(evaluare("Ridge (normal)", y_test, pred_ridge_normal))

ridge_parametri = {
    "alpha": [0.01, 0.1, 1.0, 10, 100]
}
ridge_cautare = GridSearchCV(Ridge(), ridge_parametri,
                          cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1)
ridge_cautare.fit(x_train, y_train)
print("Parametri optimi Ridge:", ridge_cautare.best_params_)
ridge_optimizat = ridge_cautare.best_estimator_
pred_ridge_optimizat = ridge_optimizat.predict(x_test)
rezultate.append(evaluare("Ridge (optimizat)", y_test, pred_ridge_optimizat))

#Random Forest
rf_normal = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_normal.fit(x_train, y_train)
pred_rf_normal = rf_normal.predict(x_test)
rezultate.append(evaluare("Random Forest (normal)", y_test, pred_rf_normal))

rf_parametri = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
rf_cautare = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),rf_parametri,
                       cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
rf_cautare.fit(x_train, y_train)
print("Parametri optimi RF:", rf_cautare.best_params_)
rf_optimizat = rf_cautare.best_estimator_
pred_rf_optimizat = rf_optimizat.predict(x_test)
rezultate.append(evaluare("Random Forest (optimizat)", y_test, pred_rf_optimizat))


#KNN
knn_normal = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
knn_normal.fit(x_train, y_train)
pred_knn_normal= knn_normal.predict(x_test)
rezultate.append(evaluare("KNN (normal)", y_test, pred_knn_normal))

knn_parametri = {
    "n_neighbors": [3, 5, 7, 11, 15, 21],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}
knn_cautare = GridSearchCV(KNeighborsRegressor(n_jobs=-1), knn_parametri,
                        cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1)
knn_cautare.fit(x_train, y_train)
print("Parametri optimi KNN:", knn_cautare.best_params_)
knn_optimizat = knn_cautare.best_estimator_
pred_knn_optimizat = knn_optimizat.predict(x_test)
rezultate.append(evaluare("KNN (optimizat)", y_test, pred_knn_optimizat))

#XGBoost
xgb_normal = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6,
                   random_state=42, eval_metric="rmse", n_jobs=-1)
xgb_normal.fit(x_train, y_train)
pred_xgb_normal = xgb_normal.predict(x_test)
rezultate.append(evaluare("XGBoost (normal)", y_test, pred_xgb_normal))

xgb_parametri = {
    "n_estimators": randint(100, 400),
    "max_depth": randint(3, 10),
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}
xgb_rand = RandomizedSearchCV(XGBRegressor(eval_metric="rmse", random_state=42, n_jobs=-1),
                              xgb_parametri, n_iter=15, cv=3,
                              scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=42)
xgb_rand.fit(x_train, y_train)
print("Parametri optimi XGB:", xgb_rand.best_params_)
xgb_optimizat = xgb_rand.best_estimator_
pred_xgb_optimizat = xgb_optimizat.predict(x_test)
rezultate.append(evaluare("XGB (optimizat)", y_test, pred_xgb_optimizat))


#valori reale vs valori prezise
plt.figure(figsize=(8,6))
plt.scatter(y_test, pred_xgb_optimizat, alpha=0.3, s=10, color="steelblue")
plt.plot([0, 100], [0, 100], color="red", linestyle="--", label="Predictie perfecta")
plt.xlabel("Popularitate reala")
plt.ylabel("Popularitate prezisa")
plt.title("Valori reale vs valori prezise - XGBoost")
plt.legend()
plt.savefig("../figuri_regresie/valori_reale_vs_prezise.pdf", bbox_inches="tight")
plt.show()

#reziduuri(xgb)
residuals = y_test - pred_xgb_optimizat
plt.figure(figsize=(8,6))
plt.scatter(pred_xgb_optimizat, residuals, alpha=0.3, s=10, color="seagreen")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Popularitate prezisa")
plt.ylabel("Reziduuri")
plt.title("Grafic reziduuri-XGBoost")
plt.savefig("../figuri_regresie/reziduuri.pdf", bbox_inches="tight")
plt.show()

#comparatie mae vs rmse
model = ["Regresie liniara", "Ridge", "Random Forest", "KNN", "XGBoost"]
scor_mae = [mean_absolute_error(y_test, p) for p in [pred_rl, pred_ridge_optimizat, pred_rf_optimizat, pred_knn_optimizat, pred_xgb_optimizat]]
scor_rmse = [np.sqrt(mean_squared_error(y_test, p)) for p in [pred_rl, pred_ridge_optimizat, pred_rf_optimizat, pred_knn_optimizat, pred_xgb_optimizat]]

x = np.arange(len(model))
width = 0.35
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(x-width/2, scor_mae, width, label="MAE", color="steelblue")
ax.bar(x+width/2, scor_rmse, width, label="RMSE", color="seagreen")
ax.set_xticks(x)
ax.set_xticklabels(model)
ax.set_ylabel("Eroare")
ax.set_title("Comparatie MAE vs RMSE")
ax.legend()
plt.tight_layout()
plt.savefig("../figuri_regresie/comparatie_mae_rmse.pdf", bbox_inches="tight")
plt.show()

#r2
scor_r2 = [r2_score(y_test, p) for p in [pred_rl, pred_ridge_optimizat, pred_rf_optimizat, pred_knn_optimizat, pred_xgb_optimizat]]
plt.figure(figsize=(8, 5))
plt.barh(model, scor_r2, color="steelblue")
plt.xlabel("Scor R2")
plt.title("Comparatie scor R2")
plt.axvline(0, color="red", linestyle="--")
plt.tight_layout()
plt.savefig("../figuri_regresie/comparatie_scor_r2.pdf", bbox_inches="tight")
plt.show()

#distributia reziduurilor(xgb)
reziduuri = y_test - pred_xgb_optimizat
plt.figure(figsize=(8,5))
plt.hist(reziduuri, bins=50, color="mediumpurple", edgecolor="white")
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Reziduuri")
plt.ylabel("Numar")
plt.title("Distributia reziduurilor - XGBoost")
plt.savefig("../figuri_regresie/distributie_reziduuri.pdf", bbox_inches="tight")
plt.show()

#importanta caracteristicilor(xgb)
importanta = xgb_optimizat.feature_importances_
indici = np.argsort(importanta)
plt.figure(figsize=(8,6))
plt.barh(range(len(indici)), importanta[indici], color="darkorange")
plt.yticks(range(len(indici)), [caracteristici[i] for i in indici])
plt.title("Importanta caracteristicilor - XGBoost regresie")
plt.savefig("../figuri_regresie/importanta_caracteristici.pdf", bbox_inches="tight")
plt.show()

#comparatie modele normale vs optimizate
model = ["Regresie liniara", "Ridge", "Random Forest", "KNN", "XGBoost"]
pred_normal = [pred_rl, pred_ridge_normal, pred_rf_normal, pred_knn_normal, pred_xgb_normal]
pred_optimizat = [pred_rl, pred_ridge_optimizat, pred_rf_optimizat, pred_knn_optimizat, pred_xgb_optimizat]

print(f"\n{'Model':<20} {'MAE normal':>10} {'MAE optimizat' :>10} {'RMSE normal':>10} {'RMSE optimizat':>10} {'R2 normal':>8}{'R2 optimizat':>8}")
print("-" * 80)
for nume, normal, optimizat in zip(model, pred_normal, pred_optimizat):
    print(f"{nume:<20}"
          f"{mean_absolute_error(y_test, normal):>10.3f}"
          f"{mean_absolute_error(y_test, optimizat):>10.3f}"
          f"{np.sqrt(mean_squared_error(y_test, normal)):>10.3f}"
          f"{np.sqrt(mean_squared_error(y_test, optimizat)):>10.3f}"
          f"{r2_score(y_test, normal):>8.3f}"
          f"{r2_score(y_test, optimizat):>8.3f}")

print("\n=== Validare incrusisata regresie 5fold ===")
for nume, model in [
    ("Regresie liniara", rl),
    ("Ridge", ridge_optimizat),
    ("Random Forest", rf_optimizat),
    ("KNN", knn_optimizat),
    ("XGBoost", xgb_optimizat),

]:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    scor = cross_val_score(pipe, df[caracteristici], df["popularity"],
                             cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    print(f"{nume}: RMSE={(-scor.mean()):.3f} (+/- {scor.std():.3f})")