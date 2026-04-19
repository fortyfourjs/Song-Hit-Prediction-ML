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

os.makedirs("../plots_regresie", exist_ok=True)

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
y = df["popularity"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#fct evaluare modele
def evaluare(name, y_test, pred):
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"\n=== {name} ===")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")
    return{"Model": name, "MAE":mae, "RMSE":rmse, "R2":r2}

#Linear Regression
lr = LinearRegression()
lr.fit(x_train, y_train)
pred_lr = lr.predict(x_test)
rezultate = [evaluare("Linear Regression", y_test, pred_lr)]

#Ridge baseline
ridge = Ridge(alpha=1.0)
ridge.fit(x_train, y_train)
pred_ridge_base = ridge.predict(x_test)
rezultate.append(evaluare("Ridge (base)", y_test, pred_ridge_base))

ridge_param_grid = {
    "alpha": [0.01, 0.1, 1.0, 10, 100]
}
ridge_grid = GridSearchCV(Ridge(), ridge_param_grid,
                          cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1)
ridge_grid.fit(x_train, y_train)
print("Best Ridge params:", ridge_grid.best_params_)
best_ridge = ridge_grid.best_estimator_
pred_ridge_tuned = best_ridge.predict(x_test)
rezultate.append(evaluare("Ridge (tuned)", y_test, pred_ridge_tuned))



#Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(x_train, y_train)
pred_rf_base = rf.predict(x_test)
rezultate.append(evaluare("Random Forest (base)", y_test, pred_rf_base))

rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),rf_param_grid,
                       cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
rf_grid.fit(x_train, y_train)
print("Best RF params:", rf_grid.best_params_)
best_rf = rf_grid.best_estimator_
pred_rf_tuned = best_rf.predict(x_test)
rezultate.append(evaluare("Random Forest (tuned)", y_test, pred_rf_tuned))


#KNN regressor
knn = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
knn.fit(x_train, y_train)
pred_knn_base= knn.predict(x_test)
rezultate.append(evaluare("KNN (base)", y_test, pred_knn_base))

knn_param_grid = {
    "n_neighbors": [3, 5, 7, 11, 15, 21],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}
knn_grid = GridSearchCV(KNeighborsRegressor(n_jobs=-1), knn_param_grid,
                        cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1)
knn_grid.fit(x_train, y_train)
print("Best KNN params:", knn_grid.best_params_)
best_knn = knn_grid.best_estimator_
pred_knn_tuned = best_knn.predict(x_test)
rezultate.append(evaluare("KNN (tuned)", y_test, pred_knn_tuned))

#XGBoost Regressor
xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6,
                   random_state=42, eval_metric="rmse", n_jobs=-1)
xgb.fit(x_train, y_train)
pred_xgb_base = xgb.predict(x_test)
rezultate.append(evaluare("XGBoost (base)", y_test, pred_xgb_base))

xgb_param_dist = {
    "n_estimators": randint(100, 400),
    "max_depth": randint(3, 10),
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}
xgb_rand = RandomizedSearchCV(XGBRegressor(eval_metric="rmse", random_state=42, n_jobs=-1),
                              xgb_param_dist, n_iter=15, cv=3,
                              scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=42)
xgb_rand.fit(x_train, y_train)
print("Best XGB params:", xgb_rand.best_params_)
best_xgb = xgb_rand.best_estimator_
pred_xgb_tuned = best_xgb.predict(x_test)
rezultate.append(evaluare("XGB (tuned)", y_test, pred_xgb_tuned))


#actual vs predicted plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, pred_xgb_tuned, alpha=0.3, s=10, color="steelblue")
plt.plot([0, 100], [0, 100], color="red", linestyle="--", label="Perfect prediction")
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Actual vs Predicted-XGBoost")
plt.legend()
plt.savefig("../plots_regresie/actual_vs_predicted.pdf", bbox_inches="tight")
plt.show()

#Residual plots
residuals = y_test - pred_xgb_tuned
plt.figure(figsize=(8,6))
plt.scatter(pred_xgb_tuned, residuals, alpha=0.3, s=10, color="seagreen")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Popularity")
plt.ylabel("Residuals")
plt.title("Residual Plot-XGBoost")
plt.savefig("../plots_regresie/residuals.pdf", bbox_inches="tight")
plt.show()

#MAE RMSE comparatie --- use "best_models.predict(x_test) for tuning
model_name = ["Linear Regression", "Ridge", "Random Forest", "KNN", "XGBoost"]
mae_scores = [mean_absolute_error(y_test, p) for p in [pred_lr, pred_ridge_tuned, pred_rf_tuned, pred_knn_tuned, pred_xgb_tuned]]
rmse_scores = [np.sqrt(mean_squared_error(y_test, p)) for p in [pred_lr, pred_ridge_tuned, pred_rf_tuned, pred_knn_tuned, pred_xgb_tuned]]

x = np.arange(len(model_name))
width = 0.35
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(x-width/2, mae_scores, width, label="MAE", color="steelblue")
ax.bar(x+width/2, rmse_scores, width, label="RMSE", color="seagreen")
ax.set_xticks(x)
ax.set_xticklabels(model_name)
ax.set_ylabel("Error")
ax.set_title("MAE vs RMSE")
ax.legend()
plt.tight_layout()
plt.savefig("../plots_regresie/mae_rmse.pdf", bbox_inches="tight")
plt.show()

#r2
r2_scores = [r2_score(y_test, p) for p in [pred_lr, pred_ridge_tuned, pred_rf_tuned, pred_knn_tuned, pred_xgb_tuned]]
plt.figure(figsize=(8, 5))
plt.barh(model_name, r2_scores, color="steelblue")
plt.xlabel("R2 Score")
plt.title("R2 Score Comparatie")
plt.axvline(0, color="red", linestyle="--")
plt.tight_layout()
plt.savefig("../plots_regresie/r2_comparatie.pdf", bbox_inches="tight")
plt.show()

#residual distribution
residuals = y_test - pred_xgb_tuned
plt.figure(figsize=(8,5))
plt.hist(residuals, bins=50, color="mediumpurple", edgecolor="white")
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.title("Distributia reziduurilor - XGBoost")
plt.savefig("../plots_regresie/residuals_distribution.pdf", bbox_inches="tight")
plt.show()

#feature importance
importance = best_xgb.feature_importances_
indices = np.argsort(importance)
plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importance[indices], color="darkorange")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title("Feature importances - XGBoost Regressor")
plt.savefig("../plots_regresie/feature_importances.pdf", bbox_inches="tight")
plt.show()

#tabel comparatie base vs tuned
model = ["Linear Regression", "Ridge", "Random Forest", "KNN", "XGBoost"]
baseline_preds = [pred_lr, pred_ridge_base, pred_rf_base, pred_knn_base, pred_xgb_base]
tuned_preds = [pred_lr, pred_ridge_tuned, pred_rf_tuned, pred_knn_tuned, pred_xgb_tuned]

print(f"\n{'Model':<20} {'MAE base':>10} {'MAE tuned' :>10} {'RMSE base':>10} {'RMSE tuned':>10} {'R2 base':>8}{'R2 tuned':>8}")
print("-" * 80)
for name, base, tuned in zip(model, baseline_preds, tuned_preds):
    print(f"{name:<20}"
          f"{mean_absolute_error(y_test, base):>10.3f}"
          f"{mean_absolute_error(y_test, tuned):>10.3f}"
          f"{np.sqrt(mean_squared_error(y_test, base)):>10.3f}"
          f"{np.sqrt(mean_squared_error(y_test, tuned)):>10.3f}"
          f"{r2_score(y_test, base):>8.3f}"
          f"{r2_score(y_test, tuned):>8.3f}")

print("\n=== Cross Validation Regresie 5fold ===")
for name, model in [
    ("Linear Regression", lr),
    ("Ridge", best_ridge),
    ("Random Forest", best_rf),
    ("KNN", best_knn),
    ("XGBoost", best_xgb),

]:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    scores = cross_val_score(pipe, df[features], df["popularity"],
                             cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    print(f"{name}: RMSE={(-scores.mean()):.3f} (+/- {scores.std():.3f})")