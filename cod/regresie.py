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

#Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(x_train, y_train)
pred_ridge = ridge.predict(x_test)
rezultate.append(evaluare("Ridge", y_test, pred_ridge))

#Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(x_train, y_train)
pred_rf = rf.predict(x_test)
rezultate.append(evaluare("Random Forest", y_test, pred_rf))

#KNN regressor
knn = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
knn.fit(x_train, y_train)
pred_knn = knn.predict(x_test)
rezultate.append(evaluare("KNN", y_test, pred_knn))

#XGBoost Regressor
xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6,
                   random_state=42, eval_metric="rmse", n_jobs=-1)
xgb.fit(x_train, y_train)
pred_xgb = xgb.predict(x_test)
rezultate.append(evaluare("XGBoost", y_test, pred_xgb))

#actual vs predicted plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, pred_xgb, alpha=0.3, s=10, color="steelblue")
plt.plot([0, 100], [0, 100], color="red", linestyle="--", label="Perfect prediction")
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Actual vs Predicted-XGBoost")
plt.legend()
plt.savefig("../plots_regresie/actual_vs_predicted.pdf", bbox_inches="tight")
plt.show()

#Residual plots
residuals = y_test - pred_xgb
plt.figure(figsize=(8,6))
plt.scatter(pred_xgb, residuals, alpha=0.3, s=10, color="seagreen")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Popularity")
plt.ylabel("Residuals")
plt.title("Residual Plot-XGBoost")
plt.savefig("../plots_regresie/residuals.pdf", bbox_inches="tight")
plt.show()

#MAE RMSE comparatie
model_name = ["Linear Regression", "Ridge", "Random Forest", "KNN", "XGBoost"]
mae_scores = [mean_absolute_error(y_test, p) for p in [pred_lr, pred_ridge, pred_rf, pred_knn, pred_xgb]]
rmse_scores = [np.sqrt(mean_squared_error(y_test, p)) for p in [pred_lr, pred_ridge, pred_rf, pred_knn, pred_xgb]]

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
r2_scores = [r2_score(y_test, p) for p in [pred_lr, pred_ridge, pred_rf, pred_knn, pred_xgb]]
plt.figure(figsize=(8, 5))
plt.barh(model_name, r2_scores, color="steelblue")
plt.xlabel("R2 Score")
plt.title("R2 Score Comparatie")
plt.axvline(0, color="red", linestyle="--")
plt.tight_layout()
plt.savefig("../plots_regresie/r2_comparatie.pdf", bbox_inches="tight")
plt.show()

#residual distribution
residuals = y_test - pred_xgb
plt.figure(figsize=(8,5))
plt.hist(residuals, bins=50, color="mediumpurple", edgecolor="white")
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.title("Distributia reziduurilor - XGBoost")
plt.savefig("../plots_regresie/residuals.pdf", bbox_inches="tight")
plt.show()

#feature importance
importance = xgb.feature_importances_
indices = np.argsort(importance)
plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importance[indices], color="darkorange")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title("Feature importances - XGBoost Regressor")
plt.savefig("../plots_regresie/feature_importances.pdf", bbox_inches="tight")
plt.show()