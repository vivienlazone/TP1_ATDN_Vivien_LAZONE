import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Importation des données
data = pd.read_csv("house_data/train.csv")
X = data[['GrLivArea']]  # variables explicatives
y = data['SalePrice']  # variable cible

# Régression linéaire
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Visualisation
plt.scatter(X, y, color='red')
plt.plot(X, y_pred, color='blue')
plt.title("Régression linéaire sur House Prices")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()

##

# Régression polynomiale de degré 2
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
y_poly_pred = model_poly.predict(X_poly)

# Visualisation
plt.scatter(X, y, color='green')
plt.plot(X, y_poly_pred, color='blue')
plt.title("Régression polynomiale sur House Prices")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()

##

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_poly, y)
y_ridge_pred = ridge_model.predict(X_poly)

# Comparaison
print("Coefficients avec Ridge : ", ridge_model.coef_)

##

scores_lin = cross_val_score(model, X, y, cv=5, scoring='r2')
scores_poly = cross_val_score(model_poly, X_poly, y, cv=5, scoring='r2')
print("Scores de la validation croisée linéaire : ", scores_lin)
print("Scores de la validation croisée polynomiale : ", scores_poly)

##

# Définition des événements

# On prend plus de 3 chambres
condition_bedrooms = data['BedroomAbvGr'] > 3

# On suppose le quartier "CollgCr" comme critère de localisation
condition_neighborhood = data['Neighborhood'] == 'CollgCr'

# Calcul de la probabilité conjointe
joint_probability = (condition_bedrooms & condition_neighborhood).mean()
print(f"Probabilité conjointe qu'une maison ait plus de 3 chambres et soit dans le quartier 'CollgCr' : {joint_probability:.2f}")

# Tableau croisé dynamique
cross_tab = pd.crosstab(condition_bedrooms, condition_neighborhood, normalize='all')
print("\nTableau croisé dynamique des probabilités conjointes :\n", cross_tab)

##

# Définition des probabilités de base
P_economique = 0.3       # Probabilité a priori qu'une voiture soit économique (P(A))
P_faible_conso_economique = 0.8  # Probabilité qu'une voiture ait faible consommation sachant qu'elle est économique (P(B|A))
P_faible_conso = 0.6     # Probabilité qu'une voiture ait faible consommation (P(B))

# Calcul de la probabilité a posteriori avec le théorème de Bayes
P_economique_faible_conso = (P_faible_conso_economique * P_economique) / P_faible_conso
print(f"Probabilité qu'une voiture soit économique sachant qu'elle a une faible consommation : {P_economique_faible_conso:.2f}")
