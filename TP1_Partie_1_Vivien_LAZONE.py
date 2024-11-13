import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error


#### Partie 1 : Régression avec Distribution Gaussienne sur des données générées ####

### I- Régréssion Linéaire

## Question 1:

# Génération des données
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 7 + 4 * X + np.random.randn(100, 1)

# Visualisation des données
plt.scatter(X, y, color='red')
plt.xlabel("X")
plt.ylabel("y")
plt.title("I- 1- Génération et visualiation des données")
plt.show()


## Question 2:

model = LinearRegression()
model.fit(X, y)
y_prediction = model.predict(X)

# Visualisation avec prédictions
plt.scatter(X, y, color='red')
plt.plot(X, y_prediction, color='blue')
plt.title("I- 2- Régression linéaire")
plt.xlabel("X")
plt.ylabel("y")
plt.show()


## Question 3:

# Résidus
residus = y - y_prediction


## Question 4:

# Histogramme des résidus
sns.histplot(residus, kde=True)
plt.title("I- 4- Histogramme des résidus")
plt.show()

# Graphique de probabilité normale (Q-Q plot) des résidus
stats.probplot(residus.flatten(), dist="norm", plot=plt)
plt.title("I- 4- Graphique de probabilité normale (Q-Q plot) des résidus")
plt.show()

'''
Interprétation:

Les résidus suivent une distribution gaussienne car ils forment 
une cloche symétrique autour de zéro dans l'histogramme des résidus, 
et s'alignent plus ou moins sur la ligne de référence dans le Q-Q plot des résidus.
'''


### II- Régression Polynomiale avec Distribution Gaussienne

## Question 5:

poly_features = PolynomialFeatures(degree=2)
X_polynomiale = poly_features.fit_transform(X)
model_polynomiale = LinearRegression()
model_polynomiale.fit(X_polynomiale, y)
y_polynomiale_prediction = model_polynomiale.predict(X_polynomiale)

# Affichage des prédictions
plt.scatter(X, y, color='green')
plt.plot(X, y_polynomiale_prediction, color='blue')
plt.title("II- 5- Régression polynomiale")
plt.xlabel("X")
plt.ylabel("y")
plt.show()


## Question 6:

# Résidus
residuals_polynomiale = y - y_polynomiale_prediction

# Histogramme des résidus
sns.histplot(residuals_polynomiale, kde=True)
plt.title("II- 6- Histogramme des résidus (Régression Polynomiale)")
plt.show()

# Graphique de probabilité normale (Q-Q plot) des résidus
stats.probplot(residuals_polynomiale.flatten(), dist="norm", plot=plt)
plt.title("II- 6- Graphique de probabilité normale (Q-Q plot) des résidus (Régression Polynomiale)")
plt.show()

'''
Interprétation:

Les résidus suivent une distribution gaussienne car ils forment 
une cloche symétrique autour de zéro dans l'histogramme des résidus, 
et s'alignent plus ou moins sur la ligne de référence dans le Q-Q plot des résidus.
'''


## Question 7:

# Modèle linéaire
r2_lineaire = r2_score(y, y_prediction)
rmse_lineaire = mean_squared_error(y, y_prediction)

# Modèle polynomial
r2_polynomial = r2_score(y, y_polynomiale_prediction)
rmse_polynomial = mean_squared_error(y, y_polynomiale_prediction)

print(f"Modèle linéaire -> R2 : {r2_lineaire}, RMSE : {rmse_lineaire}")
print(f"Modèle polynomial -> R2 : {r2_polynomial}, RMSE : {rmse_polynomial}")


## Question 8:

'''
Interprétation:

Les coefficients indiquent l'impact de chaque terme de la régression 
linéaire et polynomiale sur la variable cible y. 
Un coefficient plus élevé signifie qu'il y aura une plus grande influence 
sur y.
'''