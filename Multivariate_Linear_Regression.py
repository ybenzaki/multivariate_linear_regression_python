# -*- coding: utf-8 -*-
# Multivariate linear regression : Housing prices predictions

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

df = pd.read_excel("D:\DEV\PYTHON_PROGRAMMING\coursera_ml_exercices_in_python\Multivariate_Linear_Regression_dataset.xlsx")

# df.head permet de voir les premières lignes chargées de notre fichier Excel
print df.head()

#Récupérer le prix : les valeurs observées pour la variable Cible
Y = df["prix"]
#Récupérer les variables prédictives : La superficie en pieds² et le nb chambre
X = df[['taille_en_pieds_carre','nb_chambres']]

# Normalisation des features (Features Scaling) : les valeurs seront 
# approximativement comprises entre -1 et 1.
# La Normalisation est utile quand les ordres de grandeur des valeurs de nos 
# features sont trés différents :
# En effet, Taille d'une maison en "pieds²" est de quelques miliers,
# alors que le nombre de chambre est généralement plus petit que 10

scale = StandardScaler()
X_scaled = scale.fit_transform(X[['taille_en_pieds_carre', 'nb_chambres']].as_matrix())

#print X_scaled

# OLS : Ordinary Least Squared : une méthode de regression pour estimer une 
# variable cible
# à Noter ici que X comporte nos deux variables prédictives
est = sm.OLS(Y, X).fit()

print est.summary()

fig = plt.figure()
#use this line to print only one figure
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(1,2,1, projection='3d')
ax.scatter(df["taille_en_pieds_carre"], df["nb_chambres"], df["prix"], c='r', marker='^')

ax.set_xlabel('surface en pieds_carre')
ax.set_ylabel('nb_chambres')
ax.set_zlabel('prix en $')



frst_col_surface =  df.iloc[0:len(df),0] #selection de la première colonne de notre dataset
scnd_col_nb_chambre =  df.iloc[0:len(df),1]
third_col_prix = df.iloc[0:len(df),2]

def predict_price_of_house(taille_maison, nb_chambre):
    return 140.8611 * taille_maison + 1.698e+04 * nb_chambre # not scaled
    #return 1.094e+05 * taille_maison + (6578.3549 * nb_chambre) # scaled

def predict_all(lst_sizes, lst_nb_chmbres):
    predicted_prices = []
    for n in range(0, len(Y)):
        predicted_prices.append(predict_price_of_house(lst_sizes[n], lst_nb_chmbres[n]))
    return predicted_prices

#print predict_all(df["taille_en_pieds_carre"], df["nb_chambres"])

# set up the axes for the second plot
ax = fig.add_subplot(1, 2, 2, projection='3d')

ax.plot_trisurf(df["taille_en_pieds_carre"], df["nb_chambres"], predict_all(df["taille_en_pieds_carre"], df["nb_chambres"]))


plt.show()

print predict_price_of_house(4500,5)


