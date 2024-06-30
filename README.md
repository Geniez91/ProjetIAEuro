# Prédiction des Scores de Matchs de Football

## Algorithme de Machine Learning

### RandomForestRegressor

Nous utilisons `RandomForestRegressor` pour prédire les scores des matchs de football. Cet algorithme est basé sur les forêts aléatoires et offre plusieurs avantages :

1. **Robustesse** : En combinant plusieurs arbres de décision, il réduit la variance et améliore la précision.
2. **Réduction du Surapprentissage** : L'utilisation de bootstrap sampling et de la sélection aléatoire des caractéristiques aide à éviter le surapprentissage.
3. **Précision** : Il est souvent plus précis que les arbres de décision individuels pour des jeux de données complexes.

## Nettoyage des Données

### Étapes de Prétraitement

1. **Encodage des Variables Catégorielles** : Les noms des équipes sont encodés en utilisant `LabelEncoder`.
2. **Normalisation des Données** : Les caractéristiques numériques telles que l'humidité, la température et la vitesse du vent sont normalisées à l'aide de `StandardScaler`.
3. **Gestion des Valeurs Manquantes** : Les valeurs manquantes dans la colonne des noms des arbitres assistants sont remplacées par 'Unknown'.
4. **Conversion des Dates** : La colonne `DateandTimeCET` est convertie en type datetime pour faciliter les opérations de filtrage et de manipulation des dates.

### Entraînement du Modèle
1. Division des données en ensembles d'entraînement et de test.
2. Entraînement du modèle `RandomForestRegressor` sur les données d'entraînement.
3. Évaluation du modèle en utilisant l'erreur quadratique moyenne (MSE).

### Explication MSE
Pour interpréter cette valeur de MSE, considérons quelques points clés :

### Différence au Carré :

La MSE calcule la moyenne des carrés des différences entre les valeurs réelles et les valeurs prédites. Donc, une MSE de 2.82 signifie que les différences entre les scores réels et les scores prédits, une fois mises au carré, ont une moyenne de 2.82.

### Erreur Moyenne :

Pour comprendre l'erreur en termes de différence absolue moyenne, nous pouvons prendre la racine carrée de la MSE. Cette valeur est connue sous le nom de Root Mean Squared Error (RMSE).

Dans ce cas :
\[ \text{RMSE} = \sqrt{2.82} \approx 1.68 \]

Cela signifie que, en moyenne, la prédiction de votre modèle est à environ 1.68 unités du score réel.

## Équipe de Développement

- Jérémy Weltmann
- Ibrahima Berete
- Brayan Kutlar

Pour l'éxécution du projet : 

- Tout d'abord télécharger toute les dépendances à l'aide de la commande suivante : 
pip install -r requirements.txt

- En parallèle éxécuter la commande suivante pour l'affichage front : 
streamlit run euro.py