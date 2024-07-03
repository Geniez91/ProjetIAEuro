import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

# Charger le fichier CSV
data = pd.read_csv('match_data.csv')

# Conversion de la colonne DateandTimeCET en type datetime
data['DateandTimeCET'] = pd.to_datetime(data['DateandTimeCET'])

# Remplissage des valeurs manquantes pour les arbitres assistants (remplacement par 'Unknown')
data['AssistantRefereeWebName'].fillna('Unknown', inplace=True)

# Encodage des variables catégorielles
label_encoder = LabelEncoder()
data['HomeTeamName'] = label_encoder.fit_transform(data['HomeTeamName'])
data['AwayTeamName'] = label_encoder.fit_transform(data['AwayTeamName'])

# Sauvegarde des valeurs d'origine pour affichage ultérieur
original_data = data[['Humidity', 'Temperature', 'WindSpeed']].copy()

# Normalisation des variables numériques
scaler = StandardScaler()
data[['Humidity', 'Temperature', 'WindSpeed']] = scaler.fit_transform(data[['Humidity', 'Temperature', 'WindSpeed']])

# Sélection des fonctionnalités et de la cible
features = data[['HomeTeamName', 'AwayTeamName', 'Humidity', 'Temperature', 'WindSpeed']]
target = data[['ScoreHome', 'ScoreAway']]

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Entraînement du modèle Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)

# Importance des caractéristiques
feature_importances = model.feature_importances_
features_list = ['HomeTeamName', 'AwayTeamName', 'Humidity', 'Temperature', 'WindSpeed']

# Interface Streamlit
st.title("Analyse et Prédiction des Scores de Matchs de Football")

st.write("""
## Étapes de Prétraitement
- **Encodage des Variables Catégorielles** : Les noms des équipes sont encodés en utilisant `LabelEncoder`.
- **Normalisation des Données** : Les caractéristiques numériques telles que l'humidité, la température et la vitesse du vent sont normalisées à l'aide de `StandardScaler`.
- **Gestion des Valeurs Manquantes** : Les valeurs manquantes dans la colonne des noms des arbitres assistants sont remplacées par 'Unknown'.
- **Conversion des Dates** : La colonne `DateandTimeCET` est convertie en type datetime pour faciliter les opérations de filtrage et de manipulation des dates.

## Entraînement du Modèle
1. Division des données en ensembles d'entraînement et de test.
2. Entraînement du modèle `RandomForestRegressor` sur les données d'entraînement.
3. Évaluation du modèle en utilisant l'erreur quadratique moyenne (MSE).

## Algorithme de Machine Learning
### RandomForestRegressor

Nous utilisons `RandomForestRegressor` pour prédire les scores des matchs de football. Cet algorithme est basé sur les forêts aléatoires et offre plusieurs avantages :

- **Robustesse** : En combinant plusieurs arbres de décision, il réduit la variance et améliore la précision.
- **Réduction du Surapprentissage** : L'utilisation de bootstrap sampling et de la sélection aléatoire des caractéristiques aide à éviter le surapprentissage.
- **Précision** : Il est souvent plus précis que les arbres de décision individuels pour des jeux de données complexes.

## Explication MSE

Pour interpréter cette valeur de MSE, considérons quelques points clés :

### Différence au Carré :

La MSE calcule la moyenne des carrés des différences entre les valeurs réelles et les valeurs prédites. Donc, une MSE de 2.82 signifie que les différences entre les scores réels et les scores prédits, une fois mises au carré, ont une moyenne de 2.82.

### Erreur Moyenne :

Pour comprendre l'erreur en termes de différence absolue moyenne, nous pouvons prendre la racine carrée de la MSE. Cette valeur est connue sous le nom de Root Mean Squared Error (RMSE).

Dans ce cas :
\[ \text{RMSE} = \sqrt{2.82} \approx 1.68 \]

Cela signifie que, en moyenne, la prédiction de votre modèle est à environ 1.68 unités du score réel.
""")

st.write("""
### Description des Données
""")
st.write(data.head())

st.write("""
### Analyse Exploratoire des Données
""")

# Distribution des scores
fig, ax = plt.subplots()
sns.histplot(data['ScoreHome'], kde=True, color='blue', label='Home Score', ax=ax)
sns.histplot(data['ScoreAway'], kde=True, color='red', label='Away Score', ax=ax)
ax.legend()
ax.set_title('Distribution des scores')
ax.set_xlabel('Score')
ax.set_ylabel('Pourcentage')
st.pyplot(fig)

# Nombre de matchs par stade
fig, ax = plt.subplots()
stadium_counts = data['StadiumID'].value_counts()
sns.barplot(x=stadium_counts.index, y=stadium_counts.values, ax=ax)
ax.set_title('Nombre de matchs par stade')
ax.set_xlabel('Stadium ID')
ax.set_ylabel('Nombre de matchs')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

# Fréquence d'arbitrage par arbitre
fig, ax = plt.subplots()
referee_counts = data['RefereeWebName'].value_counts()
sns.barplot(x=referee_counts.index, y=referee_counts.values, ax=ax)
ax.set_title("Fréquence d'arbitrage par arbitre")
ax.set_xlabel("Nom de l'arbitre")
ax.set_ylabel("Nombre de matchs arbitrés")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

fig, ax = plt.subplots()
round_counts = data['RoundName'].value_counts()
sns.barplot(x=round_counts.index, y=round_counts.values, ax=ax)
ax.set_title("Distribution des matchs par round")
ax.set_xlabel("Round")
ax.set_ylabel("Nombre de matchs")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

# Nombre d'équipes par round
teams_per_round = data.groupby('RoundName').apply(lambda x: len(pd.unique(x[['HomeTeamName', 'AwayTeamName']].values.ravel('K')))).reset_index(name='TeamCount')
fig, ax = plt.subplots()
round_order = ['final tournament', 'eighth finals', 'quarter finals', 'semi finals', 'final']
sns.barplot(x=teams_per_round['RoundName'], y=teams_per_round['TeamCount'], order=round_order, ax=ax)
ax.set_title("Nombre d'équipes par round")
ax.set_xlabel("Round")
ax.set_ylabel("Nombre d'équipes")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

st.write("""
### Importance des Caractéristiques
""")

# Importance des caractéristiques
fig, ax = plt.subplots()
sns.barplot(x=features_list, y=feature_importances, ax=ax)
ax.set_title('Importance des caractéristiques')
ax.set_xlabel('Caractéristiques')
ax.set_ylabel('Importance')
plt.xticks(rotation=90) 
st.pyplot(fig)

st.write("""
### Cas d'Usage de Machine Learning
Nous allons prédire les scores des matchs en fonction des équipes et des conditions météorologiques.
""")

st.write(f"Erreur quadratique moyenne (MSE) : {mse}")

# Interface pour prédire les scores d'un match
st.write("""
### Prédiction des Scores
""")

teams = label_encoder.inverse_transform(data['HomeTeamName'].unique())
home_team = st.selectbox('Équipe à domicile', teams)
away_team = st.selectbox('Équipe en déplacement', teams)

# Vérifiez que les équipes ne s'affrontent pas elles-mêmes
if home_team == away_team:
    st.warning("L'équipe à domicile et l'équipe en déplacement doivent être différentes.")
else:
    humidity = st.slider('Humidité (%)', int(original_data['Humidity'].min()), int(original_data['Humidity'].max()))
    temperature = st.slider('Température (°C)', int(original_data['Temperature'].min()), int(original_data['Temperature'].max()))
    wind_speed = st.slider('Vitesse du vent (km/h)', int(original_data['WindSpeed'].min()), int(original_data['WindSpeed'].max()))

    # Préparation des données pour la prédiction
    home_team_encoded = label_encoder.transform([home_team])[0]
    away_team_encoded = label_encoder.transform([away_team])[0]
    input_data = pd.DataFrame([[home_team_encoded, away_team_encoded, humidity, temperature, wind_speed]],
                              columns=['HomeTeamName', 'AwayTeamName', 'Humidity', 'Temperature', 'WindSpeed'])
    input_data[['Humidity', 'Temperature', 'WindSpeed']] = scaler.transform(input_data[['Humidity', 'Temperature', 'WindSpeed']])

    # Prédiction
    predicted_score = model.predict(input_data)

    st.write(f"Score prédit pour {home_team} : {int(predicted_score[0][0])}")
    st.write(f"Score prédit pour {away_team} : {int(predicted_score[0][1])}")
