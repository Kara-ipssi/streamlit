from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datetime import datetime
from PIL import Image

# !pip install tensorflow
import tensorflow 
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras import optimizers

# Ttitre de l'application
st.title('Projet : Application de visualisation de données')

# Sidebar
sidebar = st.sidebar
sidebar.title('Sidebar')
sidebar.write('Welcome to the sidebar')
sidebar.markdown("[Introduction](#introduction)")
sidebar.markdown("[La problématique](#la-probl-matique)")
sidebar.markdown("[Le contexte](#le-contexte)")
sidebar.markdown("[Bas ⤵️](#bas)")

# Introduction
container = st.container()
container.title('Introduction')
container.write("L'intélligence artificielle")
container.write("Ce projet vise à créer une Proof concept d'une application intégrant de l'intelligence artificielle . En résolvant une problématique spécifique, nous souhaitons démontrer comment l'IA peut être utilisée pour apporter des solutions efficaces dans des domaines variés. En utilisant des données pertinentes et des algorithmes adaptés, notre objectif est de développer une application web conviviale qui permettra aux utilisateurs d'interagir avec le modèle d'IA et d'obtenir des prédictions précises. Ce projet offre une opportunité passionnante d'explorer le potentiel de l'IA et de mettre en pratique nos connaissances en développant une solution innovante. La présentation finale nous permettra de partager nos résultats, nos découvertes et les leçons apprises tout au long de ce projet captivant.")
# Graphique

# rand=np.random.normal(1, 2, size=20)
# fig, ax = plt.subplots()
# ax.hist(rand, bins=15)
# st.pyplot(fig)

# La problématique
container_2 = st.container()
container_2.write('## La problématique')
container_2.write('Comment l\'IA et le Big Data peuvent-elles améliorer la compréhension des comportements des consommateurs pour une stratégie marketing plus efficace ?')

# Le contexte
container_3 = st.container()
container_3.write('## Le contexte')
container_3.write('#### Motivation et Contexte de la problématique')
container_3.write("Avec l'évolution d’internet et les grandes avancées technologiques que nous avons connues au cours de la dernière décennie, le \n comportement  des consommateurs à radicalement été transformé. Ce changement dans les habitudes des consommateurs a également eu pour effet de totalement transformer le monde du marketing tel que nous le connaissions. Dans ce contexte, les consommateurs attendent des offres personnalisées et une expérience client de qualité, et ces attentes ne cessent de croître rapidement au cours des années.")
container_3.write("Dans cette optique d’optimisation et d’adaptation aux habitudes changeantes des consommateurs, le big data et l'intelligence artificielle jouent un rôle crucial en permettant aux entreprises d’avoir accès à de précieux  outils de récolte d’informations. Ils offrent aux entreprises des moyens modernes et efficaces de récolter, analyser et exploiter des données massives.")

# Les données
container_4 = st.container()    
container_4.write('## Les données')
container_4.write('soure : https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis')
container_4.write('#### Description des données')

# Ajout de l'image de description des données
image = Image.open('/app/assets/data.png')
st.image(image, caption='Description des données')

container_5 = st.container()
container_5.write('## Exploration, nétoyage et analyse des données')
container_5.write('### importation des données & Affichage des 5 premières lignes')
# Importation des données
df = pd.read_csv('/app/data/marketing_campaign.csv', sep=';')
container_5.write(df.head())

# Nétoyage des données
container_5.write('#### Nétoyage des données & renommage des colonnes')
# container_5.write('##### Suppression des valeurs manquantes')
# traitement des valeurs manquantes
df = df.dropna()

new_names = {'MntWines':'Wines','MntFruits':'Fruits', 'MntMeatProducts':'Meat', 
    'MntFishProducts':'Fish', 'MntSweetProducts':'Sweet_Products',
    'MntGoldProds':'Gold_Products', 'NumCatalogPurchases':'Catalog_Purchases', 'NumDealsPurchases': 'Deals_Purchases',}

df.rename(columns=new_names, inplace=True)

df['Age'] = datetime.now().year - df.Year_Birth
df["Children"]=df["Kidhome"]+df["Teenhome"]
new_df = df[['ID','Education','Marital_Status','Age','Income','Kidhome','Teenhome','Recency','Wines','Fruits','Meat','Fish','Sweet_Products','Gold_Products','Deals_Purchases','Catalog_Purchases', 'NumWebVisitsMonth', "NumWebPurchases", "NumStorePurchases"]]
container_5.write(new_df)

container_5.write("Standardisation les valeurs dans la colonne Marital_Status pour n'afficher que 'Célibataire', 'Marié', 'Divorcé', 'Veuve'")
container_5.write("Création d'une nouvelle colonne 'Autre'pour regrouper les colonnes 'Ensemble', 'Absurde' et 'YOLO'")


for n in new_df.index:
    if new_df.loc[n, 'Marital_Status'] == 'Alone':
        new_df.loc[n, 'Marital_Status'] = 'Single'
    elif new_df.loc[n, 'Marital_Status'] in ['Together','Absurd','YOLO']:
        new_df.loc[n, 'Marital_Status'] = 'Other'

container_5.write(new_df.Marital_Status.unique())

# container_5.write('#### Suppression des valeurs aberrantes')
container_5.write("Normalisation des valeurs dans la colonne Education pour n'afficher que 'Basic', 'Bachelor', 'Master' et 'PHD'")
for n in new_df.index:
    if new_df.loc[n,'Education'] == 'Graduation':
        new_df.loc[n, 'Education']= 'Bachelor'
    elif new_df.loc[n,'Education'] == '2n Cycle':
        new_df.loc[n, 'Education']= 'Master'
container_5.write(new_df.Education.unique())

container_5.write("Ajout d'une nouvelle colonne `Children` qui represente le nombre totale d'enfants + adolescents à la maison")
new_df = new_df.copy()
new_df['Children'] = new_df['Kidhome'] + new_df['Teenhome']
container_5.write(new_df.sample(10))


container_5.write("Création d'un nuage de points en utilisant seaborn pour vérifier la présence des valeurs improbable")
plt.figure(figsize=(16,8))
sns.scatterplot(new_df.Age)
plt.grid(True)
plt.xlabel('Nombre de clients')
plt.ylabel('Age Of Customers')
plt.title('Age Scatterplot')

container_5.pyplot(plt)


container_5.write("On observe la présence de valeurs aberrantes, on va donc les supprimer")
if any(new_df['Age'] > 100):
    new_df = new_df.drop(new_df[new_df['Age'] > 100].index)

plt.figure(figsize=(16,8))
sns.scatterplot(new_df.Age)
plt.grid(True)
plt.xlabel('Nombre de clients')
plt.ylabel('Age Of Customers')
plt.title('Age Scatterplot');

container_5.pyplot(plt)


# Création d'un nuage de points en utilisant seaborn pour vérifier la présence des valeurs improbable dans la colonne income
container_5.write("Création d'un nuage de points en utilisant seaborn pour vérifier la présence des valeurs improbable dans la colonne Income")
plt.figure(figsize=(16,8))
sns.scatterplot(new_df.Income)
plt.grid(True)
plt.xlabel('Nombre de clients')
plt.ylabel('Income client')
plt.title('Income Scatterplot');
# Affichage du graphique
container_5.pyplot(plt)


# Supression des valeurs abérrante dans la colonne "Income"
container_5.write("On observe la présence de valeurs aberrantes, on va donc les supprimer dans la colonne 'Income'")
if any(new_df['Income'] > 200000):
    new_df = new_df.drop(new_df[new_df['Income'] > 200000].index)

plt.figure(figsize=(16,8))
sns.scatterplot(new_df.Income)
plt.grid(True)
plt.xlabel('Income clients')
plt.ylabel('Age Of Customers')
plt.title('Age Scatterplot');
container_5.pyplot(plt)


# Créeation d'un graphique d'histogramme en utilisant la colonne 'Age' du DataFrame
container_5.write("Créeation d'un graphique d'histogramme en utilisant la colonne 'Age' du DataFrame")
plt.figure(figsize=(16, 8))
plt.hist(new_df.Age, color='r', alpha=0.5)
plt.xlabel('Age Bins')
plt.ylabel('Nombre de clients')
plt.title("Répartition par âge")

container_5.pyplot(plt)

# Ajout d'une colonne "spent" pour le totale de dépenses
container_5.write("Ajout d'une colonne 'spent' pour le totale de dépenses")
new_df["Spent"] = new_df["Wines"]+ new_df["Fruits"]+ new_df["Meat"]+ new_df["Fish"]+ new_df["Sweet_Products"]+ new_df["Gold_Products"]
container_5.write(new_df)


# Correlation, on remarque que les dépenses augmente en fonction des enfants et du nombre des articles
container_5.write("Correlation, on remarque que les dépenses augmente en fonction des enfants et du nombre des articles")
# Ajout de l'image de description des données
# image = Image.open('/app/assets/correlation.png')
# st.image(image, caption='Correlation', use_column_width=True)

# Création des groupes d'âge à l'aide de pd.cut()
groupes = pd.cut(new_df['Age'], [10, 20, 30, 40, 50, 60, 70, 80])

# Compter le nombre de valeurs dans chaque groupe d'âge
comptes_groupes = groupes.value_counts()


#Dépenses moyennes du client en produits par groupes d'âge
container_5.write("Dépenses moyennes du client en produits par groupes d'âge")
# image_2 = Image.open('app/assets/dp_m.png')
# st.image(image_2, caption="Dépenses moyennes du client en produits par groupes d'âge", use_column_width=True)


####
customer_class = ['Moyen', 'Moyen sans enfant', 'Riche','Pauvre']
####

# Save the trained model
modelFileName = '/app/model/marketing.h5'
model = Sequential()
print('model saved as', modelFileName)

# Load the saved model
model = models.load_model(modelFileName)



spent_kids_status = new_df[['Marital_Status','Children','Wines','Meat','Fish','Fruits','Sweet_Products','Gold_Products']]
spent_kids_status = pd.melt(spent_kids_status,
                            id_vars=['Marital_Status','Children'],
                            value_vars=['Wines','Meat','Fish','Fruits','Sweet_Products','Gold_Products'],
                            value_name="Amount",
                            var_name="Products"
                            )

status_grp = spent_kids_status.groupby('Marital_Status')
status_df = status_grp[['Amount','Children']].sum()
container_5.write(status_df)

#Montant totale de dépenses par status marital
container_5.write("Montant totale de dépenses par status marital")
fig, ax1 = plt.subplots(figsize=(16,8))
ax1.bar(status_df.index, status_df.Amount, color='#715a77',alpha=0.7)
ax1.set_ylabel('Total Amount Spent')
ax1.set_xlabel("Marital Status")
plt.title("Montant totale de dépenses par status marital")
container_5.pyplot(fig)

#Montant totale des dépenses par status marital par rapport au nombre d'enfants
container_5.write("Montant totale des dépenses par status marital par rapport au nombre d'enfants")
fig, ax1 = plt.subplots(figsize=(16,8))
ax1.bar(status_df.index, status_df.Amount, color='#715a77',alpha=0.7)
ax1.set_ylabel('Total Amount Spent')
ax2 = ax1.twinx()
ax2.plot(status_df.index, status_df.Children, color='r',marker='o',mfc='b',mec='black',linestyle='--')
ax2.set_ylabel("Nombre d'enfants")
plt.title("Montant totale des dépenses par status marital par rapport au nombre d'enfants ")
ax1.set_xlabel("Status marital")
ax1.legend(['Montant totale de dépenses'], loc='upper right',bbox_to_anchor=(1, 1), prop={'size': 12})
ax2.legend(["Dépenses Enfants"], loc='upper right',bbox_to_anchor=(1, 0.93), prop={'size': 12});
container_5.pyplot(fig)

# retirer la colonne "Education" car elle est corrélé avec la colonne "Income"
container_6 = st.container()
container_6.write('#### Test du modèle')
form = st.form("my_form")
income = form.slider(label='Income', min_value=0, max_value=200000, step=500, key=1)
children = form.slider(label='Number of children', min_value=0, max_value=20, step=1, key=2)
age = form.slider(label='Age', min_value=0, max_value=100, step=1, key=3)
wine = form.slider(label='Wines spend', min_value=0, max_value=20000, step=100, key=4)
fruit = form.slider(label='Fruits spend', min_value=0, max_value=20000, step=100, key=5)
meat = form.slider(label='Meat spend', min_value=0, max_value=20000, step=100, key=6)
fish = form.slider(label='Fish spend', min_value=0, max_value=20000, step=100, key=7)
sweet = form.slider(label='Sweet_Products spend', min_value=0, max_value=20000, step=100, key=8)
gold = form.slider(label='Gold_Products spend', min_value=0, max_value=20000, step=100, key=9)

avg_age = np.round(new_df.Age.mean(),1)

# container_5.write("la moyenne d'age est" + avg_age + "ans")

# Now add a submit button to the form:
submitted = form.form_submit_button("Submit")
spent  = wine + fruit + meat + fish + sweet + gold
if submitted:
    # st.write("income", income, "children", children, "Age", age, "spent", spent)
    # column = ['Income', 'Children', 'Spent', 'Age']
    # df = pd.DataFrame([[income, children, age, spent]], columns=column)
    with st.spinner('patientez pendant le traitement...'):
        time.sleep(2)
    st.success('Done!')
    x_new = np.array([[income,children,spent,age]])
    st.write('New sample: {}'.format(x_new))
    class_probabilities = model.predict(x_new)
    predictions = np.argmax(class_probabilities, axis=1)
    st.write("Prédiction du modèle")
    st.write(customer_class[predictions[0]])
    # st.write(df)
    # CReate a new array of features
   

    # Use the model to predict the class
    
    

    


# CReate a new array of features
# x_new = np.array([[60000,0,0,55]])
# print ('New sample: {}'.format(x_new))

# # Use the model to predict the class
# class_probabilities = model.predict(x_new)
# predictions = np.argmax(class_probabilities, axis=1)

# container_5.write("Prédiction du modèle")
# container_5.write(customer_class[predictions[0]])
st.write('## bas')