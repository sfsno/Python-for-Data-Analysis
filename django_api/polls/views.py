#import des libraires
from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
from django.core.cache import cache

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.io as io
import plotly.graph_objs as go
import plotly.express as px
io.renderers.default='browser'

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,auc

from collections import Counter
import json

from io import BytesIO
import base64
import matplotlib
import mpld3
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def importdata():
    filepath = r"cleaned_data.csv"
    # Vérifier si les données sont déjà en cache
    cached_data = cache.get('drug')
    if cached_data is not None:
        return cached_data
    # Si les données ne sont pas en cache, les importer
    data = pd.read_csv(filepath, sep = ",")
    data.columns= ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Canabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamin', 'Legalh', 'LSD', 'Meth','Mushrooms', 'Nicotine', 'Semer', 'VSA']
    # Mettre les données en cache
    cache.set('drug', data, timeout=None)
    return data

def importinitialdata():
    filepath = r"drug_consumption.data"
    # Vérifier si les données sont déjà en cache
    cached_data = cache.get('initialdrug')
    if cached_data is not None:
        return cached_data
    # Si les données ne sont pas en cache, les importer
    data = pd.read_csv(filepath, sep=",",encoding='utf-8',float_precision='round_trip', header=None)
    data.columns= ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Canabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamin', 'Legalh', 'LSD', 'Meth','Mushrooms', 'Nicotine', 'Semer', 'VSA']
    # Mettre les données en cache
    cache.set('initialdrug', data, timeout=None)
    return data

def datauser():
    df_users = importinitialdata()
    for i in range (0, 32):
        df_users.iloc[:, i]=df_users.iloc[:, i].replace(['CL0'], 0)
        df_users.iloc[:, i]=df_users.iloc[:, i].replace(['CL1'], 0) 
        df_users.iloc[:, i]=df_users.iloc[:, i].replace(['CL2'], 1)
        df_users.iloc[:, i]=df_users.iloc[:, i].replace(['CL3'], 1)
        df_users.iloc[:, i]=df_users.iloc[:, i].replace(['CL4'], 1)
        df_users.iloc[:, i]=df_users.iloc[:, i].replace(['CL4'], 1)
        df_users.iloc[:, i]=df_users.iloc[:, i].replace(['CL5'], 1)
        df_users.iloc[:, i]=df_users.iloc[:, i].replace(['CL6'], 1)
    return df_users

def par_age():
    df = importdata();

    drogues = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Canabis', 'Choc', 'Coke', 'Crack',
               'Ecstasy', 'Heroin', 'Ketamin', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    
    age_order = ['18-24', '25-34', '35-40','45-54', '55-64','65+']
    df['Age'] = pd.Categorical(df['Age'], categories=age_order, ordered=True)
    
    consommation_order = ['Never', 'Decade Ago', 'Last Decade', 'Last Year', 'Last Month', 'Last Week', 'Last Day']
    
    fig, axs = plt.subplots(5, 4, figsize=(17, 17))
    colors = sns.color_palette('husl', n_colors=len(consommation_order))
    
    plt.subplots_adjust(hspace=0.5)
    
    for i, drogue in enumerate(drogues):
        hue_order = pd.Categorical(df[drogue], categories=consommation_order, ordered=True)
        
        sns.countplot(x='Age', hue=hue_order, data=df, order=age_order, palette=colors, ax=axs[i // 4, i % 4])
        
        axs[i // 4, i % 4].set_title(f'Consumption of {drogue} per age')
        axs[i // 4, i % 4].set_xlabel('Age')
        axs[i // 4, i % 4].set_ylabel('Number of persons')
        axs[i // 4, i % 4].legend(title=drogue, loc='upper right', fontsize='small')
    
    if len(drogues) % 2 == 1:
        axs[-1, -1].axis('off')
        
    #plt.xticks(range(len(age_order)), age_order) 
         
    #plot_html = mpld3.fig_to_html(fig)
    #return plot_html
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    image_data = base64.b64encode(image_stream.read()).decode('utf-8')
    
    plt.close()
    
    return f'<img src="data:image/png;base64,{image_data}" />'

def par_genre():
    df = importdata();
    
    drogues = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Canabis', 'Choc', 'Coke', 'Crack',
               'Ecstasy', 'Heroin', 'Ketamin', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    
    age_order = ['18-24', '25-34', '35-40','45-54', '55-64','65+']
    df['Age'] = pd.Categorical(df['Age'], categories=age_order, ordered=True)
    
    consommation_order = ['Never', 'Decade Ago', 'Last Decade', 'Last Year', 'Last Month', 'Last Week', 'Last Day']
    
    fig, axs = plt.subplots(5, 4, figsize=(17, 17))
    colors = sns.color_palette('husl', n_colors=len(consommation_order))
    plt.subplots_adjust(hspace=0.5)
    
    for i, drogue in enumerate(drogues):
        hue_order = pd.Categorical(df[drogue], categories=consommation_order, ordered=True)
        
        sns.countplot(x='Gender', hue=hue_order, data=df, palette=colors, ax=axs[i // 4, i % 4])
        
        axs[i // 4, i % 4].set_title(f'Consumption of {drogue} by gender')
        axs[i // 4, i % 4].set_xlabel('Gender')
        axs[i // 4, i % 4].set_ylabel('Number of persons')
        axs[i // 4, i % 4].legend(title=drogue, loc='upper right', fontsize='small')
        
    if len(drogues) % 2 == 1:
        axs[-1, -1].axis('off')
    
   #plot_html = mpld3.fig_to_html(fig)
   # return plot_html
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    image_data = base64.b64encode(image_stream.read()).decode('utf-8')
    
    plt.close()
    
    return f'<img src="data:image/png;base64,{image_data}" />'
   
    

def par_pays():
    df = importdata();
    
    drogues = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Canabis', 'Choc', 'Coke', 'Crack',
               'Ecstasy', 'Heroin', 'Ketamin', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    
    consommation_order = ['Never', 'Decade Ago', 'Last Decade', 'Last Year', 'Last Month', 'Last Week', 'Last Day']
    
    fig, axs = plt.subplots(5, 4, figsize=(17, 17))
    colors = sns.color_palette('husl', n_colors=len(consommation_order))
    plt.subplots_adjust(hspace=0.8)
    
    for i, drogue in enumerate(drogues):
        hue_order = pd.Categorical(df[drogue], categories=consommation_order, ordered=True)
        
        sns.countplot(x='Country', hue=hue_order, data=df, palette=colors, ax=axs[i // 4, i % 4])
        
        axs[i // 4, i % 4].set_title(f'Consumption of {drogue} per country')
        axs[i // 4, i % 4].set_xlabel('Country')
        axs[i // 4, i % 4].set_ylabel('Number of persons')
        axs[i // 4, i % 4].legend(title=drogue, loc='upper right', fontsize='small')
        axs[i // 4, i % 4].set_xticklabels(axs[i // 4, i % 4].get_xticklabels(), rotation=40, ha='right')
        
    if len(drogues) % 2 == 1:
        axs[-1, -1].axis('off')
    
    #plot_html = mpld3.fig_to_html(fig)
    #return plot_html
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    image_data = base64.b64encode(image_stream.read()).decode('utf-8')
    
    plt.close()
    
    return f'<img src="data:image/png;base64,{image_data}" />'


def par_education():
    df = importdata();
    
    drogues = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Canabis', 'Choc', 'Coke', 'Crack',
               'Ecstasy', 'Heroin', 'Ketamin', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    
    consommation_order = ['Never', 'Decade Ago', 'Last Decade', 'Last Year', 'Last Month', 'Last Week', 'Last Day']
    
    education_order=["Left school before 16 years", "Left school at 16 years","Left school at 17 years", 
                            "Left school at 18 years","Some college or university, no certificate or degree",
                            "Professional certificate/diploma","University degree","Masters degree","Doctorate degree"]
    
    fig, axs = plt.subplots(5, 4, figsize=(17, 17))
    colors = sns.color_palette('husl', n_colors=len(consommation_order))
    plt.subplots_adjust(hspace=0.8)
    
    for i, drogue in enumerate(drogues):
        hue_order = pd.Categorical(df[drogue], categories=consommation_order, ordered=True)
        
        sns.countplot(x='Education', hue=hue_order, data=df, order=education_order, palette=colors, ax=axs[i // 4, i % 4])
        
        axs[i // 4, i % 4].set_title(f'Consumption of {drogue} per education')
        axs[i // 4, i % 4].set_xlabel('Education')
        axs[i // 4, i % 4].set_ylabel('Number of persons')
        axs[i // 4, i % 4].legend(title=drogue, loc='upper right', fontsize='small')
        
        axs[i // 4, i % 4].set_xticklabels(axs[i // 4, i % 4].get_xticklabels(), rotation=20, ha='right')
    
    if len(drogues) % 2 == 1:
        axs[-1, -1].axis('off')
    
    #plot_html = mpld3.fig_to_html(fig)
    #return plot_html
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    image_data = base64.b64encode(image_stream.read()).decode('utf-8')
    
    plt.close()
    
    return f'<img src="data:image/png;base64,{image_data}" />'

def par_ethnicite():
    df = importdata();
    
    drogues = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Canabis', 'Choc', 'Coke', 'Crack',
               'Ecstasy', 'Heroin', 'Ketamin', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    
    consommation_order = ['Never', 'Decade Ago', 'Last Decade', 'Last Year', 'Last Month', 'Last Week', 'Last Day']
    
    fig, axs = plt.subplots(5, 4, figsize=(17, 17))
    colors = sns.color_palette('husl', n_colors=len(consommation_order))
    plt.subplots_adjust(hspace=0.8)
    
    for i, drogue in enumerate(drogues):
        hue_order = pd.Categorical(df[drogue], categories=consommation_order, ordered=True)
        
        sns.countplot(x='Ethnicity', hue=hue_order, data=df, palette=colors, ax=axs[i // 4, i % 4])
        
        axs[i // 4, i % 4].set_title(f'Consumption of {drogue} per ethnicity')
        axs[i // 4, i % 4].set_xlabel('Ethnicity')
        axs[i // 4, i % 4].set_ylabel('Number of persons')
        axs[i // 4, i % 4].legend(title=drogue, loc='upper right', fontsize='small')
        axs[i // 4, i % 4].set_xticklabels(axs[i // 4, i % 4].get_xticklabels(), rotation=20, ha='right')
        
    if len(drogues) % 2 == 1:
        axs[-1, -1].axis('off')
    
    #plot_html = mpld3.fig_to_html(fig)
    #return plot_html
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    image_data = base64.b64encode(image_stream.read()).decode('utf-8')
    
    plt.close()
    
    return f'<img src="data:image/png;base64,{image_data}" />'


def corr_matrix():
    df_users = datauser()
    drug_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Canabis', 'Choc', 'Coke', 'Crack','Ecstasy', 'Heroin', 'Ketamin', 'Legalh', 'LSD', 'Meth', 'Mushrooms','Nicotine', 'Semer', 'VSA']
    df_users["User"] = df_users[drug_columns].sum(axis=1)
    df_users.head()
    print(df_users.columns)
    corr = df_users.corr()
    fig = plt.figure(figsize=(15,12))
    mask = np.triu(corr)
    sns.heatmap (corr,annot=True,lw=2,cmap='coolwarm', mask = mask)
    #plot_html = mpld3.fig_to_html(fig)
    #return plot_html
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    image_data = base64.b64encode(image_stream.read()).decode('utf-8')
    
    plt.close()
    
    return f'<img src="data:image/png;base64,{image_data}" />'


def corr_matrix_personnality():
    df_users = datauser()
    personality_columns = ['Nscore','Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']
    drug_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Canabis', 'Choc', 'Coke', 'Crack','Ecstasy', 'Heroin', 'Ketamin', 'Legalh', 'LSD', 'Meth', 'Mushrooms','Nicotine', 'Semer', 'VSA']
    
    selected_columns = personality_columns + drug_columns
    selected_data = df_users[selected_columns]
    
    correlation_matrix = selected_data.corr()
    
    fig, ax = plt.subplots(figsize=(18, 12))
    mask = np.triu(correlation_matrix)
    sns.heatmap(correlation_matrix, cmap='flare', fmt=".2f", linewidths=.5, mask = mask)
       
    #ax.set_xticklabels(np.arange(2, 28), rotation=45, ha='right')
    #ax.set_yticklabels(selected_columns, ha='right')
    #ax.set_xticks(range(len(selected_columns)))
    #ax.set_yticks(range(len(selected_columns)))
    #plt.xticks(selected_columns) 
    
    plt.title('Correlation Matrix between personnality types and drugs')
    
    #plot_html = mpld3.fig_to_html(fig)
    #return plot_html
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    image_data = base64.b64encode(image_stream.read()).decode('utf-8')
    
    plt.close()
    
    return f'<img src="data:image/png;base64,{image_data}" />'


def violin_personnality(user_input=None):
    if user_input==None:
        return "<p> </p>"
    df = importdata();    
    

    selected_columns = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'SS', 'Impulsive', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Canabis', 'Choc', 'Coke', 'Crack',
               'Ecstasy', 'Heroin', 'Ketamin', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    df_selected = df[selected_columns]
    """
    df_melted = pd.melt(df_selected, id_vars=['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'SS', 'Impulsive'], var_name='Drogue', value_name='Consommation')    

    fig = plt.figure(figsize=(12, 4))
    ax = sns.violinplot(x='Drogue', y=user_input, hue='Consommation', data=df_melted, inner='quart', palette='muted')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title(f'Distribution of {user_input} based on drug')
    plt.xlabel('Drug')
    plt.ylabel('Personality score')
    ax.legend(loc='upper right', fontsize='small')
    plt.show()
    
    plt.tight_layout()
        """
        
    fig = plt.figure(figsize=(10, 15))

    personality_types = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'SS', 'Impulsive'] 
    
    for i, personality_type in enumerate(personality_types, 1):
        plt.subplot(7, 1, i)
        ax = sns.violinplot(x=user_input, y=personality_type, hue=df_selected[user_input], data=df_selected, inner='quart', palette='muted')
        plt.title(f'{personality_type} based on {user_input} consumption')
        plt.xlabel(f'{user_input} Consumption')
        plt.ylabel('Personality score')
        ax.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    
    #plot_html = mpld3.fig_to_html(fig)
    #return plot_html
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    image_data = base64.b64encode(image_stream.read()).decode('utf-8')
    
    plt.close()
    
    return f'<img src="data:image/png;base64,{image_data}" />'

def ROC():
    df_users1 = datauser()
    list_auc = []
    drugs_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Canabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamin', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    
    fig, axes = plt.subplots(5, 4, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, drug in enumerate(drugs_columns):
        df_users1[drug] = df_users1[drug].astype(int)
        X = df_users1[['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore']]
        y = df_users1[drug]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    
        model = LogisticRegression()
        model.fit(X_train, y_train)
    
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
    
        list_auc.append(auc_score)
    
        axes[i].plot(fpr, tpr, lw=2, label='ROC curve of ' + str(drug) + ' (AUC = %0.2f)' % auc_score)
        axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[i].set_title('ROC Curve for ' + str(drug))
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].legend(loc='lower right')
    
    axes[-1].axis('off')
        
    plt.tight_layout()

    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    image_data = base64.b64encode(image_stream.read()).decode('utf-8')
    
    plt.close()
    
    return f'<img src="data:image/png;base64,{image_data}" />'

def ROC_compl():
    df_users1 = datauser()
    list_auc = []
    drugs_columns = ['Alcohol', 'Amphet', 'Amyl','Benzos', 'Caff', 'Canabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamin', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    
    plt.figure(figsize=(8, 6))
    
    for drug in drugs_columns:
        df_users1[drug] = df_users1[drug].astype(int)
        X = df_users1[['Nscore','Escore','Oscore','Ascore','Cscore']]
        y = df_users1[drug]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
        
        model = LogisticRegression()  # Initialisation du modèle
        model.fit(X_train, y_train)  # Entraînement du modèle
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilité pour la classe positive
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)  # Calcul des FPR et TPR
        auc_score = auc(fpr, tpr)  # Calcul de l'AUC
        
        list_auc.append(auc_score)  # Stockage de l'AUC
        
        # Affichage de la courbe ROC correspondant à chaque modèle
        plt.plot(fpr, tpr, lw=2, label='ROC curve of ' + str(drug) + ' (AUC = %0.2f)' % auc_score)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Ligne de référence aléatoire
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves of Different Drugs')
    plt.legend(loc='lower right')

    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    image_data = base64.b64encode(image_stream.read()).decode('utf-8')
    
    plt.close()
    
    return f'<img src="data:image/png;base64,{image_data}" />'

    
def index0(request):
    df = importdata(); 
    df_classes = df.columns   
    context = {
       'les_labels_du_df': df_classes,
    }
    return render(request, "template1.html", context)


def index1(request):
    template = loader.get_template("template1.html")
    
    criteria = request.GET.get('criteria', None)
    if criteria == "par_age":
        plot_html = par_age()
    elif criteria == "par_genre":
        plot_html = par_genre()
    elif criteria == "par_education":
        plot_html = par_education()
    elif criteria == "par_pays":
        plot_html = par_pays()
    elif criteria == "par_ethnicite":
        plot_html = par_ethnicite()
    else:
        plot_html = "<p> </p>"
    
    plot_html_violon = violin_personnality()
    context = {
       'plot_html' : plot_html,
       
    }
    return HttpResponse(template.render(context, request))


def index2(request):

    template = loader.get_template("template1.html")
    personnality = request.GET.get('personnality', None)
    plot_html_violon = violin_personnality(personnality)
    context = {
       'plot_html_violon' : plot_html_violon,
       
    }
    return HttpResponse(template.render(context, request))


def index3(request):
    template = loader.get_template("template1.html")
    matrice = request.GET.get('matrice', None)

    if matrice == "personnalite_drogue":
        plot_html_corr = corr_matrix_personnality()
    elif matrice == "complete":
        plot_html_corr = corr_matrix()
    context = {
       'plot_html_corr' : plot_html_corr,
       
    }
    return HttpResponse(template.render(context, request))


def index4(request):
    template = loader.get_template("template1.html")
    model = request.GET.get('model', None)
    plot_html_ROC = None
    if model == "complete":
        plot_html_ROC = ROC_compl()
    elif model == "subplots":
        plot_html_ROC = ROC()
    if plot_html_ROC is None:
       plot_html_ROC = "Default value or empty string or any appropriate default"
        
    context = {
       'plot_html_ROC': plot_html_ROC,     
    }
    return HttpResponse(template.render(context, request))







