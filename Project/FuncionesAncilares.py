# Funciones Ancilares
"""
Funciones requeridas para la ejecución del proyecto de Mineria de Datos 2020-01
"""

import category_encoders as ce
import matplotlib.pyplot as plt
import missingno as msng
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

import pickle


def drop_columns(df, cols):
    """
    Elimina columnas  determinadas en el analisis.
    """
    
    df.drop(columns = cols, inplace = True)


def binary_var(df):
    """
    Corrige variables binarias y cambia 
    tipo de variables caracteristicas.
    """
    features = df.columns.drop('target')
    feat_bin = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']

    for col in features:
        if col in feat_bin:
            df[col] = df[col].replace(['N', 'F'],0).replace(['Y', 'T'],1).astype('object')
        else:
            df[col] = df[col].astype('object')


def plot_features(df, columns):
    """
    Grafica las variables exceptuando target
    """
    features = columns
    n = len(features)
    
    if n%3 == 0:
        m =  int(n/3)
    else:
        m = int(n//3 + 1)
    
    fig, axes = plt.subplots(nrows = m, ncols = 3, figsize = (18,20))
    fig.text(0.5, 0.92, "Frecuencia de datos", fontsize = 12, horizontalalignment = 'center')
    fig.subplots_adjust(hspace = 0.8) #espacio horizontal
    for i, col in enumerate(features):
        sns.countplot(df[col].sort_values(ascending = True), ax = axes[i//3,i%3])
        axes[i//3,i%3].set_title(col)


def null_replace(df):
    """
    Imputa valores nulos por la moda
    """
    for col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace = True)


def nominal_encoder(df):
    """
    recodifica las variables nominales cambiando sus clases a numeros 
    """
    df['nom_1'] = np.where((df['nom_1'] == 'Square') |
                              (df['nom_1'] == 'Star'), 'Others', df['nom_1'])

    df['nom_2'] = np.where((df['nom_2'] == 'Cat') |
                              (df['nom_2'] == 'Snake'), 'Others', df['nom_2'])

    df['nom_3'] = np.where((df['nom_3'] == 'Canada') |
                              (df['nom_3'] == 'China'), 'Others', df['nom_3'])

    df['nom_4'] = np.where((df['nom_4'] == 'Oboe') |
                              (df['nom_4'] == 'Piano'), 'Others', df['nom_4'])
    
    # Se transforma cada clase en una etiqueta numerica
    encoder_nom = ce.OrdinalEncoder(cols = ['nom_0','nom_1','nom_2','nom_3','nom_4'])
    df = encoder_nom.fit_transform(df)
    return df        


def ordinal_encoder(df):
    """
    modifica las variables ordinales dependiendo de su orden de 1 a n.
    """
    ord_1 = {'Novice':1, 
            'Contributor':2, 
            'Expert':3, 
            'Master':4, 
            'Grandmaster':5}
    df['ord_1'] = df.ord_1.map(ord_1)
    
    ord_2 = {'Freezing':1, 
            'Cold':2, 
            'Warm':3, 
            'Hot':4, 
            'Boiling Hot':5, 
            'Lava Hot':6}
    df['ord_2'] = df.ord_2.map(ord_2)
    
    ord_3 = {'a':1,
             'b':2, 
             'c':3, 
             'd':4, 
             'e':5,
             'f':6,
             'g':7,
             'h':8,
             'i':9,
             'j':10,
             'k':11,
             'l':12,
             'm':13,
             'n':14,
             'o':15}
    df['ord_3'] = df.ord_3.map(ord_3)
    
    ord_4 = {"A":1,
             "B":2,
             "C":3,
             "D":4,
             "E":5,
             "F":6,
             "G":7,
             "H":8,
             "I":9,
             "J":10,
             "K":11,
             "L":12,
             "M":13,
             "N":14,
             "O":15,
             "P":16,
             "Q":17,
             "R":18,
             "S":19,
             "T":20,
             "U":21,
             "V":22,
             "W":23,
             "X":24,
             "Y":25,
             "Z":26}
    df['ord_4'] = df.ord_4.map(ord_4)
    
    return df


def subsampling(df):
    """
    hace un remuestreo de los datos de tipo undersampling
    """

    df = df.sample(frac=1)

    target_1 = df.loc[df['target'] == 1]
    target_0 = df.loc[df['target'] == 0][:len(target_1)]

    sub_sample = pd.concat([target_1,target_0])
    sub_sample = sub_sample.sample(frac=1, random_state = 0)

    return sub_sample


def data_processing(df):
    """
    Procesa la data de test bajo los parametros de train
    """
    if 'target' in  df.columns:
        pass
    else:
        df['target'] = ''
    
    cols_drop = ['nom_5','nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_5']
    cols_dummies = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4']
    
    drop_columns(df, cols_drop)
    null_replace(df)
    binary_var(df)
    df = nominal_encoder(df)
    df = ordinal_encoder(df)
    df = cols_dummies_transform(df, cols_dummies)
    
    df = df.drop(columns = ['target'])
    
    return df


def cols_dummies_transform(df, cols_dummies):
    """
    Aplica transformacion binaria a variables de entrada
    :param df: dataframe to process
    :param cols_dummies: cols from df to dommies apply
    :return: df
    """
    for col in cols_dummies:
        df = pd.concat([df,pd.get_dummies(df[col],prefix = col,dummy_na = True)],
                       axis = 1).drop([col], axis = 1)
    return df


def training_models_cv(X_train, y_train):
    """
    Entrena modelos
    """
    # Parametros de los modelos a entrenar
    logistic_params = {'C' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                       'penalty' : ['l1', 'l2'],
                       'max_iter' : [1000]}

    #svc_params = {'C':[0.001,0.1,10,100,10e5],'gamma':[0.1,0.01]}

    tree_params = {'criterion':['gini','entropy'],
                   'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}

    gradient_params = {'learning_rate': [0.01, 0.1,0.5],
                       'n_estimators': [50, 100, 500, 1000, 2000],
                       'subsample': [0.1,0.5,0.9]}

    # Lista de modelos a entrenar
    models = []
    models.append(("LogisticRegression",LogisticRegression(random_state = 16626),logistic_params))
    #models.append(("SVC",SVC(random_state = 16626),svc_params))
    models.append(("DecisionTree",DecisionTreeClassifier(random_state = 16626),tree_params))
    models.append(("GradientBoosting",GradientBoostingClassifier(random_state = 16626),gradient_params))

    names = []
    best_params = []
    best_score = []

    for name, model, params in models:
        result = GridSearchCV(estimator = model,
                              param_grid = params,
                              cv = 3,
                              n_jobs = -1).fit(X_train, y_train)

        names.append(name)
        best_params.append(result.best_params_)
        best_score.append(result.best_score_)

    # Se crean instancias con los mejores modelos entrenados
    logit = LogisticRegression(C = best_params[0]['C'],
                               penalty = best_params[0]['penalty'],
                               max_iter = best_params[0]['max_iter'],
                               random_state = 16626).fit(X_train, y_train)

    #svc = SVC(C = best_params[1]['C'],gamma = best_params[1]['gamma'],random_state=16626).fit(X_train, y_train)

    dec_tree = DecisionTreeClassifier(criterion = best_params[1]['criterion'],
                                      max_depth = best_params[1]['max_depth'],
                                      random_state = 16626).fit(X_train, y_train)

    gradient = GradientBoostingClassifier(learning_rate = best_params[2]['learning_rate'],
                                          n_estimators = best_params[2]['n_estimators'],
                                          subsample = best_params[2]['subsample'],
                                          random_state = 16626).fit(X_train, y_train)

    # Se guardan los modelos ajustados
    pickle.dump(logit, open('logit_model.sav', 'wb'))
    #pickle.dump(svc, open('svc_model.sav', 'wb'))
    pickle.dump(dec_tree, open('dec_tree_model.sav', 'wb'))
    pickle.dump(gradient, open('gradient_model.sav', 'wb'))

    for i in range(len(names)):
        print(names[i], best_params[i], best_score[i])

    return names, best_params, best_score


def confusion_to_df(mc):
    """
    :param mc: Matriz de Confusion
    :return: Matriz de Confusion como df
    """
    mc = mc.T
    df_mc = pd.concat([pd.Series(mc[0]), pd.Series(mc[1])], axis=1)
    df_mc.columns = ['Verdadero', 'Falso']
    df_mc.index = ['Verdadero', 'Falso']
    return df_mc


def metrics_model(ruta, X_train, y_train, X_test):
    """
    :param ruta: ruta del modelo exportado .sav
    :param X_train:  matriz de caracteristicas del modelo entrenado
    :param y_train: vector objetivo
    :param X_test: matriz de datos de test
    :return: auc y metricas del modelo
    """
    # Fit Model
    model = pickle.load(open(ruta, 'rb'))
    pred = model.predict_proba(X_train)[:, 1]
    y_hat = model.predict(X_test)
    y_pred = np.argmax(model.predict_proba(X_train), axis=1)

    print('----------------------------------------------------------------------------')
    print('Metrics Report Model ' + model.__class__.__name__)
    print('----------------------------------------------------------------------------')

    # AUC
    auc_model = roc_auc_score(y_train, pred)
    print("AUC: ", auc_model)

    # Calcula matriz de confusion
    conf = confusion_to_df(confusion_matrix(y_train, y_pred))
    print("MATRIZ DE CONFUSION")
    display(conf)

    # Metrics Resume
    metric = pd.DataFrame(classification_report(y_train, y_pred, output_dict = True))
    display(metric)

    # grafica curva roc
    fpr, tpr, thresholds = roc_curve(y_train, pred)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='tomato')
    plt.plot(fpr, tpr, marker='.', lw=2)
    plt.title("ROC curve model " + model.__class__.__name__)
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.show();

    return model, y_pred, auc_model, metric


def metrics_comparative(model_list, auc_list, metrics_list, y_train, y_pred_list):
    # Grafica las metricas
    metrics = {}

    for index, model in enumerate(model_list):
        metrics.update({'Model_'+ model.__class__.__name__: [auc_list[index],
                                                            accuracy_score(y_train, y_pred_list[index]).round(5),
                                                            metrics_list[index]['weighted avg'].iloc[0].round(5),
                                                            metrics_list[index]['weighted avg'].iloc[1].round(5),
                                                            metrics_list[index]['weighted avg'].iloc[2].round(5)]})

    print("Metrics Comparation")
    errors_df = pd.DataFrame(metrics, index=['AUC', 'Accuracy', 'Presicion', 'Recall', 'F1'])
    display(errors_df.head())
    display(errors_df.plot(title='Models Metrics for Class Models', kind='barh',
                           figsize=(8, 6)).legend(bbox_to_anchor=(1.0, 0.5)));