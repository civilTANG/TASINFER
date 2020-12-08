# -*- coding: utf-8 -*-
###############################################################################################################
#####################################  Benoetigte Bibliotheken importieren: ###################################
############################################################################################################### 
import pandas as pd ## Verarbeitung von Datentabellen (etwa analog Excel-Tabellen) moeglich
import numpy as np ## mehrdim Arrays + weitere mathematische Funktionalitaeten
import matplotlib.pyplot as plt ## fuers Grafiken erstellen, plotten etc. 
import math ## mathematische funktionen, u.a. sqrt()
from sklearn.metrics import roc_curve, auc, classification_report,confusion_matrix ## fuer die Ergebnisauswertungen (roc/auc)
from sklearn.model_selection import train_test_split ## um Datenmenge in Test- und Trainingsdaten zu splitten
from sklearn.preprocessing import LabelEncoder ## Kategorische Features in Zahlen ueberfuehren
from sklearn.model_selection import cross_val_score   #For cross validation - prueft ob Modell gegenueber Auswahl Testmenge stabil
from matplotlib.colors import ListedColormap ## Farbe fuer Plots importieren
cm3 = ListedColormap(['#0000aa', '#ff2020', '#50ff50'])
import seaborn as sns

############################################################################################################### 
##################################### Daten einlesen: #########################################################
############################################################################################################### 
## Daten aus csv als DataFrame einlesen:
df_test_challenge = pd.read_csv('../input/TestData.csv',sep=';', encoding='cp1252')
df_raw = pd.read_csv('../input/TrainData.csv',sep=';', encoding='cp1252')

############################################################################################################### 
##################################### Explorative Datenanalyse und Datenauswahl: ##############################
############################################################################################################### 
#Erste Grafik zur Pruefung auf Korrelationen zwischen numerischen Variablen 
plt.figure()
sns.heatmap(df_raw.corr())
# Grafiken fuer kategorische Daten: 
plt.figure()
sns.countplot(x='Monat', hue='Zielvariable', data=df_raw);
plt.figure()
sns.countplot(x='Geschlecht', hue='Zielvariable', data=df_raw);
plt.figure()
sns.countplot(x='Art der Anstellung', hue='Zielvariable', data=df_raw);
plt.figure()
sns.countplot(x='Familienstand', hue='Zielvariable', data=df_raw);
plt.figure()
sns.countplot(x=u'Schulabschluß', hue='Zielvariable', data=df_raw);
plt.figure()
sns.countplot(x='Kontaktart', hue='Zielvariable', data=df_raw);
plt.figure()
sns.countplot(x='Ergebnis letzte Kampagne', hue='Zielvariable', data=df_raw);

## Aenderung auf anderes Format
##df_raw['Tage seit letzter Kampagne'] = df_raw['Tage seit letzter Kampagne'].astype('int64')
#df_raw['Dauer'] = df_raw['Dauer'].astype('float64')
#df_raw['Alter'] = df_raw['Alter'].astype('float64')
#df_raw['Kontostand'] = df_raw['Kontostand'].astype('float64')

## Duplikate entfernen
## Keine vorhanden, kann aber trotzdem nicht schaden :) 
df_raw = df_raw.drop_duplicates(keep='first')

# Fehlende Werte?
#print('Fehlende Werte in Trainingsmenge:\n',df_raw.isnull().sum())
#print('Fehlende Werte in Testmenge:\n',df_test_challenge.isnull().sum())
# Nur Spalte 'Tage seit letzter Kampagne' betroffen (und natuerlich Zielvariable in Testmenge)
## NA Werte in Spalte 'Tage seit letzter Kampagne' mit hohem Wert auffuellen 
#(entspricht am ehesten einem 'Kein Kontakt in letzter Kampagne')
df_raw['Tage seit letzter Kampagne'] = df_raw['Tage seit letzter Kampagne'].fillna(10000)
df_test_challenge['Tage seit letzter Kampagne'] = df_test_challenge['Tage seit letzter Kampagne'].fillna(10000)

## Nicht relevante Spalten entfernen: 
## 'Anruf-ID': rein technische ID, macht weder methodisch noch fachlich Sinn
df_raw = df_raw.drop('Anruf-ID', 1)
df_test_challenge = df_test_challenge.drop('Anruf-ID', 1)

## Datentypen anpassen:
# benoetigte Dictionaries definieren: 
int_dic = {'ja': 1, 'nein': 0}
gender_dic = {'w': 1, 'm': 0}
campaign_dic = {'Erfolg': 1, 'Unbekannt': 0, 'Kein Erfolg': 0, 'Sonstiges':0}
contactform_dic = {'Handy': 1, 'Festnetz': 0, 'Unbekannt': 0}
# Laufende Nummer des Monats
#month_lookup = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
# Anzahl der Tage im Jahr bis 1. des betrachteten Monats (um Tag des Jahres zu ermitteln):
month_lookup = {'jan': 0, 'feb': 31, 'mar': 59, 'apr': 90, 'may': 120, 'jun': 151, 'jul': 181, 'aug': 212, 'sep': 243, 'oct': 273, 'nov': 304, 'dec': 334}
season_lookup = {'jan': 'Winter', 'feb': 'Winter', 'mar': 'Winter', 'apr': 'Spring', 'may': 'Spring', 'jun': 'Spring', 'jul': 'Summer', 'aug': 'Summer', 'sep': 'Summer', 'oct': 'Fall', 'nov': 'Fall', 'dec': 'Fall'}

# Binaere (Text-)Spalten in numerische Ueberfuehren
df_raw['Haus'] = df_raw['Haus'].map(int_dic)
df_raw['Ausfall Kredit'] = df_raw['Ausfall Kredit'].map(int_dic)
df_raw['Kredit'] = df_raw['Kredit'].map(int_dic)
df_raw['Geschlecht'] = df_raw['Geschlecht'].map(gender_dic)
df_raw['Zielvariable'] = df_raw['Zielvariable'].map(int_dic)
# Monat entsprechendem dem obigen Dictionary um-mappen
df_raw['Season'] = df_raw['Monat'].map(season_lookup)
df_raw['Monat'] = df_raw['Monat'].map(month_lookup)
df_raw['MonthFirstHalf'] = np.where(df_raw['Tag'] > 15, 0, 1)

df_test_challenge['Haus'] = df_test_challenge['Haus'].map(int_dic)
df_test_challenge['Ausfall Kredit'] = df_test_challenge['Ausfall Kredit'].map(int_dic)
df_test_challenge['Kredit'] = df_test_challenge['Kredit'].map(int_dic)
df_test_challenge['Geschlecht'] = df_test_challenge['Geschlecht'].map(gender_dic)
df_test_challenge['Zielvariable'] = df_test_challenge['Zielvariable'].map(int_dic)
# Monat entsprechendem dem obigen Dictionary um-mappen
df_test_challenge['Monat'] = df_test_challenge['Monat'].map(month_lookup)

## Logarithmus der schiefen Verteilung: 
df_raw['Log_Dauer']=np.log(1+df_raw['Dauer'])

## Outlier? Anzahl Kontakte letzte Kamp > 100, Dauer > 4000
print(df_raw.sort_values(by=['Anzahl Kontakte letzte Kampagne'], ascending=[False])['Anzahl Kontakte letzte Kampagne'].head(10))
print(df_raw.sort_values(by=['Dauer'], ascending=[False])['Dauer'].head(10))
## Max Werte ersetzen
median_numberOfContacts = df_raw.loc[df_raw['Anzahl Kontakte letzte Kampagne']<100, 'Anzahl Kontakte letzte Kampagne'].median()
df_raw['Anzahl Kontakte letzte Kampagne'] = np.where(df_raw['Anzahl Kontakte letzte Kampagne'] >100, median_numberOfContacts, df_raw['Anzahl Kontakte letzte Kampagne'])
median_duration = df_raw.loc[df_raw['Dauer']<4000, 'Dauer'].median()
df_raw['Dauer'] = np.where(df_raw['Dauer'] >4000, median_duration, df_raw['Dauer'])
## Check:
print(df_raw.sort_values(by=['Anzahl Kontakte letzte Kampagne'], ascending=[False])['Anzahl Kontakte letzte Kampagne'].head(10))
print(df_raw.sort_values(by=['Dauer'], ascending=[False])['Dauer'].head(10))

## Bevor weitere Datentransformationen stattfinden, erstmal erste Histogramme: 
## kategorische Daten in numerische Transformieren
df_num = df_raw.copy()
columns_mod = ['Art der Anstellung','Familienstand',u'Schulabschluß','Kontaktart','Ergebnis letzte Kampagne','Season']
le = LabelEncoder()
for i in columns_mod:
    df_num[i] = le.fit_transform(df_num[i])
## Histogramme ueber alle Features
## nur mit numerischen Werten moeglich
plt.figure()
fig, axes = plt.subplots(11, 2, figsize=(10, 20))
abschluss = df_num[df_num.Zielvariable == 1]
kein_abschluss = df_num[df_num.Zielvariable == 0]
ax = axes.ravel()
binsar = np.array([50,2,31,12,50,50,2,12,3,4,2,50,2,2,3,20,10,20,4,4,2,50])
for i in range(22):
    _, bins = np.histogram(df_num.values[:, i], bins=binsar[i])
    ax[i].hist(abschluss.values[:, i], bins=bins, color=cm3(0), alpha=.5)
    ax[i].hist(kein_abschluss.values[:, i], bins=bins, color=cm3(2), alpha=.5)
    ax[i].set_title(list(df_num.columns.values)[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["Abschluss", "kein Abschluss"], loc="upper left")
fig.tight_layout()
plt.show()

# Transformationen wieder entfernen:
df_raw = df_raw.drop('Log_Dauer', 1)
df_raw = df_raw.drop('Season', 1)
df_raw = df_raw.drop('MonthFirstHalf', 1)

### Vereinfachungen auf Datenspalten nach ersten Analysen:
df_raw['Ergebnis letzte Kampagne_Erfolg'] = df_raw['Ergebnis letzte Kampagne'].map(campaign_dic)
df_raw['Kontaktart_Handy'] = df_raw['Kontaktart'].map(contactform_dic)
df_test_challenge['Ergebnis letzte Kampagne_Erfolg'] = df_test_challenge['Ergebnis letzte Kampagne'].map(campaign_dic)
df_test_challenge['Kontaktart_Handy'] = df_test_challenge['Kontaktart'].map(contactform_dic)
#df_raw['Art der Anstellung Mgmt'] = np.where(df_raw['Art der Anstellung'] == 'Management', 1, 0)

## Tag im Zeitablauf ermittlen, Stammnummer raus:
## 1. Jahr: bis einschliesslich 432170075
## 2. Jahr: bis einschliesslich 432184937
df_raw['Zeitverlauf Tage'] = df_raw['Monat']+df_raw['Tag']+np.where(df_raw['Stammnummer'] > 432170075, 365, 0)+np.where(df_raw['Stammnummer'] > 432184937, 365, 0)
df_test_challenge['Zeitverlauf Tage'] = df_test_challenge['Monat']+df_test_challenge['Tag']+np.where(df_test_challenge['Stammnummer'] > 432170075, 365, 0)+np.where(df_test_challenge['Stammnummer'] > 432184937, 365, 0)
# danach Stammnummer raus
df_raw = df_raw.drop('Stammnummer', 1)
# Stammnummer fuer Ergebnis-Output speichern
output = pd.DataFrame()
output['Id'] = df_test_challenge['Stammnummer']
df_test_challenge = df_test_challenge.drop('Stammnummer', 1)


# Drop Kategorien, die kaum Einfluss haben:
columns_drop = ['Ausfall Kredit','Art der Anstellung','Kontaktart','Ergebnis letzte Kampagne',u'Schulabschluß','Familienstand']
#['Art der Anstellung','Ausfall Kredit','Kontaktart','Ergebnis letzte Kampagne',u'Schulabschluß','Familienstand']
df_raw = df_raw.drop(columns_drop, 1)
df_test_challenge = df_test_challenge.drop(columns_drop, 1)

print('df_raw.shape:', df_raw.shape )

## Restliche kategorische Daten in mehrere Spalten splitten: 
# Get dummies
df_raw = pd.get_dummies(df_raw, prefix_sep='_')#, drop_first=True)
# Manuell jeweils eine Spalte entfernen, um Kollinearitaet zu vermeiden: 
columns_drop = []
#['Art der Anstellung_Unbekannt','Familienstand_single',u'Schulabschluß_Unbekannt','Kontaktart_Unbekannt','Ergebnis letzte Kampagne_Unbekannt']
df_raw = df_raw.drop(columns_drop, 1)

print('df_raw.shape (nach get_dummies): {}' .format(df_raw.shape) )

# Eigenwerte berechnen, um auf Kollinearitaet zu pruefen:
corr=np.corrcoef(df_raw,rowvar=0)
W,V=np.linalg.eig(corr)
print('Eigenwerte:\n', W )

## Spalteninfos:
df_raw.info()

## DataFrame-Auswertung:
pd.set_option('display.expand_frame_repr', False) # Option setzen, damit alle Spalten angezeigt werden
## Ein paar aggregierte Kennzahlen zu den Daten ausgeben:
print(df_raw.describe())

## Zielvariable separieren
y = df_raw['Zielvariable']
df = df_raw.drop('Zielvariable', 1)
df_test_challenge = df_test_challenge.drop('Zielvariable', 1)

""" getestet, jedoch keine verbesserung des score
## PCA
# Import `PCA` from `sklearn.decomposition`
from sklearn.decomposition import PCA
# Build the model
pca = PCA(n_components=8)
# Reduce the data, output is ndarray
reduced_data = pca.fit_transform(df)
# Inspect shape of the `reduced_data`
reduced_data.shape
# print out the reduced data
print(reduced_data)

values = reduced_data
index = ['Row'+str(i) for i in range(1, len(reduced_data)+1)]
df = pd.DataFrame(values, index=index)
"""

## Daten in Trainings- und Testdaten separieren
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2,random_state =None)

print(X_train.shape )
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

######################################################################################################
##################################### Modellanalysen: ################################################
######################################################################################################


######################################################################################################
###### Bereitstellung Funktionen fuer Analysen: ######################################################
######################################################################################################
 
### Funktion um Plot fuer verschiedene Parameter auszugeben:
def plot_model_accuracy(settings,training_accuracy,test_accuracy,cross_val_list,cross_val_std,title,xlabel, scaleLog=False):
    plt.figure()
    plt.plot(settings, training_accuracy, label="training accuracy")
    plt.plot(settings, test_accuracy, label="test accuracy")
    if len(cross_val_list) > 0:
        plt.plot(settings, cross_val_list, label="cross validation score")
        print('Cross Val Score is ',cross_val_list)
        if len(cross_val_std) > 0:
            plt.plot(settings, np.array(cross_val_list)+ np.array(cross_val_std), 'b--', label="cross val score +/- std-dev")
            plt.plot(settings, np.array(cross_val_list)- np.array(cross_val_std), 'b--')
    plt.ylabel("Accuracy")
    plt.xlabel(xlabel)
    plt.legend()
    plt.title(title)
    if scaleLog:
        plt.xscale('log')
    plt.show()
### Plot Feature Importance: 
def plot_feature_importances_case(model,title):
    n_features = X_train.shape[1]
    plt.figure()
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), list(df.columns.values))
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.title(title)
    plt.show()
### Funktion um Ergebnisse auszugeben:
def summarize_results(model,str_name):
    y_pred = model.predict(X_test)
    #y_pred = (model.predict_proba(X_test)[:,1]>0.25).astype(int) # Anderen Schwellenwert ausprobieren
    # Ergebnis-Reports ausgeben:
    print('Classification Report ' + str_name + ':')
    print(classification_report(y_test, y_pred))
    print('Ergebnis Matrix Testdaten ' + str_name + ':')
    print(confusion_matrix(y_test, y_pred))
    # Accuracy Score
    print('Test score for ' + str_name + ':',model.score(X_test, y_test))
    
    ### Receiver Operating Characteristic (ROC) (aus sklearn-Hilfe)
    y_score = model.predict_proba(X_test)[:,1]
    # Compute ROC curve and ROC area
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    # ROC Kurve plotten
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ' + str_name)
    plt.legend(loc="lower right")
    plt.show()

############################################################################################################### 
#####################################  Logistic Regression ####################################################
############################################################################################################### 

from sklearn.linear_model import LogisticRegression

training_accuracy = []
test_accuracy = []
cross_valuation_score = []
cross_valuation_score_std= []
# Parameter C = [0.001 0.01 0.1 1 10 100 1000] . je groesser desto komplexer das Modell, sehr klein entspricht starker verallgemeinerung
settings = np.array([0.01,0.1,1,10,50,100,500,1000])
for c in settings:
    # Modell setzen
    clf = LogisticRegression(C=c) #, penalty="l1")
    # Modell mit Trainingsdaten kalibrieren 
    clf.fit(X_train, y_train)
    # Guete des Modells auf Trainingsdaten auslesen
    training_accuracy.append(clf.score(X_train, y_train))
    # Guete des Modells auf Testdaten auslesen
    test_accuracy.append(clf.score(X_test, y_test))
    ### Cross Validation Scores:
    this_scores = cross_val_score(clf, df, y, cv=5, n_jobs=1)
    cross_valuation_score.append(np.mean(this_scores))
    cross_valuation_score_std.append(np.std(this_scores))
    
    
# Modell Genauigkeit gg Anzahl Entscheidungsbaeume plotten
plot_model_accuracy(settings,training_accuracy,test_accuracy,cross_valuation_score,cross_valuation_score_std,"Logistische Regression","Parameter C",True)
# Ergebnisse ueber Fkt ausgeben:
summarize_results(clf,"Logistische Regression (C=" + str(c) +')')

"""
#### Auspraeungen der Koeffizienten betrachten:
plt.figure()
plt.plot(clf.coef_.T, 'o', label="C=1000")
#plt.plot(logreg100.coef_.T, '^', label="C=100")
#plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(X_train.shape[1]), list(df.columns.values), rotation=90)
plt.hlines(0, 0, X_train.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.title("Auspraegung Parameter bei Logistische Regression C=1000")

#### ROC_AUC Scores nach unterschiedlichen Parametern
auc_mat = np.array([[0.,0.,0.,0.,0.,0.,0.,0.,0.]])
settings_c = np.array([0.01,0.1,1,2,5,10,50,100,500])
for i in range(0, 9):
        c = settings_c[i]
        # Modell setzen
        clf = LogisticRegression(C=c)
        # Modell mit Trainingsdaten kalibrieren 
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:,1]
        # Compute ROC curve and ROC area
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        auc_mat[0,i] = roc_auc
print(auc_mat)
"""
############################################################################################################### 
#####################################  Decision Trees #########################################################
############################################################################################################### 

from sklearn.tree import DecisionTreeClassifier

training_accuracy = []
test_accuracy = []
cross_valuation_score = []
cross_valuation_score_std= []
# Parameter maxDepth. Beschraenkt die Tiefe des Entscheidungsbaumes
settings = np.array([2,3,4,5,6,7,8])
for maxDepth in settings:
    # Modell setzen
    clf = DecisionTreeClassifier(max_depth=maxDepth, random_state=0)
    # Modell mit Trainingsdaten kalibrieren 
    clf.fit(X_train, y_train)
    # Guete des Modells auf Trainingsdaten auslesen
    training_accuracy.append(clf.score(X_train, y_train))
    # Guete des Modells auf Testdaten auslesen
    test_accuracy.append(clf.score(X_test, y_test))
    ### Cross Validation Scores:
    this_scores = cross_val_score(clf, df, y, cv=5, n_jobs=1)
    cross_valuation_score.append(np.mean(this_scores))
    cross_valuation_score_std.append(np.std(this_scores))
        
# Modell Genauigkeit gg Tiefe des Entscheidungsbaumes plotten
plot_model_accuracy(settings,training_accuracy,test_accuracy,cross_valuation_score,cross_valuation_score_std,"Entscheidungsbaum","Max Tiefe")
# Feature Importance mit Fkt plotten:
plot_feature_importances_case(clf,"Feature Importance Tree Depth=" +str(maxDepth))
# Ergebnisse ueber Fkt ausgeben:
summarize_results(clf,'Tree (Depth=' +str(maxDepth) + ')')

"""
#### ROC_AUC Scores nach unterschiedlichen Parametern
auc_mat = np.array([[0.,0.,0.,0.,0.,0.,0.,0.,0.]])
settings_depth = np.array([2,4,6,8,10,12,14,16,18])
for i in range(0, 9):
        x_depth = settings_depth[i]
        # Modell setzen
        clf = DecisionTreeClassifier(max_depth=x_depth, random_state=0)
        # Modell mit Trainingsdaten kalibrieren 
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:,1]
        # Compute ROC curve and ROC area
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        auc_mat[0,i] = roc_auc
print(auc_mat)
"""
############################################################################################################### 
#####################################  Random Forest ##########################################################
############################################################################################################### 
from sklearn.ensemble import RandomForestClassifier
"""
## Nach Parameter max Tiefe:
training_accuracy = []
test_accuracy = []
cross_valuation_score = []
# Parameter max_depth. max Tiefe je Entscheidungsbaum
settings = np.array([2,3,4,5,7,20])
for n in settings:
    # Modell setzen
    clf = RandomForestClassifier(n_estimators=400,max_features=int(math.sqrt(X_train.shape[1])),max_depth=n, random_state=0) 
    # Modell mit Trainingsdaten kalibrieren 
    clf.fit(X_train, y_train)
    # Guete des Modells auf Trainingsdaten auslesen
    training_accuracy.append(clf.score(X_train, y_train))
    # Guete des Modells auf Testdaten auslesen
    test_accuracy.append(clf.score(X_test, y_test))
# Modell Genauigkeit gg Parameter plotten
plot_model_accuracy(settings,training_accuracy,test_accuracy,"Random Forest","Max Tiefe je Baum")

## Nach Parameter max Anzahl Features je Entscheidungsbaum:
training_accuracy = []
test_accuracy = []
cross_valuation_score = []
# Parameter max_features. Anzahl der Features je Entscheidungsbaum
settings = np.array([2,3,4,5,7,10])
for n in settings:
    # Modell setzen
    clf = RandomForestClassifier(n_estimators=100,max_features=n,max_depth=10, random_state=0) 
    # Modell mit Trainingsdaten kalibrieren 
    clf.fit(X_train, y_train)
    # Guete des Modells auf Trainingsdaten auslesen
    training_accuracy.append(clf.score(X_train, y_train))
    # Guete des Modells auf Testdaten auslesen
    test_accuracy.append(clf.score(X_test, y_test))
# Modell Genauigkeit gg Parameter plotten
plot_model_accuracy(settings,training_accuracy,test_accuracy,"Random Forest","Anzahl Features je Baum")
"""
## Nach Parameter Anzahl der Entscheidungsbaeume:
training_accuracy = []
test_accuracy = []
cross_valuation_score = []
cross_valuation_score_std= []
# Parameter n_estimators. Anzahl der Entscheidungsbaeume
settings = np.array([5,50,100,200,400])
for n in settings:
    # Modell setzen
    clf = RandomForestClassifier(n_estimators=n,max_features=max(4,int(math.sqrt(X_train.shape[1]))),max_depth=10, random_state=0) 
    # Modell mit Trainingsdaten kalibrieren 
    clf.fit(X_train, y_train)
    # Guete des Modells auf Trainingsdaten auslesen
    training_accuracy.append(clf.score(X_train, y_train))
    # Guete des Modells auf Testdaten auslesen
    test_accuracy.append(clf.score(X_test, y_test))
    ### Cross Validation Scores:
    this_scores = cross_val_score(clf, df, y, cv=5, n_jobs=1)
    cross_valuation_score.append(np.mean(this_scores))
    cross_valuation_score_std.append(np.std(this_scores))
    
# Modell Genauigkeit gg Anzahl Entscheidungsbaeume plotten
plot_model_accuracy(settings,training_accuracy,test_accuracy,cross_valuation_score,cross_valuation_score_std,"Random Forest","Anzahl Entscheidungsbaeume")
# Feature Importance mit Fkt plotten:
plot_feature_importances_case(clf,"Feature Importance Random Forest n=" + str(n))
# Ergebnisse ueber Fkt ausgeben:
summarize_results(clf,'Random Forest (n=' + str(n) + ')')

"""
#### ROC_AUC Scores nach unterschiedlichen Parametern
auc_mat = np.array([[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]])
settings_feat = np.array([2,3,4,5,6])
settings_depth = np.array([8,10,12,14,16,18])
for i in range(0, 5):
    for j in range(0, 6):
        x_feat = settings_feat[i]
        x_depth = settings_depth[j]
        # Modell setzen
        clf = RandomForestClassifier(n_estimators=400,max_features=x_feat,max_depth=x_depth, random_state=0) 
        # Modell mit Trainingsdaten kalibrieren 
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:,1]
        # Compute ROC curve and ROC area
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        auc_mat[i,j] = roc_auc

print(auc_mat)
"""

############################################################################################################### 
########################### Gewaehltes Modell auf Testdaten anwenden: 
############################################################################################################### 
clf = RandomForestClassifier(n_estimators=400,max_features=5,max_depth=10, random_state=0)
clf.fit(df, y) #hier kann die Gesamtmenge an Daten verwendet werden

y_score = clf.predict_proba(df_test_challenge)[:,1]
output['Expected']= y_score

output.to_csv('./Loesung_Enno_kaggle.csv', encoding='utf-8', index=False)


