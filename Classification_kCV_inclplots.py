#%%
'''
Credits: This is a showcase of the methodic transferability from the
device recognition approach from: 
Dittmann, S., Semantische Assistenzen für komplexe Digitale Zwillinge, 2024
The following code is based on the results from a term paper supervised in the
context of above mentioned work:
Doganer et al., Übertragung semantischer Assistenzen des Fabrikbetriebs 
auf die Medizintechnik,2021
The code is slightly new structured and slightly updated to include a basic 
algorithm competition incl. newly presented time series classifiers 
(HIVE COTE 2.0)
'''


#%%Import packages
#Base
import os
import pandas as pd
import numpy as np
import wfdb
import wget
import pickle
import tarfile
import zipfile
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sn

#Import kFold CV package
from sklearn.model_selection import RepeatedStratifiedKFold



#time series classifiers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,accuracy_score, precision_score
from sktime.classification.interval_based import RandomIntervalSpectralEnsemble
from sktime.classification.hybrid import HIVECOTEV2
from sktime.transformations.panel.shapelet_transform import (RandomShapeletTransform)
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
from sktime.classification.sklearn import RotationForest

#%% Set working directory
#After initial download (see section with downloads below), set working directory 
#to path with data files
os.chdir('<Please specify working directory here>')
os.getcwd()


#Call customized data preparation from file in same working directory
from build_sktime_data import build_sktime_data
from plot_cm import plot_cm


#%%Load data fist time
#NOTE: Make sure to only run this section once as the fetched 
#files are relatively big (ca.600MB)

'''
Load ECG data:
1. Lugovaya T.S., Biometric human identification based on electrocardiogram.
Retrieved from https://www.physionet.org/content/ecgiddb/1.0.0/
Cited in thesis as: [Lug-00]
2. Yakushenko E. et al., St.-Petersburg Institute of Cardiological Technics 12-lead Arrhythmia Database
Retrieved from https://physionet.org/content/incartdb/1.0.0/
Cited in thesis as: [Yak-00]

#Fetch zip file from [Lug-00]
url="https://www.physionet.org/static/published-projects/ecgiddb/ecg-id-database-1.0.0.zip"
data_ecg_lug=wget.download(url)

#Extract zip file in currently selected working dir
with zipfile.ZipFile('ecg-id-database-1.0.0.zip','r') as zip_ref:
    zip_ref.extractall('')


#Fetch zip file from [Yak-00]
url="https://physionet.org/static/published-projects/incartdb/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0.zip"
data_ecg_yak=wget.download(url)

#Extract zip file in currently selected working dir
with zipfile.ZipFile('st-petersburg-incart-12-lead-arrhythmia-database-1.0.0.zip','r') as zip_ref:
    zip_ref.extractall('')
'''

'''
Load Bioreactor data:
Loening, M., sktime-tutorial-pydata-amsterdam-2020.
Retrieved from: https://github.com/sktime/sktime-tutorial-pydata-amsterdam-2020/blob/main/notebooks/01_overview_of_learning_tasks.ipynb
Cited in thesis as: [Loe-20]
Originating from:
Rieth, C. et al., Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation
Cited in thesis as: [Rie-17]


#Fetch zip file from [Loe-20]
url="https://raw.githubusercontent.com/sktime/sktime-tutorial-pydata-amsterdam-2020/main/data/chemical_process_data.csv"
data_bre_loe=wget.download(url)
'''

#%% Read in and structure [Lug-00] data

#List and count all (anonymised) persons in the dataset
List = os.listdir("ecg-id-database-1.0.0/")
List_of_persons = [i for i in List if i.startswith('Person_')]
length = len(List_of_persons)

#List nbr of files per person
file_count = []
for i in range (0, length):
    path, dirs, files = next(os.walk(os.path.join('ecg-id-database-1.0.0', 
    List_of_persons[i])))
    file_count.append(len(files))

#Check nbr of measurements per Person (3 files per Measurement)
measure=[]
for i in range (0, len(file_count)):
    measure.append(int(file_count[i] / 3))

#Create general dict
dict_methodic_transf = {}

#Creat ecg dict in dict
dict_methodic_transf['ecg'] = {}

#Iterate over persons in [Lug-00] data and store data in dict
for k in range(0, len(measure)):
    #Iterate over records
    for i in range(1, measure[k]+1):
        #Read .dat ecg data incl. metadata in .hea via wfdb package
        #For details see e.g. https://danielsepulvedaestay.medium.com
        #/reading-wfdb-data-into-powerbi-f2aa46bc2092
        person_rdsamp = wfdb.rdsamp(os.path.join('ecg-id-database-1.0.0',
        List_of_persons[k],'rec_%d' %i))
        
        if i == 1:
            #Definde structure of touples, regard only voltage time series
            #Tile entries from 1000 (original) to 5000 entries by repetition
            person = pd.DataFrame(np.tile(person_rdsamp[0],(5,1)), 
            columns=['voltage_1','voltage_2'])
        else:
            person = pd.DataFrame(np.tile(person_rdsamp[0],(5,1)), 
            columns=['voltage_1','voltage_2'])
        
        #Store newly structured data in general dict under ecg key
        dict_methodic_transf['ecg']['Person_{}_rec{}'.format(k+1, i)] = person


#%% Read in and structure data from bioreactor
breactor_data = pd.read_csv('chemical_process_data.csv', skiprows=[1,2], 
                            usecols=pd.read_csv('chemical_process_data.csv',
                            nrows=1).columns[1:])
print(breactor_data)

suffix_1 = 0
suffix_2 = 0
temporary_list=[]
#Rename columns
for i in range(0, len(breactor_data.columns)):
    
    #rename pressure data from bioreactor
    if breactor_data.columns[i].startswith('pressure')==True:
        temporary_list.append('pressure_'+str(suffix_1))
        suffix_1=suffix_1+1
    
    #rename temperature data from bioreactor
    if breactor_data.columns[i].startswith('temperature')==True:
        temporary_list.append('temperature_'+str(suffix_2))
        suffix_2=suffix_2+1

breactor_data.columns = temporary_list 

#Store bioreactor data in general dictionary for subsequent classification
count = 0
count_p = 0
count_t = 0

#initialize pressure dict within general dict
dict_methodic_transf['pressure'] = {}
#initialize temperature dict within general dict
dict_methodic_transf['temperature'] = {}

#Iterate through breactor_data df and store time series in general dict
for column in breactor_data:
    count = count +1
    
    #np.tile is below used to repeat the 98 entries per timeseries to reach
    #more than 5000 entries. Later truncated to 5000 resp. 1000 again.
    if 'pressure' in str(column):
        
        dict_methodic_transf['pressure']['Measurement_'+str(count)] = {}
        dict_methodic_transf['pressure']['Measurement_'+str(count)]=pd.DataFrame(
            np.tile(breactor_data.loc[:,str(column):str(column)],(52,1)),
            columns=['pressure_%d' %count_p])
        count_p = count_p+1

    if 'temperature' in str(column):
        
        dict_methodic_transf['temperature']['Measurement_'+str(count)] = {}
        dict_methodic_transf['temperature']['Measurement_'+str(count)]=pd.DataFrame(
            np.tile(breactor_data.loc[:,str(column):str(column)],(52,1)),
            columns=['temperature_%d' %count_t])
        count_t = count_t+1

#%% Read in further universum data.
'''
The aim is to introduce a special kind of noise into the dataset
For further information see: Semantische Assistenzen für komplexe Digitale Zwillinge

Partial Data from dataset published in Dittmann, S. et al., Device recognition 
assistants as additional data management method for Digital Twins
Cited in thesis as: [Dit-unv]
'''

#Before running code below, ensure that the file sktime_test_OS_data is 
#in current wd

#Read in "universum" data
filename = 'sktime_test_OS_data.spydata'
tar = tarfile.open(filename, "r")

tar.extractall()
extracted_files = tar.getnames()

for file in extracted_files:
    if file.endswith('.pickle'):
        with open(file,'rb') as fdesc:
            dict_new = pd.read_pickle(fdesc)

dict_new = dict_new['dict_new']

#Add data to main Dictionary
for column in dict_new:
    dict_methodic_transf[column]={}
    dict_methodic_transf[column]=dict_new[column]
    
#%%

#%%

#%% Define base model for subsequent k-fold cross-validation
hc2 = HIVECOTEV2(
    stc_params={
        "estimator": RotationForest(n_estimators=3),
        "n_shapelet_samples": 500, 
        "max_shapelets": 20,
        "batch_size": 100,
    },
    drcif_params={"n_estimators": 10},
    arsenal_params={"num_kernels": 100, "n_estimators": 5},
    tde_params={
        "n_parameter_samples": 25,
        "max_ensemble_size": 5,
        "randomly_selected_params": 10,
    },
    random_state=123,
    time_limit_in_minutes=5
    )

#%% def function to calc metrics for repeated cv below

def measure_KPIs(classifier, X_train, y_train, X_test, y_test):
        
        #train & predict
        classifier.fit(X_train, y_train)
        preds = classifier.predict(X_test)
        
        #calculate cm, accuracy and precision
        cm = confusion_matrix(y_test, preds)
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='macro')
                
        model_name = str(classifier)
        
        return cm, acc, precision, model_name, preds


#%%Define data for repeated kCV

sktimedata = build_sktime_data(dict_methodic_transf)
X, y = sktimedata['X'], sktimedata['y']    

    
#%%HIVE-COTE BASE kCV for transferability 

n_splits = 3
n_repeats = 3
number_of_algorithms = 1


res_cm = []
res_acc = []
res_precision = []
res_best_model = []
res_best_params = []
res_y_pred = []
res_y_test = []

rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state = 123)

#produce series of random but reproducible seeds to use during loops
rng = np.random.default_rng(seed=123)
loop_random_seeds = rng.integers(low=0, high=1000000, size = n_splits*n_repeats*number_of_algorithms)  


#Define loop for repeated kCV
i = 0

for train_index, test_index in rskf.split(X, y):

    #Define storage location for split results
    results_table = []
    
    #Define data for split
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #Convert to DataFrame
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    
    
    #Train model and calc metrics
    cm, acc, precision, model_name, preds = measure_KPIs(hc2, X_train, y_train, X_test, y_test)
    
    res_cm.append(cm)
    res_acc.append(acc)
    res_precision.append(precision)
    res_y_pred.append(preds)
    res_y_test.append(y_test)
    
    i = i+1


#%% Calc weighted precision score
"""
Please note, the above calculated precision score does not yet account for the inbalance
between the groups in the data set and is therefore misleading.
Therefore, below the weighted precision score is additionally calculated.
"""
res_weight_precision = []

for i in range(0, len(res_y_test)):
    print(i)
    weight_prec = precision_score(res_y_test[i], res_y_pred[i], average='weighted')
    res_weight_precision.append(weight_prec)





#%% Read in and structure [Yak-00] data

list_of_files=os.listdir("C:/Users/USER/OneDrive - Technische Universität Berlin/Desktop/D/Mongrafie/Code/DevRecogAssist_Healthcare_Code_ECG/files")

temp_list=[]

#-1 below to adjust offset that list indizes start from 0
for i in range(0, len(list_of_files)):
    if list_of_files[i].endswith('hea'):
        temp_list2=list_of_files[i].split(".")
        temp_list.append(temp_list2[0])
        
#read ecg data from [Yak-00] in dictionary
dict_ecg_yak = {}
dict_ecg_yak['ecg'] = {}

for i in range(0, len(temp_list)):
    ecg_data_read=wfdb.rdsamp(os.path.join('C:/Users/USER/OneDrive - Technische Universität Berlin/Desktop/D/Mongrafie/Code/DevRecogAssist_Healthcare_Code_ECG/files', temp_list[i]))
    dict_ecg_yak['ecg']['Record_{}'.format(i+1)]=pd.DataFrame(ecg_data_read[0],
                                                columns=['I','II','III','AVR',
                                                         'AVL','AVF', 'V1','V2',
                                                         'V3','V4','V5','V6'])

#%%Use new ecg data only as test data
sktimedata2 = build_sktime_data(dict_ecg_yak)

#Define data mostly as test data, discard train data
X_train, X_test, y_train, y_test = train_test_split(sktimedata2['X'], sktimedata2['y'],
                                                    train_size=0.01, test_size=0.99,
                                                    random_state=101, 
                                                    stratify=sktimedata2['y'])

X_test = pd.DataFrame(X_test)
#apply pre-trained HIVE COTE Algorithm to unseen [Yak-00] ECG Data
predicted_values_yak = hc2.predict(X_test)
#show confusion matrix
cm_yak=confusion_matrix(y_test, predicted_values_yak)


#plot confusion matrix
cm_yak = confusion_matrix(y_test, predicted_values_yak)

class_dict={0: 'acceleration_sensor',
            1: 'ecg'}

plot_cm(cm_yak, class_dict, y_test)



'''
Results of the base version:
    1. The base classifier (hc2) acieves an accuracy of 99,6% in the dataset
    with 5 classes incl. the biorector timeseries data and universum data.
    2. From the 891 Testcases in the unseen ecg data from a second dataset 
    ([Yag-00]) captured in a different context, 99% are correctly classified.
    5 ecg timeseries are classified as acceleration data.

'''
#%%
"""
Following code only for plotting results
"""

counter = 0

class_dict={0: 'acceleration_sensor',
            1: 'ecg',
            2: 'gyroscope',
            3: 'pressure',
            4: 'temperature'}

for elem in res_cm:
    plot_cm(elem, class_dict, res_y_test[counter])
    counter = counter+1
    
#%%
#Plot exemplary cm
#define color code for plot
low_color = "#DAE8FC"
high_color = "#6C8EBF"

pd.set_option('display.float_format', '{:.0f}'.format)

#create color map
cmp_cstm = LinearSegmentedColormap.from_list(
    "cstm",[low_color, high_color], N=4)


sn.set(font='serif')
sn.heatmap(res_cm[7], fmt='d',annot=True, cmap=cmp_cstm, linewidth=.5,
            xticklabels=['Beschleunigung', 'EKG', 'Gyroskop', 'Druck', 
                        'Temperatur'], 
           yticklabels=['Beschleunigung', 'EKG', 'Gyroskop', 'Druck', 
                       'Temperatur'])

plt.xticks(rotation=45, ha='right')
plt.xlabel("Vorhersage", labelpad=10)
plt.ylabel("Wahre Geräteklasse", labelpad=10)

#plt.savefig()
plt.show()

#%%Plot of exemplary time series for appendix

#Prepare data

#ECG Data
ecg_data_plot_1 = dict_methodic_transf['ecg']['Person_1_rec1']['voltage_1'][:500]
ecg_data_plot_2 = dict_methodic_transf['ecg']['Person_10_rec1']['voltage_1'][:500]

#Pressure Data
pressure_data_plot_1 = dict_methodic_transf['pressure']['Measurement_1']['pressure_0'][:500]
pressure_data_plot_2 = dict_methodic_transf['pressure']['Measurement_10']['pressure_9'][:500]

#Temperature Data
temperature_data_plot_1 = dict_methodic_transf['temperature']['Measurement_502']['temperature_1'][:500]
temperature_data_plot_2 = dict_methodic_transf['temperature']['Measurement_504']['temperature_3'][:500]

#Universum Data
acceleration_data_plot_1 = dict_methodic_transf['acceleration_sensor']['cnc_mill_tool_wear_exp1']['x1_actualacceleration'][:500]
gyroscope_data_plot_2 = dict_methodic_transf['gyroscope']['sensiml']['gyroscopex'][:500]

#Smoothening for better Plot quality
def smoothing_ts(ts, window_size=8):
    return ts.rolling(window=window_size).mean()

ecg_data_plot_1 = smoothing_ts(ecg_data_plot_1)
ecg_data_plot_2 = smoothing_ts(ecg_data_plot_2)

pressure_data_plot_1 = smoothing_ts(pressure_data_plot_1)
pressure_data_plot_2 = smoothing_ts(pressure_data_plot_2)

temperature_data_plot_1 = smoothing_ts(temperature_data_plot_1)
temperature_data_plot_2 = smoothing_ts(temperature_data_plot_2)

acceleration_data_plot_1 = smoothing_ts(acceleration_data_plot_1)
gyroscope_data_plot_2 = smoothing_ts(gyroscope_data_plot_2)

#%%Plot

plt.rcParams['font.family'] ='serif'

DIN_width = 8.27
DIN_hight = 11.69

plot_hight = DIN_hight*1

ratio = DIN_width / plot_hight

fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(DIN_width, plot_hight), gridspec_kw={'height_ratios':[1]*4, 'wspace':0.5, 'hspace':0.5})
ax = ax.flatten()

plt.tight_layout()


ax[2].yaxis.set_label_coords(-0.35, 0.5)
ax[3].yaxis.set_label_coords(-0.35, 0.5)
ax[4].yaxis.set_label_coords(-0.35, 0.5)
ax[5].yaxis.set_label_coords(-0.35, 0.5)

ax[0].plot(ecg_data_plot_1, color = '#6C8EBF')
ax[0].set_facecolor('white')
ax[0].spines['bottom'].set_color('black')
ax[0].spines['left'].set_color('black')
ax[0].spines['top'].set_color('black')
ax[0].spines['right'].set_color('black')
ax[0].tick_params(axis='both', colors='black')
ax[0].yaxis.tick_left()
ax[0].xaxis.tick_bottom()
ax[0].set_ylabel('EKG [mV]', rotation=90, color='black', labelpad=10)
ax[0].set_xlabel('Beobachtungen (500Hz)', color='black')


ax[1].plot(ecg_data_plot_2, color = '#6C8EBF')
ax[1].set_facecolor('white')
ax[1].spines['bottom'].set_color('black')
ax[1].spines['left'].set_color('black')
ax[1].spines['top'].set_color('black')
ax[1].spines['right'].set_color('black')
ax[1].tick_params(axis='both', colors='black')
ax[1].yaxis.tick_left()
ax[1].xaxis.tick_bottom()
ax[1].set_ylabel('EKG [mV]', rotation=90, color='black', labelpad=10)
ax[1].set_xlabel('Beobachtungen (500Hz)', color='black')


ax[2].plot(pressure_data_plot_1, color = '#6C8EBF', label = ['Druck'])
ax[2].set_facecolor('white')
ax[2].spines['bottom'].set_color('black')
ax[2].spines['left'].set_color('black')
ax[2].spines['top'].set_color('black')
ax[2].spines['right'].set_color('black')
ax[2].tick_params(axis='both', colors='black')
ax[2].yaxis.tick_left()
ax[2].xaxis.tick_bottom()
ax[2].set_ylabel('Druck []', rotation=90, color='black', labelpad=10)
ax[2].set_xlabel('Beobachtungen', color='black')

ax[3].plot(pressure_data_plot_2, color = '#6C8EBF', label = ['Druck'])
ax[3].set_facecolor('white')
ax[3].spines['bottom'].set_color('black')
ax[3].spines['left'].set_color('black')
ax[3].spines['top'].set_color('black')
ax[3].spines['right'].set_color('black')
ax[3].tick_params(axis='both', colors='black')
ax[3].yaxis.tick_left()
ax[3].xaxis.tick_bottom()
ax[3].set_ylabel('Druck []', rotation=90, color='black', labelpad=10)
ax[3].set_xlabel('Beobachtungen', color='black')


ax[4].plot(temperature_data_plot_1, color = '#6C8EBF', label = ['Temperatur'])
ax[4].set_facecolor('white')
ax[4].spines['bottom'].set_color('black')
ax[4].spines['left'].set_color('black')
ax[4].spines['top'].set_color('black')
ax[4].spines['right'].set_color('black')
ax[4].tick_params(axis='both', colors='black')
ax[4].yaxis.tick_left()
ax[4].xaxis.tick_bottom()
ax[4].set_ylabel('Temperatur []', rotation=90, color='black', labelpad=10)
ax[4].set_xlabel('Beobachtungen', color='black')


ax[5].plot(temperature_data_plot_2, color = '#6C8EBF', label = ['Temperatur'])
ax[5].set_facecolor('white')
ax[5].spines['bottom'].set_color('black')
ax[5].spines['left'].set_color('black')
ax[5].spines['top'].set_color('black')
ax[5].spines['right'].set_color('black')
ax[5].tick_params(axis='both', colors='black')
ax[5].yaxis.tick_left()
ax[5].xaxis.tick_bottom()
ax[5].set_ylabel('Temperatur []', rotation=90, color='black', labelpad=10)
ax[5].set_xlabel('Beobachtungen', color='black')


ax[6].plot(acceleration_data_plot_1, color = '#6C8EBF', label = ['Beschleunigung'])
ax[6].set_facecolor('white')
ax[6].spines['bottom'].set_color('black')
ax[6].spines['left'].set_color('black')
ax[6].spines['top'].set_color('black')
ax[6].spines['right'].set_color('black')
ax[6].tick_params(axis='both', colors='black')
ax[6].yaxis.tick_left()
ax[6].xaxis.tick_bottom()
ax[6].set_ylabel('Beschleunigung [mm/s$^2$]', rotation=90, color='black', labelpad=10)
ax[6].set_xlabel('Beobachtungen (1 kHz)', color='black')


ax[7].plot(gyroscope_data_plot_2, color = '#6C8EBF', label = ['Gyroskop'])
ax[7].set_facecolor('white')
ax[7].spines['bottom'].set_color('black')
ax[7].spines['left'].set_color('black')
ax[7].spines['top'].set_color('black')
ax[7].spines['right'].set_color('black')
ax[7].tick_params(axis='both', colors='black')
ax[7].yaxis.tick_left()
ax[7].xaxis.tick_bottom()
ax[7].set_ylabel('Gyroskop []', rotation=90, color='black', labelpad=10)
ax[7].set_xlabel('Beobachtungen (1 kHz)', color='black')

fig.text(0.48, 0.05, 'Einheiten und Abtastrate der Zeitreihen sind, wo verfügbar, angegeben', ha='center', fontsize=12, color='black')


#plt.savefig()            
plt.show()
