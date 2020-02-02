# -*- coding: utf-8 -*-
"""
This program implements a Neural Network to predict Wildfire risk using weather data

@author: Isita Talukdar
"""

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#Set this to True to test the ANN with random output for training and testing
#Prediction accuracy should be bad(close to 50%)
use_random_data_for_testing = False
#Control use of previous month(s) weather data
#0 for current month, 1 for current+prev 1 month, 2 for current+prev 2 months
month_hist_cnt = 0
#Epoch Num
ep_num = 100
#County Num
county_cnt = 3
#Number of Years- going back from 2016
num_years = 10
#Number of Hidden Layer
hidden_layer_cnt = 2
#Multiplier for hidden layer size = factor*input_size
hidden_layer_size_factor = 0.25

months = {
		 0 : "Jan",
		 1 : "Feb",
     2 : "Mar",
     3 : "Apr",
     4 : "May",
     5 : "Jun",
     6 : "Jul",
     7 : "Aug",
     8 : "Sep",
     9 : "Oct",
    10 : "Nov",
    11 : "Dec",
	}

weather_headers = [
    'Year',
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec',   
]

county_list = {
		 0 : "LA County",
		 1 : "Tuolumne",
     2 : "Ventura",
     3: "Alpine"
	}

#County Number to Index
county_num_to_index = {
        19 : 0,
        55 : 1,
        56 : 2,
         2 : 3,
        }

#Creating List of Weather Data files asscociated with a county number that
#is used in the fire data spreadsheet. This puts the same county number info in
#all the dataframes for consistency

#List of Temp Files of All Counties Along with County Index used in fire data
allTempFiles = {
        19 : 'LA_County_AvgTemp.csv',
        55 : 'Tuolumne_AvgTemp.csv',
        56 : 'Ventura_AvgTemp.csv',
        2 : 'Alpine_AvgTemp.csv',
        }

#List of Precip Files of All Counties Along with County Index used in fire data
allPrecipFiles = {
        19 : 'LA_County_Precip.csv',
        55 : 'Tuolumne_Precip.csv',
        56 : 'Ventura_Precip.csv',
        2 : 'Alpine_Precip.csv',
        }

#List of Palmer Files of All Counties Along with County Index used in fire data
allPalmerFiles = {
        19 : 'LA_County_Palmer.csv',
        55 : 'Tuolumne_Palmer.csv',
        56 : 'Ventura_Palmer.csv',
        2 : 'Alpine_Palmer.csv',
        }




"""
Defining the main execution block of ANN as a function
"""
def wf_ann(
        use_random_data_for_testing = False,
        month_hist_cnt = 2,
        ep_num = 100,
        county_cnt = 4,
        num_years = 50,
        hidden_layer_cnt = 2,
        hidden_layer_size_factor = 2,
        ):

    if(month_hist_cnt == 2):
        use_2_prev_month = True
        use_1_prev_month = False
    elif(month_hist_cnt == 1):
        use_2_prev_month = False
        use_1_prev_month = True
    else:
        use_2_prev_month = False
        use_1_prev_month = False
    
    max_year = 2016 #DO NOT USE 2017 it has an error in the weather data
    min_year = max_year-num_years+1
    if(min_year < 1900):
        min_year = 1900
    
    
    #Temperature
    list_ = []
    for key in allTempFiles:
        #print(allTempFiles[key])
        if(county_num_to_index[key] < county_cnt):
            df1 = pd.read_csv(allTempFiles[key],index_col=None, header = None, names = weather_headers)
            df1['county_index'] = key
            list_.append(df1)
    all_county_temp = pd.concat(list_, axis = 0, join='inner', ignore_index = False)
    
    #Precip
    list_ = []
    for key in allPrecipFiles:
        #print(allPrecipFiles[key])
        if(county_num_to_index[key] < county_cnt):
            df1 = pd.read_csv(allPrecipFiles[key],index_col=None, header = None, names = weather_headers)
            df1['county_index'] = key
            list_.append(df1)
    all_county_precip = pd.concat(list_, axis = 0, join='inner', ignore_index = False)
    
    #Palmer
    list_ = []
    for key in allPalmerFiles:
        #print(allPalmerFiles[key])
        if(county_num_to_index[key] < county_cnt):
            df1 = pd.read_csv(allPalmerFiles[key],index_col=None, header = None, names = weather_headers)
            df1['county_num'] = key
            list_.append(df1)
    all_county_palmer = pd.concat(list_, axis = 0, join='inner', ignore_index = False)
    
    #Compute Main Dataframe Dimensions
    county_row_cnt = (max_year - min_year + 1)*12
    total_row_cnt = county_cnt*(county_row_cnt)
     
    
    df = pd.DataFrame(index=range(0,total_row_cnt), columns=['county', 'year', 'month', 'temp', 'precip', 'palmer',
                          'temp_p1', 'precip_p1', 'palmer_p1',
                          'temp_p2', 'precip_p2', 'palmer_p2',
                          'fire'])
    
    #Set county, year, month of all rows in a fixed pattern
    for county_index in range(0, county_cnt):
        for row_index in range(0, county_row_cnt):
            df.set_value(row_index+county_row_cnt*county_index, 'county', county_list[county_index], takeable=False)
        for year_index in range(min_year, max_year+1):
            for month_index in range(0, 12):
                df.set_value(county_row_cnt*county_index+12*(year_index-min_year)+month_index, 'year',year_index, takeable=False)
                df.set_value(county_row_cnt*county_index+12*(year_index-min_year)+month_index, 'month',months[month_index], takeable=False)
    
    #Merge Weather Data into main Dataframe
    row_cnt = all_county_temp.shape[0]
    for row_index in range(0, row_cnt):
        year = int(all_county_temp.iloc[row_index, 0])
        county_num = all_county_temp.iloc[row_index, 13]
        county_index = county_num_to_index[county_num]
        if (year >= min_year and year <= max_year):
            for month_index in range(0, 12):
                val = all_county_temp.iloc[row_index, month_index+1]
                target_row_index = county_row_cnt*county_index+12*(year-min_year)+month_index
                df.set_value(target_row_index, 'temp',val, takeable=False)
                val = all_county_precip.iloc[row_index, month_index+1]
                df.set_value(target_row_index, 'precip',val, takeable=False)
                val = all_county_palmer.iloc[row_index, month_index+1]
                df.set_value(target_row_index, 'palmer',val, takeable=False)
                #Boundary Case: At the switching of county, copy the same month data
                if(target_row_index % county_row_cnt < 1):
                    prev_row_index = target_row_index
                else:
                    prev_row_index = target_row_index-1
                val = df.iloc[prev_row_index,df.columns.get_loc("temp")]
                df.set_value(target_row_index, 'temp_p1',val, takeable=False)
                val = df.iloc[prev_row_index,df.columns.get_loc("precip")]
                df.set_value(target_row_index, 'precip_p1',val, takeable=False)
                val = df.iloc[prev_row_index, df.columns.get_loc("palmer")]
                df.set_value(target_row_index, 'palmer_p1',val, takeable=False)
                #Boundary Case: At the switching of county, copy the same month data
                if(target_row_index % county_row_cnt < 2):
                    prev_row_index = target_row_index
                else:
                    prev_row_index = target_row_index-2
                val = df.iloc[prev_row_index,df.columns.get_loc("temp")]
                df.set_value(target_row_index, 'temp_p2',val, takeable=False)
                val = df.iloc[prev_row_index,df.columns.get_loc("precip")]
                df.set_value(target_row_index, 'precip_p2',val, takeable=False)
                val = df.iloc[prev_row_index, df.columns.get_loc("palmer")]
                df.set_value(target_row_index, 'palmer_p2',val, takeable=False)           
                
                if(use_random_data_for_testing):
                    fire_value = random.randint(0,1)
                else:
                    fire_value = int(0)
                df.set_value(county_row_cnt*county_index+12*(year-min_year)+month_index, 'fire',fire_value, takeable=False)
            
    
    #Read in fire data 
    fire_data = pd.read_csv('CA_fires_counties_all.csv')
    row_cnt = fire_data.shape[0]
    for row_index in range(0, row_cnt):
        #print(row_index)
        county_num = fire_data.iloc[row_index, 18]
        if county_num in county_num_to_index and county_num_to_index[county_num] <county_cnt:
           county_index = county_num_to_index[county_num]
        else:
            continue 
        if(fire_data.iloc[row_index, 2] != 'NaN'):
            year = fire_data.iloc[row_index, 2]
        else:
            continue
        #The fire spreadsheet has some invalid month entries
        if(fire_data.iloc[row_index, 20].isdigit()):
            month = int(fire_data.iloc[row_index, 20])
            if(month <= 0 or month >=13):
                continue
        else:
            continue
        if(year >= min_year and year <= max_year):
            df.set_value(county_row_cnt*county_index+12*(year-min_year)+month-1, 'fire',int(1), takeable=False)       
    
    #change fire column to int to allow handling by confusion matrix for testing
    df['fire'] = df.fire.astype(int)
    
    #Selecting Columns for Use in ANN
    if(use_2_prev_month):
        X = df.iloc[:, [df.columns.get_loc("county"),df.columns.get_loc("month"),
                    df.columns.get_loc("temp"), df.columns.get_loc("precip"), df.columns.get_loc("palmer"),
                    df.columns.get_loc("temp_p1"), df.columns.get_loc("precip_p1"), df.columns.get_loc("palmer_p1"),
                    df.columns.get_loc("temp_p2"), df.columns.get_loc("precip_p2"), df.columns.get_loc("palmer_p2")]].values
    elif(use_1_prev_month): 
        X = df.iloc[:, [df.columns.get_loc("county"),df.columns.get_loc("month"),
                    df.columns.get_loc("temp"), df.columns.get_loc("precip"), df.columns.get_loc("palmer"),
                    df.columns.get_loc("temp_p1"), df.columns.get_loc("precip_p1"), df.columns.get_loc("palmer_p1")]].values
    else:
        X = df.iloc[:, [df.columns.get_loc("county"),df.columns.get_loc("month"),
                    df.columns.get_loc("temp"), df.columns.get_loc("precip"), df.columns.get_loc("palmer")]].values
    y = df.iloc[:,df.columns.get_loc("fire")].values
    
    #Encoding Categorical Data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    #Encode County Name
    labelencoder_X_0 = LabelEncoder()
    X[:, 0] = labelencoder_X_0.fit_transform(X[:, 0])
    #Encode Month
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) 
    
    #One hot Encode County
    onehotencoder = OneHotEncoder(categorical_features = [0])
    X = onehotencoder.fit_transform(X).toarray()     
    #One hot encoding puts new columns at front
    X = X[:, 1:]
    
    #One hot Encode Month
    onehotencoder = OneHotEncoder(categorical_features = [county_cnt-1])
    X = onehotencoder.fit_transform(X).toarray()     
    #One hot encoding puts new columns at front
    X = X[:, 1:]
    
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
               
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    #Main Neural Network 
    
    #Import Keras Library and Packages
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    
    classifier = Sequential()
    #CalculateInput Size: how many counties-1 + how many months(12) -1 
    #+ (how many months(1,2,3)*how many weather data(3))
    if(use_2_prev_month):
        ann_input_size = county_cnt-1+(12-1)+3*3
    elif(use_1_prev_month): 
        ann_input_size = county_cnt-1+(12-1)+3*2
    else:
        ann_input_size = county_cnt-1+(12-1)+3*1
        
    #Compute hidden layer size based on input size
    hidden_layer_size = int(hidden_layer_size_factor*ann_input_size)
    
    #Add Input Layer and 1st Hidden Layer
    classifier.add(Dense(output_dim = hidden_layer_size, init = 'uniform', activation = 'relu', input_dim = ann_input_size))
    
    
    #N additional Hidden Layers
    for hidden_layer_index in range(0, hidden_layer_cnt):
        classifier.add(Dense(output_dim = hidden_layer_size, init = 'uniform', activation = 'relu'))
    
    #add Output Layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    
    #Compiling ANN
    classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #Fit ANN into Training Set
    classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = ep_num)
    
    #Predictions and Evaluations
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    score = (accuracy_score(y_test, y_pred))*100
    #print("Prediction Accuracy =  ", score)
    return score
#End of Function  wf_ann()



#Main Code
#Start of Test Loop

f = open("wf_result_epoch_num_10_150.csv", "w")
f.write("Accuracy over number of epoch\n") 
f.write("Epoch,Accuracy\n")
for epoch_index in range(10, 150, 10):
    res_str = str(epoch_index)
    score = wf_ann(
            ep_num = epoch_index
           )
    res_str += ","+str(score)
    res_str += "\n"
    f.write(res_str)
f.close()

f = open("wf_result_hidden_layer_num_1_10.csv", "w")
f.write("Accuracy over number of hidden layers\n") 
f.write("Hidden Layers,Accuracy\n")
for hidden_layer_num in range(1, 10):
    res_str = str(hidden_layer_num)
    score = wf_ann(
            hidden_layer_cnt = hidden_layer_num
           )
    res_str += ","+str(score)
    res_str += "\n"
    f.write(res_str)
f.close()

f = open("wf_result_hidden_layer_nodes_factor_1_9.csv", "w")
f.write("Accuracy over number of hidden layer nodes, shown as factor of input layer size\n") 
f.write("Hidden Layer Size Factor,Accuracy\n")
for hidden_layer_factor_index in range(1, 9):
    hidden_layer_factor = hidden_layer_factor_index*0.50
    res_str = str(hidden_layer_factor)
    score = wf_ann(
            hidden_layer_size_factor = hidden_layer_factor
           )
    res_str += ","+str(score)
    res_str += "\n"
    f.write(res_str)    
f.close()

f = open("wf_result_year_10_116_month_hist_0_3.csv", "w") 
f.write("Accuracy over number of years and month histories\n") 
f.write("Year,Hist=0,Hist=1,Hist=2\n")
for year_index in range(10, 116, 5):
    res_str = str(year_index)
    for hist_index in range(0,3):
        score = wf_ann(
                month_hist_cnt = hist_index,
                num_years = year_index,
               )
        
        res_str += ","+str(score)
    res_str += "\n"
    f.write(res_str)
f.close()

f = open("wf_result_year_10_100_num_counties_2_5.csv", "w")
f.write("Accuracy over number of years and number of counties\n") 
f.write("Year,Num Counties=1,Num Counties=2,Num Counties=3,Num Counties=4\n")
for year_index in range(10, 100, 10):
    res_str = str(year_index)
    for county_num_index in range(2,5):
        score = wf_ann(
                county_cnt = county_num_index,
                num_years = year_index,
               )
        
        res_str += ","+str(score)
    res_str += "\n"
    f.write(res_str)
f.close()

print("DONE")
#End of Test Loop

#----End of Code----

#County Codes for Reference
"""
1	Alameda
2	Alpine
3	Amador
4	Butte
5	Calaveras
6	Colusa
7	Contra Costa
8	Del Norte
9	El Dorado
10	Fresno
11	Glenn
12	Humboldt
13	Imperial
14	Inyo
15	Kern
16	Kings
17	Lake
18	Lassen
19	Los Angeles
20	Madera
21	Marin
22	Mariposa
23	Mendocino
24	Merced
25	Modoc
26	Mono
27	Monterey
28	Napa
29	Nevada
30	Orange
31	Placer
32	Plumas
33	Riverside
34	Sacramento
35	San Benito
36	San Bernardino
37	San Diego
38	San Francisco
39	San Joaquin
40	San Luis Obispo
41	San Mateo
42	Santa Barbara
43	Santa Clara
44	Santa Cruz
45	Shasta
46	Sierra
47	Siskiyou
48	Solano
49	Sonoma
50	Stanislaus
51	Sutter
52	Tehama
53	Trinity
54	Tulare
55	Tuolumne
56	Ventura
57	Yolo
58	Yuba
"""