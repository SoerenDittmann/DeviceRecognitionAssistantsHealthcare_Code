#%%Define function to transform raw data into the format expected from sktime
'''
Credits:Based on Code from: Mathieu, MP., Enabling scalability in digital twin data 
acquisition: A machine learning driven device recognition assistant, 2021
Published as part of the experimental evaluation in: Dittmann, S. et al., 
Device recognition assistants as additional data management method for 
Digital Twins
Cited in thesis as: [Dit-unv]
'''

#%%Import required packages
import pandas as pd



#%%Define function to transform raw data into the format expected from sktime


def build_sktime_data(input_dict):
    
    all_data = pd.DataFrame(columns=['X', 'Y'])
    ts_length = 1000
    data_list=[]
    
    #iterate over all device types in dict
    for classlabel, device_type in enumerate(input_dict.keys()):
        #iterate over all data sources of device type
        for key in input_dict[device_type].keys():

            #Store individual ts as dataframe, truncate data to 5000 entries
            dataset = pd.DataFrame(input_dict[device_type][key])[:ts_length]
            
            for device in dataset:
                #Set naming convention in dataframe
                device_name = key + '_' + device
                new_row = [device_name, dataset[device], device_type, len(dataset)]
                data_list.append(new_row)
    
    all_data = pd.DataFrame(data=data_list, columns= ['dataset', 'X', 'y', 
                                                      'series_length'])
    return all_data
            