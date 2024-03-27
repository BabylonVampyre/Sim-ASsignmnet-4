import numpy as np
import pandas as pd

def extract_data_traffic():
    #function to call first that imports the data files
    roads = ['N1', 'N102', 'N104', 'N105' ,'N2', 'N204', 'N207' ,'N208']
    #roads = ['N1', 'N102']
    df_traffic_list = []
    for road in roads:
        df_traffic_list.append(f'df_{road}')



    for road,df in zip(roads,df_traffic_list):
        #print(road,df)

        df = pd.read_html(f"../data/RMMS/{road}.traffic.htm")
        table = pd.concat(df[4:])
        column_names = table.iloc[2]
        column_names[0] = 'Link number'
        column_names[1] = 'Link name'
        table.columns = column_names
        table = table.drop(index = [0,1,2])
        df = table.reset_index(drop=True)
        print(df)


extract_data_traffic()

