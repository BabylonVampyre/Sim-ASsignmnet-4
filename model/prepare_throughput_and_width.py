import numpy as np
import pandas as pd

def extract_data_traffic(road):
    """
    This method extracts the traffic data from the HTM files and stores it in a dataframe
    :param road: a string of what road data is needed
    :return: a formatted dataframe containing the Chainage of the start of the link,
    the Chainage of the end of the link, the length of the link and the throuput in AADT
    """
    #import data and format it into a readable dataframe
    #it takes a string of what road needs to be read and formatted
    df = pd.read_html(f"../data/RMMS/{road}.traffic.htm")
    #format data and rename headers
    table = pd.concat(df[4:])
    column_names = table.iloc[2]
    column_names[0] = 'Link number'
    column_names[1] = 'Link name'
    column_names[4] = 'Chainage start location'
    column_names[7] = 'Chainage end location'
    column_names[8] = 'Link length'
    column_names[25] = 'AADT'
    table.columns = column_names
    table = table.drop(index = [0,1,2])
    df = table.reset_index(drop=True)

    columns_to_drop = ['Link number', 'Link name', 'LRP', 'Offset', 'Heavy Truck', 'Medium Truck',
                       'Small Truck', 'Large Bus', 'Medium Bus', 'Micro Bus', 'Utility', 'Car',
                       'Auto Rickshaw', 'Motor Cycle', 'Bi-Cycle', 'Cycle Rickshaw', 'Cart',
                       'Motorized', 'Non Motorized', 'Total AADT']
    df = df.drop(columns=columns_to_drop)

    return df

def extract_widths_data(road):
    """
    This Method extracts and formats the width data
    :param road: a string of what road data is needed
    :return: a formatted dataframe containing the Chainage of the start of the no of lanes,
    the Chainage of the end of the no of lanes and the number of lanes
    """
    #open the file
    with open(f"../data/RMMS/{road}.widths.txt", 'r') as file:
        lines = file.readlines()

    # Process each line and split the entries based on '&' and ':'
    data = []
    for line in lines:
        line_data = line.strip().split('&')
        for entry in line_data:
            entry_data = entry.split(':')
            data.append(entry_data)

    # Create DataFrame
    df = pd.DataFrame(data)
    #transpose the data
    df = df.T
    #drop wrong columns titles
    df.drop(0,inplace=  True)
    #give the column names
    df.columns = ['TotalRecord', 'RoadID', 'RoadNo', 'StartChainage', 'EndChainage', 'CWayWidth',
                                     'NoOfLanes', 'RoadName', 'RoadClass', 'RoadLength', 'StartLocation',
                  'EndLocation','ROADW']
    #drop unnecessary columns
    columns_to_drop = ['TotalRecord', 'RoadID', 'RoadNo','CWayWidth','RoadName', 'RoadClass', 'RoadLength',
                       'StartLocation', 'EndLocation','ROADW']
    df = df.drop(columns = columns_to_drop)
    df = df.reset_index(drop=True)

    return df

#list of roads needed
roads = ['N1', 'N102', 'N104', 'N105' ,'N2', 'N204', 'N207' ,'N208']
#empty list to be filled with dataframes of the road data
df_traffic_list = []
df_width_list = []
#call extraction function and fill list
for road in roads:
    df_traffic_list.append(extract_data_traffic(road))
    df_width_list.append(extract_widths_data(road))


for i in df_traffic_list:
    print(i)

for j in df_width_list:
    print(j)


