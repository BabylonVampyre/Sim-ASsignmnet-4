import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

nr_scenarios = 5
nr_runs = 20

# open the csv and save as dataframes
dataframes = {}

# collect average driving time per scenario
avg_bridge_delay_times = []

# collect all trucks per scenario
all_bridge_delay = pd.DataFrame()

for i in range(nr_scenarios):  # save as dataframes and collect average driving times.
    dataframes[i] = pd.read_csv('../model/experiment/Scenario{}_Bridge_delay_time.csv'.format(i))  # save the csv
    index =  dataframes[i]['ID'] #save the index of the bridges
    dataframes[i] = dataframes[i].drop(columns=[col for col in dataframes[i].columns if 'KPI' not in col]) #select only the columns with KPI in their name
    avg_bridge_delay_times = dataframes[i].mean(axis=1)  # save the average delay time per bridge
    all_bridge_delay = pd.concat([all_bridge_delay, avg_bridge_delay_times],axis = 1) #put the list of delay time per bridge in the big dataframe

#insert the index and rename headers
all_bridge_delay.insert(loc=0, column='ID', value=index)
all_bridge_delay.columns = ['ID', 'Scenario 0', 'Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4']

#make a new df that doesnt have the columns that arent needed for the plot
all_bridge_delay_plot = all_bridge_delay.drop(columns=['ID','Scenario 0'])

# boxplot per scenario, aggregated over all runs
ax = sns.boxplot(all_bridge_delay_plot)
#ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
#plt.savefig('boxplot_per_scenario.png')
plt.show()

#print Average bridge delay time per scenario
print('Average per scenario:\n', all_bridge_delay_plot.mean())


#itterate through columns and select id for top 5 most delay time
for column in all_bridge_delay.columns[1:]:
    top_5_entries = all_bridge_delay.nlargest(5, column)[['ID', column]]  # Get top 5 entries for the current scenario
    print(f"Top 5 for {column}:")
    for index, row in top_5_entries.iterrows():
        print(f"ID: {row['ID']}, Value: {row[column]}")
    print()