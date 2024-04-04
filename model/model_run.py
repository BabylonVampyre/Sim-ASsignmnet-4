from model import BangladeshModel
import pandas as pd
from components import Bridge
"""
    Run simulation
    Print output at terminal
"""
# ---------------------------------------------------------------
# iniciate scenario's, last number is the scenario number
S0 = [0, 0, 0, 0, 0]
S1 = [0, 0, 0, 5, 1]
S2 = [0, 0, 5, 10, 2]
S3 = [0, 5, 10, 20, 3]
S4 = [5, 10, 20, 40, 4]


# run time 5 x 24 hours; 1 tick 1 minute
run_length = 7200

# run time 1000 ticks
#run_length = 1000

seed_list = [1234567, 1234568, 1234569, 1234560, 1234561, 1234562, 1234563, 1234564, 1234565, 1234566,
             2234567, 2234568, 2234569, 2234560, 2234561, 2234562, 2234563, 2234564, 2234565, 2234566]
scenario_list = [S0,S1, S2, S3, S4]
#seed_list = [1234567,1234568,1234569]
# scenario_list = [S0]

# create variables to keep track of the total runs and how much is done
number_of_runs = len(scenario_list)*len(seed_list)
run_counter = 0
scenario_counter = 0

# itterate through the seed and scenario lists and run the model for the run_length
for scenario_index, i in enumerate(scenario_list):
    # make an empty dataframe to save all runs of a single scenario and one for the total delay time of bridges
    df_scenario = pd.DataFrame()
    bridge_columns = ['ID', 'Total delay time','Total passed traffic','KPI']
    df_bridge_delay_time = pd.DataFrame(columns=bridge_columns)
    for seed_index, j in enumerate(seed_list):
        sim_model = BangladeshModel(scenario=i, seed=j)

        for k in range(run_length):
            sim_model.step()
            #if the final setp, get the id and total delay time of the bridges out of the model and into a dataframe
            if k == (run_length-1):
                df_bridge_delay_time_singlerun = pd.DataFrame(columns=bridge_columns)
                for agent in sim_model.schedule.agents: #itterate through agents
                    if isinstance(agent, Bridge): #if bridge, then save its required attributes
                        df_bridge_delay_time_singlerun.loc[len(df_bridge_delay_time_singlerun)] =\
                            {'ID': int(agent.unique_id), 'Total delay time': int(agent.total_delay_time)
                                ,'Total passed traffic' : int(agent.total_passed_traffic),
                             'KPI' :int(agent.total_delay_time/agent.total_passed_traffic)}

        # call the dataframe of the single run and change the column names to reflect what seed it used
        df_scenario_singlerun = sim_model.df_driving_time
        df_scenario_singlerun.columns = ['Driving_Time' + str(j)]
        # merge the single run data frame with the base dataframe to save all runs of a single scenario in one DF
        df_scenario = pd.concat([df_scenario, df_scenario_singlerun], axis=1)

        # call the dataframe of the single run and change the column names to reflect what seed it used
        df_bridge_delay_time_singlerun.columns = ['ID', 'Total delay time'+ str(j),'Total passed traffic'+ str(j),'KPI' + str(j)]
        # merge the single run data frame with the base dataframe to save all runs of a single scenario in one DF
        if df_bridge_delay_time.empty:
            df_bridge_delay_time = df_bridge_delay_time_singlerun
        else:
            df_bridge_delay_time = pd.merge(df_bridge_delay_time, df_bridge_delay_time_singlerun,
                                            on='ID', how='outer')

        # print how far the full run is
        run_counter += 1
        print(run_counter, '/', number_of_runs, 'Done:', scenario_counter, 'seed: ', j)
    scenario_counter += 1
    df_scenario.to_csv(f"../model/experiment/Scenario{scenario_index}_Vehicle_delay_time.csv",
                       index=False)
    df_bridge_delay_time.fillna(0)
    df_bridge_delay_time.to_csv(f"../model/experiment/Scenario{scenario_index}_Bridge_delay_time.csv",
                                index=False)


