import os
import pandas as pd
from tableshift import get_dataset

dataset_name = 'acsfoodstamps' # terminated by signal SIGKILL (Forced quit)
dataset_name = 'acsincome' # terminated by signal SIGKILL (Forced quit)
#dataset_name = 'acspubcov' # terminated by signal SIGKILL (Forced quit)
#dataset_name = 'acsunemployment' # terminated by signal SIGKILL (Forced quit)
#dataset_name = 'anes' # FileNotFoundError: [Errno 2] No such file or directory: 'tmp/anes_timeseries_cdf_csv_20220916/anes_timeseries_cdf_csv_20220916.csv'
dataset_name = 'assistments' # ok
#dataset_name = 'brfss_blood_pressure' # ok
#dataset_name = 'brfss_diabetes' # ok
dataset_name = 'college_scorecard' # ok
dataset_name = 'diabetes_readmission' # ok
dataset_name = 'meps' # ok
#dataset_name = 'mimic_extract_los_3' # The data file can be accessed at https://storage.googleapis.com/mimic_extract/all_hourly_data.h5 after  obtaining access as described at  https://github.com/MLforHealth/MIMIC_Extract
#dataset_name = 'mimic_extract_mort_hosp' # The data file can be accessed at https://storage.googleapis.com/mimic_extract/all_hourly_data.h5 after  obtaining access as described at https://github.com/MLforHealth/MIMIC_Extract
#dataset_name = 'nhanes_lead' #ValueError: Document does not match SAS Version 5 or 6 Transport (XPORT) format
#dataset_name = 'physionet' # a lot of files
#dataset_name = 'sipp' # FileNotFoundError: [Errno 2] No such file or directory: 'tmp/sipp/sipp_2014.csv'

datasets_to_generate = []
datasets_to_generate = datasets_to_generate + ['acsfoodstamps']
datasets_to_generate = datasets_to_generate + ['acsincome']
datasets_to_generate = datasets_to_generate + ['assistments']
datasets_to_generate = datasets_to_generate + ['college_scorecard']
datasets_to_generate = datasets_to_generate + ['diabetes_readmission']
datasets_to_generate = datasets_to_generate + ['meps']

#env_definition_method = 'originalenv'
env_definition_method = 'binaryood'

for dataset_name in datasets_to_generate:
    print('####################################### dataset', dataset_name)
    print('looking for', f'tmp/full_df_{dataset_name}_envdefinition_{env_definition_method}.csv')
    if os.path.exists(f"tmp/full_df_{dataset_name}_envdefinition_{env_definition_method}.csv"):
        print(f"tmp/full_df_{dataset_name}_envdefinition_{env_definition_method}.csv already exists")
        continue
    dset = get_dataset(dataset_name, use_cached=False)
    list_X, list_y, list_group, list_domain = [], [], [], []
    for split in dset.splits:
        x, y, group, domain = dset.get_pandas(split)
        list_X.append(x)
        list_y.append(y)
        list_group.append(group)
        if env_definition_method == 'originalenv':
            pass
        elif env_definition_method == 'binaryood':
            env_value = [1 if 'ood' in split else 0] 
            domain = pd.Series([env_value]*domain.shape[0], index=domain.index)
        list_domain.append(domain)

    full_X_df = pd.concat(list_X)
    full_y_series = pd.concat(list_y)
    full_domain_series = pd.concat(list_domain)
    print(type(full_X_df))
    print(type(full_y_series))
    print(type(full_domain_series))

    print(full_y_series.shape)
    print(full_domain_series.shape)

    full_y_df = pd.DataFrame(data=list(full_y_series), columns=["y"], index=full_y_series.index)
    env_df = pd.DataFrame(list(full_domain_series), columns=["env"], index=full_X_df.index)

    print(full_X_df.shape)
    print(full_y_df.shape)
    print(env_df.shape)

    # assert all indexes are the same
    assert env_df.index.equals(full_X_df.index)
    assert env_df.index.equals(full_y_df.index)

    full_df = pd.concat([env_df, full_X_df, full_y_df], axis=1)
    print("full_df.shape", full_df.shape)
    print(full_df['env'].value_counts())
    print(full_df.head())

    # count nans in total
    print("full_df.isnull().sum().sum()", full_df.isnull().sum().sum())

    full_df.to_csv(f"tmp/full_df_{dataset_name}_envdefinition_{env_definition_method}.csv", index=False)