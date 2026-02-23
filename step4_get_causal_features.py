import os
import pandas as pd
from tableshift import get_dataset

datasets_to_try = []
for file in os.listdir('datasets'):
    if file.endswith('.csv'):
        dataset_name = file.split('_envdefinition')[0]
        causal_dataset_name = dataset_name + '_causal'
        if f'{causal_dataset_name}_features.pkl' in os.listdir('datasets'):
            print(f'{causal_dataset_name}_features.pkl already exists')
            continue
        if "food" in dataset_name:
            continue
        if 'employment' in dataset_name:
            continue
        datasets_to_try = datasets_to_try + [causal_dataset_name]

# remove duplicates
datasets_to_try = list(set(datasets_to_try))
print('datasets_to_try', datasets_to_try)

for causal_dataset_name in datasets_to_try:
    causaldset = get_dataset(causal_dataset_name, use_cached=False)
    X, y, _, _ = causaldset.get_pandas("train")
    causal_variables = list(X.columns)
    # save as pickle
    import pickle
    with open(f'datasets/{causal_dataset_name}_features.pkl', 'wb') as f:
        pickle.dump(causal_variables, f)
