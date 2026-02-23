from exp_utils import run_exp
from joblib import Parallel, delayed

N_THREADS = 1

datasets_to_run = []
datasets_to_run = datasets_to_run + ['diabetes_readmission_envdefinition_binaryood_quantized_50']
datasets_to_run = datasets_to_run + ['college_scorecard_envdefinition_binaryood_quantized_50']
datasets_to_run = datasets_to_run + ['meps_envdefinition_binaryood_quantized_50']
datasets_to_run = datasets_to_run + ['acsfoodstamps_envdefinition_binaryood_quantized_50']
datasets_to_run = datasets_to_run + ['assistments_envdefinition_binaryood_quantized_50']
datasets_to_run = datasets_to_run + ['acsincome_envdefinition_binaryood_quantized_50']

datsaet_to_splitproportion = {
    'acsfoodstamps_envdefinition_binaryood_quantized_50'            : 0.1,
    'acsincome_envdefinition_binaryood_quantized_50'                : 0.1,
    'assistments_envdefinition_binaryood_quantized_50'              : 0.1,
    'college_scorecard_envdefinition_binaryood_quantized_50'        : 0.5,
    'diabetes_readmission_envdefinition_binaryood_quantized_50'     : 0.5,
    'meps_envdefinition_binaryood_quantized_50'                     : 0.5,
}


print('datasets_to_run', datasets_to_run)


algos_to_run = []
algos_to_run = algos_to_run + ['SCM']
#algos_to_run = algos_to_run + ['ICSCM']

splits_to_run = list(range(20))

print('datasets_to_run', datasets_to_run)
print('algos_to_run', algos_to_run)

for dataset_name in datasets_to_run:
    split_proportion = datsaet_to_splitproportion[dataset_name]
    print('##### dataset', dataset_name)
    for algo in algos_to_run:
        Parallel(n_jobs=N_THREADS)(delayed(run_exp)(dataset_name=dataset_name, algo=algo, split_id=split_id, split_proportion=split_proportion) for split_id in splits_to_run)
