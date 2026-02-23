# load files in tmp folder starting with 'full_df'

import os
import time
import pandas as pd

source_path = 'tmp'
destination_path = 'datasets'
quantization_bins = 50 

if not os.path.exists(destination_path):
    os.makedirs(destination_path)

def load_files():
    files = []
    for file in os.listdir('tmp'):
        if file.startswith('full_df'):
            files.append(file)
    return files

files = load_files()

print(files)

def prepare_data(file, quantization_bins):
    dataset_name = '_'.join(file.split('_')[2:])[:-4]
    filename = f'{dataset_name}_quantized_{quantization_bins}.csv'
    if filename in os.listdir(destination_path):
        print(f'{filename} already exists')
        return "finished"
    # file size
    print(f'{file} size: {os.path.getsize(f"{source_path}/{file}") / 1e6:.2f} MB')
    t0 = time.time()
    df = pd.read_csv(f'{source_path}/{file}')
    print(f'Loaded {file} in {time.time() - t0:.2f} s')
    shape_before = df.shape
    print(df.shape)
    print(df.head())
    features_to_quantize = list(df)[1:-1]
    print(f'Quantizing {len(features_to_quantize)} features...')
    for f in features_to_quantize:
        if df[f].nunique() < quantization_bins:
            print(f'Feature {f} has less unique values than quantization bins. Skipping...')
            continue
        df[f] = pd.qcut(df[f], q=quantization_bins, duplicates='drop', labels=False)
    shape_after = df.shape
    print('... done')
    print(df.shape)
    assert shape_before == shape_after
    # save file
    df.to_csv(f'{destination_path}/{filename}', index=False)
    print(f'Saved {filename} in {time.time() - t0:.2f} s')
    return "finished"

for file in files:
    prepare_data(file, quantization_bins)
