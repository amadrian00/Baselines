import re
import os
import glob
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind='correlation')
warnings.filterwarnings("ignore")

def get_dataframe(name = 'ABIDE'):
    if name == 'ABIDE':
        path_data, path_demo, columns, columns_assignment = get_paths()

        df_demo = prepare_demo(path_demo, columns, columns_assignment)

        df = prepare_dataframe(path_data)

        df['site_id'] = pd.Categorical(df['site']).codes
        df = pd.merge(df, df_demo, on='sid', how='inner')
        df['stratify_col'] = df['site_id'].astype(str) + "_" + df['label'].astype(str)
    else:
        df = prepare_dataframe2('/DataCommon3/daheo/ADNI3/step4_postprocess/1_Atlas_BOLD_Extract/GSR/CC200')
    return df

def prepare_demo(path, columns, columns_assignment):
    df_demo = pd.read_csv(path)[columns]
    df_demo.rename(columns=columns_assignment,inplace=True)
    df_demo['handedness'] = df_demo['handedness'].apply(
        lambda x: 1 if isinstance(x, str) and x.startswith('R')
        else 0 if isinstance(x, str) and x.startswith('L')
        else 0.5 if isinstance(x, str) and (x.startswith('Ambi') or x == 'Mixed')
        else x
        ).astype(float)

    columns = df_demo.columns
    if 'handedness_score' in columns:
        df_demo['handedness'] = df_demo['handedness'].fillna(df_demo['handedness_score'] / 100)
        df_demo.drop('handedness_score', axis=1, inplace=True)
    if 'label' in columns:
        df_demo['label'] = df_demo['label'].replace(2, 0)
    return df_demo

def get_paths():
    path_data = '/DataCommon3/daheo/ABIDE/ABIDEI/ABIDEI_Provided_AtlasBOLD/cpac/filt_global/rois_cc200'
    path_demo = '/DataCommon4/aayuso/Phenotypic_V1_0b_preprocessed1_original.csv'
    columns = ['SUB_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX', 'HANDEDNESS_CATEGORY', 'HANDEDNESS_SCORES']
    columns_assignment = {'SUB_ID': 'sid', 'DX_GROUP': 'label', 'AGE_AT_SCAN': 'age', 'SEX': 'sex',
                              'HANDEDNESS_CATEGORY': 'handedness', 'HANDEDNESS_SCORES': 'handedness_score'}

    return path_data, path_demo, columns, columns_assignment

def prepare_dataframe(path_data):
    file_list = sorted(glob.glob(os.path.join(path_data, "*.1D")))
    pattern = re.compile(r"([A-Za-z_]+_[A-Za-z0-9]+|[A-Za-z_]+)_([\d]+)_rois")

    correlation_matrices = []
    time_series = []
    data={'site':[], 'sid':[]}
    for file in file_list:
        time_serie = [np.loadtxt(file)]
        time_series.append(time_serie)

        correlation_matrix = correlation_measure.fit_transform(time_serie)[0]
        correlation_matrix[correlation_matrix!=correlation_matrix]=0
        correlation_matrices.append(correlation_matrix)

        match = pattern.search(file)
        if match:
            data['site'].append(match.group(1))
            data['sid'].append(int(match.group(2)))


    df_dict = {
        'time': time_series,
        'corr': correlation_matrices,
        'sid': data['sid'],
        'site': data['site']
    }
    df = pd.DataFrame.from_dict(df_dict)

    return df

def prepare_dataframe2(path_data):
    file_list = sorted(glob.glob(os.path.join(path_data, "**", "*.npz"), recursive=True))
    subject_count = defaultdict(int)

    data_dict = {
        'time': [],
        'corr': [],
        'sid': [],
        'sex': [],
        'age': [],
        'label': []
    }

    for file in file_list:
        data = np.load(file)
        base_id = data['eid'].item()
        unique_id = f"{base_id}_{subject_count[base_id]}"
        subject_count[base_id] += 1

        data_dict['time'].append(data['bold'].tolist())
        data_dict['corr'].append(data['corr'])
        data_dict['sex'].append(data['gender'])
        data_dict['age'].append(float(data['age']))
        data_dict['sid'].append(unique_id)
        data_dict['label'].append(int(data['dx']!='CN'))
        data.close()

    return pd.DataFrame.from_dict(data_dict)

