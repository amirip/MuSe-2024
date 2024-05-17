import os
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
import pandas as pd
import pickle
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from config import PATH_TO_FEATURES, PATH_TO_LABELS, PARTITION_FILES, HUMOR, PERCEPTION


################# GLOBAL UTILITY METHODS #############################################

def get_data_partition(partition_file) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Reads mappings from subject ids to their partition and vice versa
    :param partition_file: path to the partition file (csv with two columns: id, partition)
    :return: dicts subject2partition, partition2subject
    """
    subject2partition, partition2subject = {}, {}
    if not os.path.exists(partition_file):
        print(os.path.abspath(partition_file))
    df = pd.read_csv(partition_file)

    for row in df.values:
        subject, partition = str(row[0]), row[-1]
        subject2partition[subject] = partition
        if partition not in partition2subject:
            partition2subject[partition] = []
        if subject not in partition2subject[partition]:
            partition2subject[partition].append(subject)

    return subject2partition, partition2subject


def get_all_training_csvs(task, feature) -> List[str]:
    """
    Loads a list of all feature csvs that are used for training a certain task
    :param task: humor, stress etc.
    :param feature: name of the feature folder (e.g. 'egemaps')
    :return: list of csvs
    """
    _, partition_to_subject = get_data_partition(PARTITION_FILES[task])
    feature_dir = os.path.join(PATH_TO_FEATURES[task], feature)
    csvs = []
    for subject in tqdm(partition_to_subject['train']):
        if task == PERCEPTION:
            csvs.append(os.path.join(feature_dir, f'{subject}.csv'))
        elif task == HUMOR:
            csvs.extend(sorted(glob(os.path.join(feature_dir, subject, "*.csv"))))

    return csvs


def fit_normalizer(task:str, feature:str, feature_idx=2) -> StandardScaler:
    """
    Fits a sklearn StandardScaler based on training data
    :param task: task
    :param feature: feature
    :param feature_idx: index in the feature csv where the features start
    (typically 2, features starting after segment_id, timestamp)
    :return: fitted sklearn.preprocessing.StandardScaler
    """
    # load training subjects
    training_csvs = get_all_training_csvs(task, feature)
    #print(f"All training csvs: {training_csvs}")
    df = pd.concat([pd.read_csv(training_csv) for training_csv in training_csvs])
    values = df.iloc[:, feature_idx:].values
    #print(f'Scaling values')
    normalizer = StandardScaler().fit(values)
    return normalizer


################# TASK-SPECIFIC LOADER METHODS FOR SINGLE SUBJECTS #############################################

# --------------------------------------  humor ---------------------------------------------------------------#

def load_humor_subject(feature, subject_id, normalizer) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Loads data for a single subject for the humor task
    :param feature: feature name
    :param subject_id: subject name
    :param normalizer: fitted StandardScaler, can be None if no normalization is desired
    :return: features, labels, metas.
        features is a list of ndarrays of shape (seg_len, feature_dim)
        labels is a ndarray of shape (len(features), 1) (label for each element in the features list)
        metas is a ndarray of shape (len(features), 1, 1+len(label columns)) (segment_id, seq_start, seq_end, segment_id)
    """
    # parse labels
    label_path = PATH_TO_LABELS[HUMOR]
    label_files = sorted(glob(os.path.join(label_path, subject_id + '/*.csv')))
    assert len(label_files) > 0, f'Error: no available humor label files for coach "{subject_id}": "{label_files}".'
    label_df = pd.concat([pd.read_csv(label_file) for label_file in label_files])

    # idx of the data frame (column) where features start
    feature_idx = 2
    feature_path = PATH_TO_FEATURES[HUMOR]

    feature_files = sorted(glob(os.path.join(feature_path, feature, subject_id + '/*.csv')))
    assert len(
        feature_files) > 0, f'Error: no available "{feature}" feature files for coach "{subject_id}": "{feature_files}".'
    feature_df = pd.concat([pd.read_csv(feature_file) for feature_file in feature_files])
    if not (normalizer is None):
        feature_values = feature_df.iloc[:, feature_idx:].values
        feature_df.iloc[:, feature_idx:] = normalizer.transform(feature_values)
    feature_dim = len(feature_df.columns) - feature_idx

    # load features for each label
    features = []
    for _, y in label_df.iterrows():
        start = y['timestamp_start']
        end = y['timestamp_end']
        segment_id = y['segment_id']
        segment_features = feature_df[feature_df.segment_id == segment_id]
        label_features = segment_features[(segment_features.timestamp >= start) &
                                          (segment_features.timestamp < end)].iloc[:, feature_idx:].values
        # imputation?
        if label_features.shape[0] == 0:
            label_features = np.zeros((1, feature_dim))
        features.append(label_features)

    # store
    # expand for compatibility with the dataset class
    labels = np.expand_dims(label_df.iloc[:, -1].values, -1)
    metas = np.expand_dims(label_df.iloc[:, :-1].values, 1)

    return features, labels, metas


# --------------------------------------  perception ---------------------------------------------------------------#


def load_perception_subject(feature, subject_id, normalizer, label_dim) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Loads data for a single subject for the perception task
    :param feature: feature name
    :param subject_id: subject name/ID
    :param normalizer: fitted StandardScaler, can be None if no normalization is desired. It is created in the load_data
    method, so no need to take care of that. It just needs to be called in the load_mimic_subject method somewhere
    to normalize the features
    :return: features, labels, metas.
        Assuming every subject consists of n segments of lengths l_1,...,l_n:
            features is a list (length n) of ndarrays of shape (l_i, feature_dim)  - each item corresponding to a segment
            labels is a ndarray of shape (n, num_classes) (labels for each element in the features list, assuming every segment
                has num_classes labels)
            metas is a ndarray of shape (n, 1, x) where x is the number of columns needed to describe the segment
                Typically something like (subject_id, segment_id, seq_start, seq_end) or the like
                They are only used to write the predictions: a prediction line consists of all the meta data associated
                    with one data point + the predicted label(s)
    """
    # parse label

    label_path = PATH_TO_LABELS[PERCEPTION]
    labels_df = pd.read_csv(label_path)
    label_values = labels_df[(labels_df.subj_id==int(subject_id))][label_dim].values
    assert len(label_values) == 1
    label = np.array([[label_values[0]]])

    feature_path = os.path.join(PATH_TO_FEATURES[PERCEPTION], feature)
    feature_df = pd.read_csv(os.path.join(feature_path, f'{subject_id}.csv'))

    any_nan = feature_df.isna().any().any()
    if any_nan:
        feature_df.dropna(inplace=True)

    feature_idx = 2
    features = feature_df.iloc[:, feature_idx:].values
    if not (normalizer is None):
        features = normalizer.transform(features)
    features = [features]

    metas = np.array([subject_id]).reshape((1, 1, 1))

    return features, label, metas


################# LOAD DATASETS USING THE SPECIFIC METHODS ABOVE #############################################

def load_data(task:str,
              paths:Dict[str, str],
              feature:str,
              label_dim: Optional[str],
              normalize: Optional[Union[bool, StandardScaler]] = True,
              save=False,
              ids: Optional[Dict[str, List[str]]] = None,
              data_file_suffix: Optional[str]=None) \
        -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    Loads the complete data sets
    :param task: task
    :param paths: dict for paths to data and partition file
    :param feature: feature to load
    :param label_dim: label dimension to load labels for - only relevant for perception task
    :param normalize: whether normalization is desired
    :param save: whether to cache the loaded data as .pickle
    :param segment_train: whether to do segmentation on the training data
    :param ids: only consider these IDs (map 'train', 'devel', 'test' to list of ids) - only relevant for personalisation
    :param data_file_suffix: optional suffix for data file, may be useful for personalisation
    :return: dict with keys 'train', 'devel' and 'test', each in turn a dict with keys:
        feature: list of ndarrays shaped (seq_length, features)
        labels: corresponding list of ndarrays shaped (seq_length, 1) for n-to-n tasks like stress, (1,) for n-to-1
            task humor, (4,) for n-to-4 task mimic
        meta: corresponding list of ndarrays shaped (seq_length, metadata_dim) where seq_length=1 for n-to-1/n-to-4
    """

    data_file_name = f'data_{task}_{feature}_{label_dim + "_" if len(label_dim) > 0 else ""}_{"norm_" if normalize else ""}' \
                     f'{f"_{data_file_suffix}" if data_file_suffix else ""}.pkl'
    data_file = os.path.join(paths['data'], data_file_name)

    if os.path.exists(data_file):  # check if file of preprocessed data exists
        print(f'Find cached data "{os.path.basename(data_file)}".')
        data = pickle.load(open(data_file, 'rb'))
        return data

    print('Constructing data from scratch ...')
    data = {'train': {'feature': [], 'label': [], 'meta': []},
            'devel': {'feature': [], 'label': [], 'meta': []},
            'test': {'feature': [], 'label': [], 'meta': []}}
    subject2partition, partition2subject = get_data_partition(paths['partition'])

    if not(normalize is None):
        if type(normalize) == bool:
                normalizer = fit_normalizer(task=task, feature=feature) if normalize else None
        else:
            normalizer = normalize
    else:
        normalizer = None
    for partition, subject_ids in partition2subject.items():
        if ids:
            subject_ids = [s for s in subject_ids if s in ids[partition]]


        for subject_id in tqdm(subject_ids):
            if task == HUMOR:
                features, labels, metas = load_humor_subject(feature=feature, subject_id=subject_id,
                                                             normalizer=normalizer)
            elif task == PERCEPTION:
                features, labels, metas = load_perception_subject(feature=feature, subject_id=subject_id,
                                                                normalizer=normalizer, label_dim=label_dim)

            data[partition]['feature'].extend(features)
            data[partition]['label'].extend(labels)
            data[partition]['meta'].extend(metas)

    if save:  # save loaded and preprocessed data
        print('Saving data...')
        pickle.dump(data, open(data_file, 'wb'))
    return data