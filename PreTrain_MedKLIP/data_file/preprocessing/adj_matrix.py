"""
Code copied from AGXNet:
https://github.com/batmanlab/AGXNet
"""

"""Create adjacency matrix for representing the relations between anatomical landmarks and observations."""

import argparse
import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser(description='Create Adjacency matrix Matrix.')

parser.add_argument('--input-path', default='/PROJECT DIR/preprocessing/mimic-cxr-radgraph-sentence-parsed.csv',
                    help='Itemized input data path.'
                    )

# List of most common normal observations
NORM_OBS = ['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free']

# exclude
EXCLUDED_OBS = ['none', 'unchanged', 'change', 'great', 'similar', 'large', 'small', 'moderate', 'mild',
                'median', 'decrease', 'bad', 'more', 'constant', 'worsen', 'new', 'improve',
                'status', 'position', 'sternotomy', 'cabg', 'replacement', 'postoperative', 'assessment',
                'patient']

# top 90% abnormal observations
ABNORM_OBS = ['effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process',
              'abnormality', 'enlarge', 'tip', 'low', 'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly',
              'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence', 'device',
              'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative',
              'pacemaker', 'thicken', 'marking', 'scar', 'hyperinflate', 'blunt', 'loss', 'widen', 'collapse',
              'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
              'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware',
              'dilation', 'chf', 'redistribution', 'aspiration']

# final row and column names in adjacent matrix
LANDMARK_NAME = ['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural', 'right_pleural',
                 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm', 'right_diaphragm',
                 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe', 'lower_right_lobe',
                 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
                 'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung',
                 'right_apical_lung', 'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic',
                 'right_costophrenic', 'costophrenic_unspec', 'cardiophrenic_sulcus', 'mediastinal', 'spine',
                 'clavicle', 'rib', 'stomach', 'right_atrium', 'right_ventricle', 'aorta', 'svc', 'interstitium',
                 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes',
                 'unspecified', 'other']

OBSERVATION_CLASS = ['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free', 'effusion',
                     'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process',
                     'abnormality', 'enlarge', 'tip', 'low', 'pneumonia', 'line', 'congestion', 'catheter',
                     'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification',
                     'prominence', 'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire',
                     'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar', 'hyperinflate', 'blunt',
                     'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate',
                     'obscure', 'deformity', 'hernia', 'drainage', 'distention', 'shift', 'stent', 'pressure',
                     'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution',
                     'aspiration', 'tail_abnorm_obs', 'excluded_obs']

DICT_ANATOMICAL_LANDMARKS = {
    'trachea': {'a': ['trachea', 'tracheal'], 'm1': [], 'm2': [], 'sc': [], 't': 'm0'},
    'left_hilar': {'a': ['hilar', 'hilum', 'perihilar', 'infrahilar'], 'm1': ['left'], 'm2': ['right'],
                   'sc': [],
                   't': 'm1+m2-'},
    'right_hilar': {'a': ['hilar', 'hilum', 'perihilar', 'infrahilar'], 'm1': ['right'], 'm2': ['left'],
                    'sc': [],
                    't': 'm1+m2-'},
    'hilar_unspec': {'a': ['hilar', 'hilum', 'perihilar', 'infrahilar'], 'm1': ['left', 'right'], 'm2': [],
                     'sc': ['hila', 'perihilar|right|left', 'perihilar|left|right'], 't': 'm1-'},
    'left_pleural': {'a': ['pleural'], 'm1': ['left'], 'm2': ['right'], 'sc': [], 't': 'm1+m2-'},
    'right_pleural': {'a': ['pleural'], 'm1': ['right'], 'm2': ['left'], 'sc': [], 't': 'm1+m2-'},
    'pleural_unspec': {'a': ['pleural'], 'm1': ['left', 'right'], 'm2': [],
                       'sc': ['pleural|left|right', 'pleural|right|left', 'pleural|bilateral|right|left',
                              'pleural|bilateral|left|right'], 't': 'm1-'},
    'heart_size': {'a': ['heart', 'cardiac'], 'm1': ['border', 'borders'], 'm2': [], 'sc': [], 't': 'm1-'},
    'heart_border': {'a': ['heart', 'cardiac'], 'm1': ['border', 'borders'], 'm2': [], 'sc': [], 't': 'm1+'},
    'left_diaphragm': {'a': ['diaphragm', 'hemidiaphragm'], 'm1': ['left'], 'm2': ['right'], 'sc': [],
                       't': 'm1+m2-'},
    'right_diaphragm': {'a': ['diaphragm', 'hemidiaphragm'], 'm1': ['right'], 'm2': ['left'], 'sc': [],
                        't': 'm1+m2-'},
    'diaphragm_unspec': {'a': ['diaphragm', 'diaphragms', 'hemidiaphragms', 'hemidiaphragm'],
                         'm1': ['left', 'right'], 'm2': [],
                         'sc': ['hemidiaphragm|left|right', 'hemidiaphragm|right|left'], 't': 'm1-'},
    'retrocardiac': {'a': ['retrocardiac'], 'm1': [], 'm2': [], 'sc': [], 't': 'm0'},
    'lower_left_lobe': {'a': ['lobe'], 'm1': ['left'], 'm2': ['lower'], 'sc': [], 't': 'm1+m2+'},
    'upper_left_lobe': {'a': ['lobe'], 'm1': ['left'], 'm2': ['upper'], 'sc': ['lingula', 'lingular'],
                        't': 'm1+m2+'},
    'lower_right_lobe': {'a': ['lobe'], 'm1': ['right'], 'm2': ['lower'], 'sc': [], 't': 'm1+m2+'},
    'middle_right_lobe': {'a': ['lobe'], 'm1': ['right'], 'm2': ['middle'], 'sc': [], 't': 'm1+m2+'},
    'upper_right_lobe': {'a': ['lobe'], 'm1': ['right'], 'm2': ['upper'], 'sc': [], 't': 'm1+m2+'},
    'left_lower_lung': {'a': ['lung'], 'm1': ['left'], 'm2': ['lower', 'base', 'basilar', 'basal', 'basis'],
                        'sc': ['base|left', 'basilar|left', 'basal|left', 'lung|left|bases'], 't': 'm1+m2+'},
    'left_mid_lung': {'a': ['lung'], 'm1': ['left'], 'm2': ['middle', 'mid'], 'sc': ['midlung|left'],
                      't': 'm1+m2+'},
    'left_upper_lung': {'a': ['lung'], 'm1': ['left'], 'm2': ['upper'], 'sc': [], 't': 'm1+m2+'},
    'left_apical_lung': {'a': ['apex', 'apical', 'apical', 'apicolateral'], 'm1': ['left'], 'm2': ['right'],
                         'sc': [], 't': 'm1+m2-'},
    'left_lung_unspec': {'a': ['lung', 'hemithorax'], 'm1': ['left', 'left-sided'],
                         'm2': ['volume', 'volumes', 'right', 'lower', 'base', 'bases', 'basilar', 'basilar',
                                'basal', 'basis', 'middle', 'mid', 'upper', 'apex', 'apical', 'perihilar'],
                         'sc': ['left', 'left side', 'thorax|left|hemi'], 't': 'm1+m2-'},
    'right_lower_lung': {'a': ['lung'], 'm1': ['right'], 'm2': ['lower', 'base', 'basilar', 'basal', 'basis'],
                         'sc': ['base|right', 'basilar|right', 'basal|right', 'lung|right|bases'],
                         't': 'm1+m2+'},
    'right_mid_lung': {'a': ['lung'], 'm1': ['right'], 'm2': ['middle', 'mid'], 'sc': [], 't': 'm1+m2+'},
    'right_upper_lung': {'a': ['lung'], 'm1': ['right'], 'm2': ['upper'], 'sc': [], 't': 'm1+m2+'},
    'right_apical_lung': {'a': ['apex', 'apical', 'apical', 'apicolateral'], 'm1': ['right'], 'm2': ['left'],
                          'sc': [], 't': 'm1+m2-'},
    'right_lung_unspec': {'a': ['lung', 'hemithorax'], 'm1': ['right', 'right-sided'],
                          'm2': ['volume', 'volumes', 'left', 'lower', 'base', 'bases', 'basilar', 'basilar',
                                 'basal', 'basis', 'middle', 'mid', 'upper', 'apex', 'apical', 'perihilar'],
                          'sc': ['right', 'right side', 'thorax|right|hemi'], 't': 'm1+m2-'},
    'lung_apices': {'a': ['apices', 'apical'], 'm1': ['left', 'right'], 'm2': [],
                    'sc': ['biapical', 'lungs|upper'],
                    't': 'm1-'},
    'lung_bases': {'a': ['lung', 'lungs'], 'm1': ['left', 'right'],
                   'm2': ['bibasilar', 'basilar', 'base', 'bases', 'bibasal', 'basal'],
                   'sc': ['lung|lower', 'lungs|lower', 'bibasilar', 'basilar', 'bases', 'bibasal', 'basal',
                          'basal|bilateral', 'lobe|lower', 'lobes|lower', 'lobe|bilateral|lower', 'bases|both',
                          'bibasilar|left|right', 'bibasilar|right|left'], 't': 'm1-m2+'},
    'left_costophrenic': {'a': ['costophrenic'], 'm1': ['left'], 'm2': ['right'], 'sc': [], 't': 'm1+m2-'},
    'right_costophrenic': {'a': ['costophrenic'], 'm1': ['right'], 'm2': ['left'], 'sc': [], 't': 'm1+m2-'},
    'costophrenic_unspec': {'a': ['costophrenic'], 'm1': ['left', 'right'], 'm2': [], 'sc': [], 't': 'm1-'},
    'cardiophrenic_sulcus': {'a': ['cardiophrenic'], 'm1': [], 'm2': [], 'sc': [], 't': 'm0'},
    'mediastinal': {'a': ['mediastinal', 'cardiomediastinal', 'mediastinum', 'cardiomediastinum'], 'm1': [],
                    'm2': [], 'sc': [], 't': 'm0'},
    'spine': {'a': ['spine', 'spinal'], 'm1': [], 'm2': [], 'sc': [], 't': 'm0'},
    'clavicle': {'a': ['clavicle', 'clavicles'], 'm1': [], 'm2': [], 'sc': [], 't': 'm0'},
    'rib': {'a': ['rib', 'ribs'], 'm1': [], 'm2': [], 'sc': [], 't': 'm0'},
    'stomach': {'a': ['stomach', 'abdomen', 'abdominal'], 'm1': [], 'm2': [], 'sc': [], 't': 'm0'},
    'right_atrium': {'a': ['atrium', 'atrial'], 'm1': ['right'], 'm2': ['left'], 'sc': [], 't': 'm1+m2-'},
    'right_ventricle': {'a': ['ventricle', 'ventricular'], 'm1': ['right'], 'm2': ['left'], 'sc': [],
                        't': 'm1+m2-'},
    'aorta': {'a': ['aorta', 'aortic'], 'm1': [], 'm2': [], 'sc': [], 't': 'm0'},
    'svc': {'a': ['svc'], 'm1': [], 'm2': [], 'sc': [], 't': 'm0'},
    'interstitium': {'a': ['interstitium', 'interstitial'], 'm1': [], 'm2': [], 'sc': [], 't': 'm0'},
    'parenchymal': {'a': ['parenchymal'], 'm1': [], 'm2': [], 'sc': [], 't': 'm0'},
    'cavoatrial_junction': {'a': ['cavoatrial junction'], 'm1': [], 'm2': [], 'sc': [], 't': 'm0'},
    'cardiopulmonary': {'a': ['cardiopulmonary'], 'm1': [], 'm2': [], 'sc': [], 't': 'm0'},
    'pulmonary': {'a': ['pulmonary'], 'm1': [], 'm2': [], 'sc': [], 't': 'm0'},
    'lung_volumes': {'a': ['lungs', 'lung', 'volume', 'volumes'],
                     'm1': ['left', 'right', 'lower', 'base', 'bases', 'basilar', 'basal', 'basis', 'middle',
                            'mid',
                            'upper', 'apex', 'apical', 'apical'], 'm2': [], 'sc': [], 't': 'm1-'}
}
class LandmarkObservationAdjacentMatrix(Dataset):
    def __init__(self, LANDMARK_NAME, OBSERVATION_CLASS, df_anatomy_label):
        self.LANDMARK_NAME = LANDMARK_NAME
        self.OBSERVATION_CLASS = OBSERVATION_CLASS
        self.df_anatomy_label = df_anatomy_label

        # get all study ids
        self.sids = list(self.df_anatomy_label['study_id'].unique())

    def __getitem__(self, idx):
        sid = self.sids[idx]
        df_sid = self.df_anatomy_label[self.df_anatomy_label['study_id'] == sid]
        landmark_observation_adj_mtx = np.zeros((len(LANDMARK_NAME), len(OBSERVATION_CLASS))) - 1.0
        for index, row in df_sid.iterrows():
            try:
                observation_idx = self.OBSERVATION_CLASS.index(
                    row.obs_lemma_grp)  # if a rare observation, skip this instance
                landmark_idx = self.LANDMARK_NAME.index(row.landmark_name)

                curr_val = landmark_observation_adj_mtx[landmark_idx, observation_idx]

                # for obs_lemma_grp, such as tail_abnorm_obs
                # if one observation is DP, then 1.0
                if row.label == 'OBS-DP':
                    landmark_observation_adj_mtx[landmark_idx, observation_idx] = 1.0
                elif row.label == 'OBS-DA':
                    landmark_observation_adj_mtx[landmark_idx, observation_idx] = np.maximum(curr_val, 0.0)
            except:
                pass
        return sid, landmark_observation_adj_mtx

    def __len__(self):
        return len(self.sids)

def anatomy_to_landmark(x, a, m1=[], m2=[], sc=[], t='m0'):
    """
    Args:
        x: input anatomy, e.g., 'lobe|left|lower'
        a: base anatomy set, e.g., ['hilar', 'hilum', 'perihilar']
        m1: level 1 modifier, e.g., ['left', 'right']
        m2: level 2 modifier, e.g., ['upper', 'middle', 'lower']
        s: special cases, e.g., ['chest']
        t: type, ['m2+', 'm1+m2-']
    Return:
        flag: boolean, matched or not matched
    """
    s = set(x.split('|'))
    if t == 'm1+m2+':
        flag = (len(s & set(a)) > 0) & (len(s & set(m1)) > 0) & (len(s & set(m2)) > 0)
    elif t == 'm1+m2-':
        flag = (len(s & set(a)) > 0) & (len(s & set(m1)) > 0) & (len(s & set(m2)) == 0)
    elif t == 'm1-m2+':
        flag = (len(s & set(a)) > 0) & (len(s & set(m1)) == 0) & (len(s & set(m2)) > 0)
    elif t == 'm1-m2-':
        flag = (len(s & set(a)) > 0) & (len(s & set(m1)) == 0) & (len(s & set(m2)) == 0)
    elif t == 'm1+':
        flag = (len(s & set(a)) > 0) & (len(s & set(m1)) > 0)
    elif t == 'm2+':
        flag = (len(s & set(a)) > 0) & (len(s & set(m2)) > 0)
    elif t == 'm1-':
        flag = (len(s & set(a)) > 0) & (len(s & set(m1)) == 0)
    elif t == 'm2-':
        flag = (len(s & set(a)) > 0) & (len(s & set(m2)) == 0)
    elif t == 'm0':
        flag = (len(s & set(a)) > 0)

    if sc:
        flag = flag | (x in sc)
    return flag

def create_adj_matrix(args):
    # load anatomy label table, text table and master table
    print('Loading parsed RadGraph data...')
    df_anatomy_label = pd.read_csv(args.input_path, dtype=str)

    # manual lemmatization correction
    idx_replace = df_anatomy_label['obs_lemma'].isin(['enlargement', 'increase'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'enlarge'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['engorge'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'engorgement'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['opacification', 'opacity-'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'opacity'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['calcify'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'calcification'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['effusion ;'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'effusion'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['atelectatic', 'atelectasis ;', 'atelectase'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'atelectasis'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['aeration'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'aerate'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['distend', 'distension'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'distention'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['wide'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'widen'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['prominent'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'prominence'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['haze'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'haziness'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['masse'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'mass'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['kyphotic'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'kyphosis'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['degenerate'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'degenerative'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['obscuration'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'obscure'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['fibrotic'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'fibrosis'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['nodular', 'nodularity'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'nodule'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['ventilate'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'ventilation'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['tortuosity'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'tortuous'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['elongate'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'elongation'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['elevate'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'elevation'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['drain'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'drainage'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['deviate'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'deviation'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['consolidative', 'consolidate'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'consolidation'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['dilate', 'dilatation'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'dilation'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['hydropneumothorax', 'pneumothoraces', 'pneumothorace'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'pneumothorax'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['improvement', 'improved'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'improve'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['can not be assess', 'can not be evaluate', 'not well see',
                                                      'not well assess', 'can not be accurately assess',
                                                      'not well evaluate', 'not well visualize',
                                                      'difficult to evaluate',
                                                      'poorly see'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'difficult to assess'

    idx_replace = df_anatomy_label['obs_lemma'] == 'pacer'
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'pacemaker'

    idx_replace = df_anatomy_label['obs_lemma'].isin(['infection', 'infectious', 'infectious process'])
    df_anatomy_label.loc[idx_replace, 'obs_lemma'] = 'pneumonia'

    df_anatomy_label.loc[df_anatomy_label['label'].isna(), 'label'] = 'OBS-NA'

    # step 1: map anatomy name to landmark name
    landmark_name = []
    for index, row in tqdm(df_anatomy_label.iterrows(), total=df_anatomy_label.shape[0]):
        x = row.anatomy
        flag = False
        for k, v in DICT_ANATOMICAL_LANDMARKS.items():
            flag = anatomy_to_landmark(x, v['a'], v['m1'], v['m2'], v['sc'], v['t'])
            if flag:
                landmark_name.append(k)
                break
        if (not flag) & (row.anatomy == 'unspecified'):
            landmark_name.append('unspecified')
        elif (not flag) & (row.anatomy != 'unspecified'):
            landmark_name.append('other')

    df_anatomy_label['landmark_name'] = landmark_name

    # create a new obs_lemma column to grouop other abnormal observation class
    df_anatomy_label['obs_lemma_grp'] = df_anatomy_label['obs_lemma']

    idx1 = df_anatomy_label['obs_lemma'].isin(NORM_OBS)
    idx2 = df_anatomy_label['obs_lemma'].isin(ABNORM_OBS)
    idx3 = df_anatomy_label['obs_lemma'].isin(EXCLUDED_OBS)

    df_anatomy_label.loc[idx3, 'obs_lemma_grp'] = 'excluded_obs'

    idx = (~idx1) & (~idx2) & (~idx3)  # abnormal observations that are in the tail
    df_anatomy_label.loc[idx, 'obs_lemma_grp'] = 'tail_abnorm_obs'

    # step 2: get landmark - observation adjacent matrix
    dataset = LandmarkObservationAdjacentMatrix(LANDMARK_NAME, OBSERVATION_CLASS, df_anatomy_label)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=False)

    sid_lst = []
    adj_mtx_lst = []
    for index, data in tqdm(enumerate(loader), total=len(loader)):
        sid, landmark_observation_adj_mtx = data
        sid_lst.append(sid)
        adj_mtx_lst.append(landmark_observation_adj_mtx)

    # step 3: convert outputs to a dictionary and then save to a pickel file
    full_sids = np.concatenate(sid_lst, axis=0)
    full_adj_mtx = np.concatenate(adj_mtx_lst, axis=0)
    dict_adj_mtx = {}
    for i in trange(len(full_sids)):
        sid = full_sids[i]
        dict_adj_mtx[sid] = full_adj_mtx[i]

    np.save('landmark_observation_sids.npy', full_sids)
    print('landmark_observation_sids.npy has been saved!')
    np.save('landmark_observation_adj_mtx.npy', full_adj_mtx)
    print('landmark_observation_sids.npy has been saved!')



if __name__ == '__main__':
    args = parser.parse_args()
    create_adj_matrix(args)