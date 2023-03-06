"""
Code copied from AGXNet:
https://github.com/batmanlab/AGXNet
"""

import argparse
import pandas as pd
from tqdm import tqdm
import spacy
sp = spacy.load('en_core_web_sm')

parser = argparse.ArgumentParser(description='Pharse RadGraph Relations.')

parser.add_argument('--input-path', default='/PROJECT DIR/preprocessing/mimic-cxr-radgraph-itemized.csv',
                    help='Itemized input data path.'
                    )
parser.add_argument('--output-path', default='/PROJECT DIR/preprocessing/mimic-cxr-radgraph-sentence-parsed.csv',
                    help='Output path for parsed relations.')


def obs_lemmatization(x):
    """
    Lemmatize observation
    Args:
        x: a observation token
    Return:
        normalized observation
    """
    w_lst = []
    for word in sp(str(x)):
        w_lst.append(word.lemma_)
    return ' '.join(w_lst)


def radgraph_parse(args):
    """Pharse RadGraph relations."""

    print('Loading itemized RadGraph data...')
    df_itemized = pd.read_csv(args.input_path)

    # get all study_id
    sid_lst = list(df_itemized['study_id'].unique())

    tuple_lst = []
    print('Preprocessing sentences...')
    for sid in tqdm(sid_lst):
        idx_s = df_itemized['study_id'] == sid
        df_sid = df_itemized[idx_s]

        # unique sentence index
        sen_ids = list(df_sid['sentence_ix'].unique())

        for si in sen_ids:
            idx_sen = df_sid['sentence_ix'] == si
            df_sen = df_sid[idx_sen]
            sen = df_sen['sentence'].iloc[0]

            # step 1, select all target anatomy entities (e.g., lobe) with label = ANAT-DP and target = NaN
            idx_a = (df_sen['label'] == 'ANAT-DP') & (df_sen['target'].isnull())
            df_a = df_sen[idx_a]

            if sum(idx_a) > 0:
                for _, row_a in df_a.iterrows():
                    anatomy_source_keys = []
                    sen = row_a.sentence
                    source_key = row_a.source

                    # step 2, get detailed target anatomy (e.g., lower left lobe)
                    token_a = [row_a['token'].lower()]
                    anatomy_source_keys.append(source_key)
                    idx_t = (df_sen['label'] == 'ANAT-DP') & (df_sen['target'] == source_key)
                    if sum(idx_t) > 0:
                        df_t = df_sen[idx_t]
                        for _, row in df_t.iterrows():
                            token_a += [row['token'].lower()]
                            anatomy_source_keys.append(
                                row['source'])  # save keys of all anatomy token, i.e., lower, left, lobe
                        anatomy = '|'.join(token_a)

                    else:
                        anatomy = row_a['token'].lower()

                    # step 3: get observations associated with the target anatomy (e.g., normal, effusion)
                    idx_o = (df_sen['label'].isin(['OBS-DA', 'OBS-DP', 'OBS-U'])) & (
                        df_sen['target'].isin(anatomy_source_keys)) & (df_sen['relation'] == 'located_at')
                    if sum(idx_o) > 0:
                        df_o = df_sen[idx_o]

                        anatomy_lst = []
                        obs_lst = []
                        label_lst = []
                        obs_modify_lst = []
                        obs_suggestive_lst = []

                        for _, row_o in df_o.iterrows():
                            anatomy_lst.append(anatomy)
                            obs_lst.append(row_o['token'].lower())
                            label_lst.append(row_o['label'])

                            # step 4: get obs modification
                            idx_o_m = (df_sen['target'] == row_o.source) & (df_sen['relation'] == 'modify')
                            obs_modify = None
                            if sum(idx_o_m) > 0:
                                df_o_m = df_sen[idx_o_m]
                                temp_lst = []
                                for _, row_om in df_o_m.iterrows():
                                    # if the modification is present
                                    if row_om.label == 'OBS-DP':
                                        temp_lst.append(row_om['token'].lower())
                                if len(temp_lst) > 0:
                                    obs_modify = '|'.join(temp_lst)
                            obs_modify_lst.append(obs_modify)

                            # step 5: get suggestive of obs
                            idx_o_s = (df_sen['target'] == row_o.source) & (df_sen['relation'] == 'suggestive_of')
                            obs_suggestive = None
                            if sum(idx_o_s) > 0:
                                df_o_s = df_sen[idx_o_s]
                                temp_lst = []
                                for _, row_os in df_o_s.iterrows():
                                    # if the modification is present
                                    if row_os.label == 'OBS-DP':
                                        temp_lst.append(row_os['token'].lower())
                                if len(temp_lst) > 0:
                                    obs_suggestive = '|'.join(temp_lst)
                            obs_suggestive_lst.append(obs_suggestive)

                    else:
                        anatomy_lst = [anatomy]
                        obs_lst = [None]
                        label_lst = [None]
                        obs_modify_lst = [None]
                        obs_suggestive_lst = [None]

                    # step 4: get observations that are not associated with the target anatomy
                    idx_oo = (df_sen['label'].isin(['OBS-DA', 'OBS-DP', 'OBS-U'])) & (df_sen['target'].isna()) & (
                        df_sen['relation'].isna())
                    if sum(idx_oo) > 0:
                        df_oo = df_sen[idx_oo]
                        for _, row_oo in df_oo.iterrows():
                            anatomy_lst.append('unspecified')
                            obs_lst.append(row_oo['token'].lower())
                            label_lst.append(row_oo['label'])
                            # obs_modify_lst.append(None)
                            # obs_suggestive_lst.append(None)

                            # step 5: get obs modification
                            idx_o_m = (df_sen['target'] == row_oo.source) & (df_sen['relation'] == 'modify')
                            obs_modify = None
                            if sum(idx_o_m) > 0:
                                df_o_m = df_sen[idx_o_m]
                                temp_lst = []
                                for _, row_om in df_o_m.iterrows():
                                    # if the modification is present
                                    if row_om.label == 'OBS-DP':
                                        temp_lst.append(row_om['token'].lower())
                                if len(temp_lst) > 0:
                                    obs_modify = '|'.join(temp_lst)
                            obs_modify_lst.append(obs_modify)

                            # step 5: get suggestive of obs
                            idx_o_s = (df_sen['target'] == row_oo.source) & (df_sen['relation'] == 'suggestive_of')
                            obs_suggestive = None
                            if sum(idx_o_s) > 0:
                                df_o_s = df_sen[idx_o_s]
                                temp_lst = []
                                for _, row_os in df_o_s.iterrows():
                                    # if the modification is present
                                    if row_os.label == 'OBS-DP':
                                        temp_lst.append(row_os['token'].lower())
                                if len(temp_lst) > 0:
                                    obs_suggestive = '|'.join(temp_lst)
                            obs_suggestive_lst.append(obs_suggestive)

                    # step 6: create tuple of 7 values (sid, sentence_id, sentence, anatomy, obs, label)
                    t_lst = []
                    for i in range(len(obs_lst)):
                        t_lst.append(
                            (sid, si, sen, anatomy_lst[i], obs_lst[i], label_lst[i], obs_modify_lst[i],
                             obs_suggestive_lst[i]))

                    # remove duplicates caused by 1 obs "located_at" multiple anatomies
                    tuple_lst.append(list(set(t_lst)))

            # if the sentence does not have any ANATOMY token
            else:
                idx_o = (df_sen['label'].isin(['OBS-DA', 'OBS-DP', 'OBS-U'])) & (df_sen['target'].isnull())
                if sum(idx_o) > 0:
                    df_o = df_sen[idx_o]

                    obs_lst = []
                    label_lst = []
                    obs_modify_lst = []
                    obs_suggestive_lst = []

                    for _, row_o in df_o.iterrows():
                        obs_lst.append(row_o['token'].lower())
                        label_lst.append(row_o['label'])

                        # step 4: get obs modification
                        idx_o_m = (df_sen['target'] == row_o.source) & (df_sen['relation'] == 'modify')
                        obs_modify = None
                        if sum(idx_o_m) > 0:
                            df_o_m = df_sen[idx_o_m]
                            temp_lst = []
                            for _, row_om in df_o_m.iterrows():
                                # if the modification is present
                                if row_om.label == 'OBS-DP':
                                    temp_lst.append(row_om['token'].lower())
                            if len(temp_lst) > 0:
                                obs_modify = '|'.join(temp_lst)
                        obs_modify_lst.append(obs_modify)

                        # step 5: get suggestive of obs
                        idx_o_s = (df_sen['target'] == row_o.source) & (df_sen['relation'] == 'suggestive_of')
                        obs_suggestive = None
                        if sum(idx_o_s) > 0:
                            df_o_s = df_sen[idx_o_s]
                            temp_lst = []
                            for _, row_os in df_o_s.iterrows():
                                # if the modification is present
                                if row_os.label == 'OBS-DP':
                                    temp_lst.append(row_os['token'].lower())
                            if len(temp_lst) > 0:
                                obs_suggestive = '|'.join(temp_lst)
                        obs_suggestive_lst.append(obs_suggestive)
                else:
                    obs_lst = [None]
                    label_lst = [None]
                    obs_modify_lst = [None]
                    obs_suggestive_lst = [None]

                # step 6: create tuple of 7 values (sid, sentence_id, sentence, anatomy, obs, label)
                t_lst = []
                for i in range(len(obs_lst)):
                    t_lst.append(
                        (sid, si, sen, 'unspecified', obs_lst[i], label_lst[i], obs_modify_lst[i],
                         obs_suggestive_lst[i]))

                # remove duplicates if existing
                tuple_lst.append(list(set(t_lst)))

    # flatten nested list
    df_lst = [item for sublist in tuple_lst for item in sublist]
    df_anatomy_label = pd.DataFrame(df_lst,
                                    columns=['study_id', 'sen_id', 'sentence', 'anatomy', 'observation', 'label',
                                             'obs_modify', 'obs_suggestive'])

    # lemmatize observation tokens (e.g., normalize opacities to opacity)
    obs_lemma_lst = []
    print('Lemmatizing observation tokens...')
    for t in tqdm(df_lst):
        obs = t[4]
        obs_lemma = obs_lemmatization(obs)
        obs_lemma_lst.append(obs_lemma)

    # save preprocessed sentence level data
    df_anatomy_label['obs_lemma'] = obs_lemma_lst
    df_anatomy_label.to_csv(args.output_path, index=False)
    print('Output file has been saved!')

if __name__ == '__main__':
    args = parser.parse_args()
    radgraph_parse(args)