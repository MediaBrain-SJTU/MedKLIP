"""
Code copied from AGXNet:
https://github.com/batmanlab/AGXNet
"""

import argparse
import pandas as pd
import json
from tqdm import tqdm
import nltk


parser = argparse.ArgumentParser(description='Itemize RadGraph Dataset.')

parser.add_argument('--data-path', default='/PATH TO RADGRAPH DATA/RadGraph/physionet.org/files/radgraph/1.0.0/MIMIC-CXR_graphs.json',
                    help='RadGraph data path.'
                    )
parser.add_argument('--output-path', default='/PROJECT DIR/preprocessing/mimic-cxr-radgraph-itemized.csv',
                    help='Output path for itemized RadGraph data.')


def get_ids(key):
    """Convert keys in the RadGraph file into IDs"""
    lst = key.split('/')
    partition = lst[0] # dataset partition
    pid = lst[1][1:] # patient id
    sid = lst[2].split('.')[0][1:] # study id, remove .txt
    return partition, pid, sid


def get_sen_from_token_ix(text, ix):
    """get the sentence to which the input token index belongs."""
    sen_lst = nltk.sent_tokenize(text)
    dict_ws = {}
    ix_w = 0
    ix_s = 0
    for s in sen_lst:
        words = nltk.word_tokenize(s)
        for w in words:
            dict_ws[ix_w] = ix_s
            ix_w += 1
        ix_s += 1
    return dict_ws[ix], sen_lst[dict_ws[ix]]


def get_entity_relation(value):
    """itemize each relation"""
    source_lst = []
    target_lst = []
    token_lst = []
    token_ix_lst = []
    label_lst = []
    relation_lst = []
    sen_lst = []
    sen_ix_lst = []

    text = value['text']

    entities = value['entities']
    for k, v in entities.items():
        six, sen = get_sen_from_token_ix(text, v['start_ix'])
        relations = v['relations']

        # source node has no out going edge
        if (len(relations) == 0) or (relations[0] is None):
            source_lst.append(k)
            token_ix_lst.append(v['start_ix'])
            token_lst.append(v['tokens'])
            label_lst.append(v['label'])
            relation_lst.append(None)
            target_lst.append(None)
            sen_ix_lst.append(six)
            sen_lst.append(sen)
        else:
            for r in relations:
                source_lst.append(k)
                token_ix_lst.append(v['start_ix'])
                token_lst.append(v['tokens'])
                label_lst.append(v['label'])
                relation_lst.append(r[0])
                target_lst.append(r[1])
                sen_ix_lst.append(six)
                sen_lst.append(sen)

    # save outputs in a dataframe
    return pd.DataFrame({'source': source_lst, 'token': token_lst, 'token_ix': token_ix_lst, 'label': label_lst,
                       'relation': relation_lst, 'target': target_lst, 'sentence_ix': sen_ix_lst, 'sentence': sen_lst})


def radgraph_itemize(args):
    """Convert nested RadGraph data to itemized examples."""

    print('Loading RadGraph data...')
    f = open(args.data_path)
    data = json.load(f)
    print('RadGraph data is loaded.')

    # create itemized RadGraph data
    df_lst = []
    pid_lst = []
    sid_lst = []
    text_lst = []
    print('Itemizing RadGraph data...')
    for key, value in tqdm(data.items()):
        _, pid, sid = get_ids(key)
        pid_lst.append(pid)
        sid_lst.append(sid)
        text_lst.append(data[key]['text'])
        df = get_entity_relation(value)
        df['subject_id'] = pid
        df['study_id'] = sid
        df_lst.append(df)

    # entity level dataframe
    df_itemized = pd.concat(df_lst)

    # save dataframes to a .csv file
    df_itemized.to_csv(args.output_path, index=False)
    print('Outputs have been saved!')

if __name__ == '__main__':
    args = parser.parse_args()
    radgraph_itemize(args)


