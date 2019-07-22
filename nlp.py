import sys
import os
import pandas as pd
import numpy as np
import spacy

# absolute path to current dir
curr_dir = os.path.dirname(os.path.abspath(__file__))

# add the submodule tagger to path for importing libraries and code
sys.path.insert(1, os.path.join(curr_dir, 'tagger'))

from predictors.svm_predictor import SVMPredictor
from config import Config


def build_tagger():
    print('Build Dialogue Act Tagger')

    cfg = Config.from_json(f"{sys.path[1]}/models/Model.SVM/meta.json")

    # prepend submodule folder so the predictors can load models correctly
    cfg.out_folder = f"tagger/{cfg.out_folder}"
    cfg.corpora = f"tagger/{cfg.corpora}"

    return SVMPredictor(cfg)


def tag_dialogue_act(tagger, emo):
    print('Tag Dialogue Act')

    acts = emo['Utterance'].map(lambda s: tagger.dialogue_act_tag(s))

    # store all results
    has_acts = []
    dialogue_acts = []

    # store the most likely
    act_dimensions = []
    comm_funcs = []

    for act in acts:
        has_act = len(act) > 0
        has_acts.append(has_act)

        if has_act:
            dialogue_acts.append(str(act))
            act_dimensions.append(act[0]['dimension'])
            comm_funcs.append(act[0]['communicative_function'])
        else:
            dialogue_acts.append('[]')
            act_dimensions.append('')
            comm_funcs.append('')

    return emo.assign(
        Has_Dialogue_Acts = has_acts,
        Dialogue_Acts = dialogue_acts,
        Dialogue_Act_Dimension = act_dimensions,
        Dialogue_Act_Comm_Func = comm_funcs
    )


def extract_entities(emo):
    print('Extract Entities')

    # check if utterance mentions any entities
    nlp = spacy.load("en_core_web_sm")
    docs = list(nlp.pipe(emo['Utterance'], disable = ["tagger", "parser"]))

    return emo.assign(
        Has_Entities = [len(doc.ents) > 0 for doc in docs],
        Entities = [str([(e.text, e.label_) for e in doc.ents]) for doc in docs]
    )


# input: lines
# Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Old_Dialogue_ID,Old_Utterance_ID,Season,Episode,StartTime,EndTime
# output: each line has two new cols: speaker ID, relationship ID
def annotate_interlocutor_relationship(emo):
    print('Annotate interlocutor relationship')

    # build two maps:
    # 1) speaker to id
    # 2) listener id
    # 3) dialogue to relationship id

    # 1
    speakers = list(
        np.sort(
            emo['Speaker'].unique()
        )
    )

    # 2 + 3
    # - group by dialogue id
    # - iteratively build a dictionary of speaker ID pair, and give it ID
    # - record pair of speakers so that listener ID can be extracted later
    # - add new column map dialogue id to relationship ID
    dialogues = emo[['Speaker', 'Dialogue_ID']].drop_duplicates()
    relationship_id = 0

    # map dialogue id to pair of speakers
    dialogue_speakers = {}

    # map pair of speakers to a relationship ID
    speaker_relationships = {}

    # map dialogue id to relationship ID counter
    # to be added as a new feature
    dialogue_relationships = {}

    is_first_speaker = True
    curr_dialogue_id = None
    a_id = None
    for row in dialogues.itertuples(index = False):
        speaker = row.Speaker
        dialogue_id = row.Dialogue_ID

        speaker_idx = speakers.index(speaker)

        # compute listener id
        if curr_dialogue_id != dialogue_id:
            dialogue_speakers[dialogue_id] = list(dialogues[dialogues['Dialogue_ID'] == dialogue_id]['Speaker'])

        # compute relationship id
        if is_first_speaker:
            # store
            a_id = speaker_idx
        else:
            pair = tuple(sorted([a_id, speaker_idx]))

            # set relationship ID
            if pair not in speaker_relationships:
                speaker_relationships[pair] = relationship_id
                relationship_id += 1

            dialogue_relationships[dialogue_id] = speaker_relationships[pair]

        is_first_speaker = not is_first_speaker

    return emo.assign(
        Speaker_ID = emo['Speaker'].map(lambda s: speakers.index(s)),
        Listener_ID = emo[['Speaker', 'Dialogue_ID']].apply(
            lambda r: speakers.index([s for s in dialogue_speakers[r.Dialogue_ID] if s != r.Speaker][0]),
            axis = 1
        ),
        Relationship_ID = emo['Dialogue_ID'].map(lambda d: dialogue_relationships[d])
    )


def load_data(name = 'dev'):
    print(f"Load dataset '{name}'")
    # Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Old_Dialogue_ID,Old_Utterance_ID,Season,Episode,StartTime,EndTime
    return pd.read_csv(f'datasets/meld/{name}_sent_emo_dya.csv')


def save_data(emo, fname = 'meld_yolo.csv'):
    print(f"Save to '{fname}'")
    with open(f'{curr_dir}/datasets/{fname}', 'w') as yolo:
        emo.to_csv(yolo, index = False)
    return emo


def enrich_nlp_features(emo):
    return tag_dialogue_act(
        build_tagger(),
        extract_entities(
            annotate_interlocutor_relationship(
                emo
            )
        )
    )


def test():
    # return tag_dialogue_act(build_tagger(), load_data())
    return enrich_nlp_features(load_data('train'))


def main():
    return save_data(
        enrich_nlp_features(
            load_data('train')
        )
    )


if __name__ == '__main__':
    main()
