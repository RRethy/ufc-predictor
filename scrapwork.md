label_cols = ['finish', 'finish_details', 'finish_round', 'finish_round_time']
irrelevant_cols = ['total_fight_time_secs', 'R_fighter', 'B_fighter', 'R_odds', 'B_odds', 'R_ev', 'B_ev', 'title_bout', 'empty_arena']

def features(examples):
    return one_hot_encode_categorical(integer_encode_categorical(examples.drop(columns=label_cols+irrelevant_cols)))

def labels(examples):
    return examples[set(label_cols)]

def preprocess(examples):
    return features(examples), labels(examples)

X, Y = preprocess(df)
df.columns
