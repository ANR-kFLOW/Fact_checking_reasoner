import pandas as pd
import random

file_path = '/data/Youss/RE/new_data/cs_splited/cleaned_duplicates/transformed/test.csv'
df = pd.read_csv(file_path)


stratified_sample = df.groupby('relation', group_keys=False).apply(
    lambda x: x.sample(frac=0.5, random_state=42)).reset_index(drop=True)

shuffled_sample = stratified_sample.sample(frac=1, random_state=42).reset_index(drop=True)


negative_examples = []
for i in range(len(stratified_sample)):
    row1 = stratified_sample.iloc[i]
    row2 = shuffled_sample.iloc[i]
    swap_type = random.choice(['subject', 'object'])

    if swap_type == 'subject':
        new_sub = row2['sub']
        new_obj = row1['obj']
    elif swap_type == 'object':
        new_sub = row1['sub']
        new_obj = row2['obj']


    negative_examples.append({
        'corpus': row1['corpus'],
        'doc_id': row1['doc_id'],
        'sent_id': row1['sent_id'],
        'eg_id': row1['eg_id'],
        'index': row1['index'],
        'text': row1['text'],
        'sub': new_sub,
        'obj': new_obj,
        'relation': f'negative_{row1["relation"]}'
    })

negative_df = pd.DataFrame(negative_examples)


negative_output_path = '/data/Youss/RE/new_data/cs_splited/cleaned_duplicates/transformed/test_negatives.csv'
negative_df.to_csv(negative_output_path, index=False)

print(f"Negative examples saved to {negative_output_path}")
