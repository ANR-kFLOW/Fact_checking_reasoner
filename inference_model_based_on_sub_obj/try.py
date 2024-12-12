import pandas as pd

# File paths
original_file_path = '/data/Youss/RE/new_data/cs_splited/cleaned_duplicates/transformed/dev.csv'
negative_file_path = '/data/Youss/RE/new_data/cs_splited/cleaned_duplicates/transformed/dev_negatives.csv'

# Load the original and negative examples
original_df = pd.read_csv(original_file_path)
negative_df = pd.read_csv(negative_file_path)
negative_df['relation'] = 0
# Concatenate the DataFrames
combined_df = pd.concat([original_df, negative_df], ignore_index=True)

# Save the combined DataFrame to a new file
output_path = '/data/Youss/RE/new_data/cs_splited/cleaned_duplicates/transformed/dev.csv'
combined_df.to_csv(output_path, index=False)

print(f"Combined file saved to {output_path}")
