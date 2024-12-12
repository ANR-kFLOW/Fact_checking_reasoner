import pandas as pd
df=pd.read_csv('/data/Youss/Fact_cheking/reasoner/use_cases.csv')
def adjust_triples(df, column):
    def fix_format(triple):
        # Remove parentheses and split by comma
        elements = triple.strip("()").split(", ")

        # If correctly formatted, return as-is
        if len(elements) == 3:
            return triple

        # If misformatted, identify the relation
        relations = {'cause', 'enable', 'intend', 'prevent'}
        for i, elem in enumerate(elements):
            if elem in relations:
                relation = elem
                # Join the subject (before relation) and object (after relation) by space
                subject = " ".join(elements[:i])
                obj = " ".join(elements[i + 1:])
                return f"({subject}, {relation}, {obj})"

        # Return original if no relation is found (edge case)
        return triple

    # Apply the fix to the specified column
    df[column] = df[column].apply(fix_format)
    return df


# Adjust the DataFrame
adjusted_df = adjust_triples(df, 'answer_ERE')

adjusted_df.to_csv('/data/Youss/Fact_cheking/reasoner/use_cases_fixed.csv', index=False)
