import pandas as pd

class DataReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_and_filter_columns(self):
        desired_columns = ['Claim', 'Label', 'Answer', 'claim_ERE', 'answer_ERE', 'Annotation']
        data = pd.read_csv(self.file_path)
        if not all(col in data.columns for col in desired_columns):
            missing_columns = [col for col in desired_columns if col not in data.columns]
            raise ValueError(f"The following required columns are missing in the file: {missing_columns}")
        return data[desired_columns]
