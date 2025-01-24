import pandas as pd
df=pd.read_csv('/data/Youss/Fact_cheking/reasoner/LLMs.csv')
def calculate_TPTNFPFN(df):
    TP=0
    TN=0
    FP=0
    FN = 0
    for idx, row in df.iterrows():
        if row['Prediction_Verdict'] == row['Label']:
            TP+=1
            TN+=1
        elif  pd.isnull(row['Prediction_Verdict']):
            print('FN')
            FN+=1
        else:
            FP+=1
    return TP, TN, FP, FN

def calculate_metrics(TP, TN, FP, FN ):
    recall= TP / (TP + FN)
    Precision= TP / (TP + FP)
    F1_score= 2*recall*Precision /(recall+Precision)
    return recall, Precision, F1_score


TP, TN, FP, FN=calculate_TPTNFPFN(df)
recall, Precision, F1_score=calculate_metrics(TP, TN, FP, FN )

print(recall, Precision, F1_score)