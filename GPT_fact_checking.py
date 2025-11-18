import os
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sklearn.metrics import classification_report, f1_score
from collections import Counter

df = pd.read_csv(r"")

def most_common_label(labels):
    return Counter(labels).most_common(1)[0][0]

grouped = df.groupby("Claim").agg({
    "Answer": list,
    "Label": most_common_label
}).reset_index()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key="")


template = """
You are an expert fact-checking assistant. 

Given a claim and a list of evidences from news , analyze all evidences for each claim and determine the overall verdict.

Verdict options:
- Supported
- Refuted
- Conflicting Evidence / Cherrypicking

Respond with only one of the three verdicts.

Claim: {claim}
Evidences: {evidences}

Final Verdict:
"""

prompt = PromptTemplate(
    input_variables=["claim", "evidences"],
    template=template,
)

chain = LLMChain(llm=llm, prompt=prompt)

predictions = []
for _, row in grouped.iterrows():
    evidences_str = "\n- " + "\n- ".join(row["Answer"])
    verdict = chain.run(claim=row["Claim"], evidences=evidences_str).strip()
    predictions.append(verdict)

def normalize_label(text):
    if not isinstance(text, str):
        return "Unknown"
    text = text.strip().lower()
    text = text.replace("/", " / ")
    text = " ".join(text.split())
    mapping = {
        "supported": "Supported",
        "refuted": "Refuted",
        "conflicting evidence / cherrypicking": "Conflicting Evidence / Cherrypicking"
    }
    return mapping.get(text, text.title())
grouped["Prediction"] = predictions
grouped["Label"] = grouped["Label"].apply(normalize_label)
grouped["Prediction"] = grouped["Prediction"].apply(normalize_label)

print("Results saved to output_with_prediction.csv")
print(classification_report(grouped["Label"], grouped["Prediction"]))
print("Weighted F1-score:", f1_score(grouped["Label"], grouped["Prediction"], average="weighted"))


