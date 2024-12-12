import torch
from transformers import BertTokenizer, BertModel, pipeline
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer, util
from inf import perform_ere_on_text as ERE
import ast
class Compute:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentiment_pipeline = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
        self.sim_model = SentenceTransformer('all-mpnet-base-v2', device=self.device)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)

    def compute_similarity(self, sentence1, sentence2):
        embedding1 = self.sim_model.encode(sentence1, convert_to_tensor=True, device=self.device)
        embedding2 = self.sim_model.encode(sentence2, convert_to_tensor=True, device=self.device)
        return util.pytorch_cos_sim(embedding1, embedding2).item()

    def compute_sentiment(self, text):
        sentiment = self.sentiment_pipeline(text)[0]['label']
        return sentiment

    def compute_pearson_correlation(self, sentence1, sentence2):
        tokens1 = self.bert_tokenizer(sentence1, return_tensors='pt').to(self.device)
        tokens2 = self.bert_tokenizer(sentence2, return_tensors='pt').to(self.device)
        with torch.no_grad():
            embedding1 = self.bert_model(**tokens1).last_hidden_state.mean(dim=1).squeeze()
            embedding2 = self.bert_model(**tokens2).last_hidden_state.mean(dim=1).squeeze()
        return pearsonr(embedding1.cpu().numpy(), embedding2.cpu().numpy())[0]

    def extract_events_R(self, triples):
        triples = ast.literal_eval(triples)
        return [(t[0], t[1], t[2]) for t in triples]


    def analyze_claim_answer(self, text1, text2):
        similarity = self.compute_similarity(text1, text2)
        sentiment1 = self.compute_sentiment(text1)
        sentiment2 = self.compute_sentiment(text2)
        polarity = (
            "PP" if sentiment1 == "POSITIVE" and sentiment2 == "POSITIVE"
            else "NN" if sentiment1 == "NEGATIVE" and sentiment2 == "NEGATIVE"
            else "PN"
        )
        pearson_corr = self.compute_pearson_correlation(text1, text2)
        return similarity, polarity, pearson_corr

    def perform_ere_on_text(self, text, model_path, rebel_model_path):

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        relation = ERE(model_path, text, rebel_model_path)
        return relation

