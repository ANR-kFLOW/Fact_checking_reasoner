import torch
from transformers import AutoTokenizer, GenerationConfig
import io
import warnings

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def extract_SOR(sentence, prediction):
    sentence = sentence.lower()
    prediction = prediction.lower()
    words = prediction.split()
    relation = words[-1]
    remaining_words = words[:-1]
    sentence_words = sentence.split()

    if len(words) == 3:
        subject = words[0]
        obj = words[1]
    else:
        subject = []
        obj = []
        sentence_index = 0
        if remaining_words[0] in sentence_words:
            try:
                sentence_index = sentence_words.index(remaining_words[0])
            except ValueError:
                return 'invalid', relation, 'invalid'

            subject_ended = False
            for word in remaining_words:
                if not subject_ended:
                    if sentence_index < len(sentence_words) and word == sentence_words[sentence_index]:
                        subject.append(word)
                        sentence_index += 1
                    else:
                        subject_ended = True
                        obj.append(word)
                else:
                    obj.append(word)

            subject = ' '.join(subject)
            obj = ' '.join(obj)
        else:
            subject = 'invalid'
            obj = 'invalid'

    return subject, relation, obj


def perform_ere_on_text(text, model_path, rebel_model_path):
    sentences_to_infer = [text]
    ere_results = test_model(sentences_to_infer, rebel_model_path)
    
    if ere_results:
        subject, relation, obj = ere_results[0]
        return relation
    else:
        return 'No relation found'


def test_model(sentences, rebel_model_path):
    with open(rebel_model_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
    model = torch.load(buffer, map_location=device).to(device)
    model = torch.nn.DataParallel(model)

    tokenizer = AutoTokenizer.from_pretrained('Babelscape/rebel-large')
    pred = []


    generation_config = GenerationConfig()

    for sentence in sentences:
        encoding = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in encoding.items()}


        with torch.no_grad():
            outputs = model.module.generate(**inputs, generation_config=generation_config)

        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred.extend([extract_SOR(sentence, output) for sentence in sentences])

    return pred


def main():
    model_path = '/data/Youss/Fact_cheking/AVeriTeC/ERE_models/best_model_st1.pt'
    rebel_model_path = '/data/Youss/Fact_cheking/AVeriTeC/ERE_models/augmneted_model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    input_text = "the vaccine was made in order to prevent the soread of covid19"

    relation = perform_ere_on_text(input_text, model_path, rebel_model_path)
    print(f"Extracted Relation: {relation}")


if __name__ == "__main__":
    main()

