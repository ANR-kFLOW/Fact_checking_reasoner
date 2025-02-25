import os
import torch
import pandas as pd
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

os.environ["HF_HOME"] = "/data/Youss/huggingface"

torch.random.manual_seed(0)
model_id = "microsoft/Phi-3-medium-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='balanced',
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 300,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

def generate_relation(row, event_type='subject', template=None):
    claim_ere = eval(row['claim_ERE'])
    answer_ere = eval(row['answer_ERE'])

    if event_type == 'subject':
        event1 = claim_ere[0]
        event2 = answer_ere[0]
    elif event_type == 'object':
        event1 = answer_ere[2]
        event2 = claim_ere[2]

    # Format the message with the dynamic event1 and event2
    prompt = template.format(input_event1=event1, input_event2=event2)

    messages = [
        {"role": "user", "content": f"knowing that I want to extract refined causal relations between two given events, and it only can be cause, intend, prevent, enable, no relation. You have to answer only with the relation name, no explanation. what will be the relation between earthquake and death"},
        {"role": "assistant", "content": "cause."},
        {"role": "user", "content": f"What about relation between {event1} and {event2}"},
    ]

    output = pipe(messages, **generation_args)
    print(output[0]['generated_text'])

    # Extract the relation from the outpu
    return output[0]['generated_text']


# Read the data
df = pd.read_csv('/data/Youss/Fact_cheking/reasoner/output_file_4.csv')


template = "What is the relation between event1: {input_event1} and event2: {input_event2}?"

# Apply the function to get the relation for subject and object
df['relation_subject'] = df.apply(generate_relation, axis=1, template=template, event_type='subject')
df['relation_object'] = df.apply(generate_relation, axis=1, template=template, event_type='object')

# Save the result to a new file
df.to_csv('output_file.csv', index=False)
