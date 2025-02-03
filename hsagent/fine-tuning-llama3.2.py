# https://www.datacamp.com/tutorial/fine-tuning-llama-3-2

#Importing the dataset
dataset = load_dataset(dataset_name, split="train")
dataset = dataset.shuffle(seed=65).select(range(1000)) # Only use 1000 samples for quick demo
instruction = """You are a top-rated customer service agent named John. 
    Be polite to customers and answer all their questions.
    """
def format_chat_template(row):
    
    row_json = [{"role": "system", "content": instruction },
               {"role": "user", "content": row["instruction"]},
               {"role": "assistant", "content": row["response"]}]
    
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc= 4,
)

