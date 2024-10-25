from transformers import LlamaTokenizer

# THIS TOKENIZER DOES NOT WORK YET CUZ WE DON'T HAVE ACCESS TO LLAMA
tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b')

def tokenize_df(df):
    tokens = tokenizer(df['text'].tolist(), return_tensors="pt", padding=True, truncation= True)
    return tokens

def tokenize_text(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation= True)
    return tokens

def trial(text):
  result = tokenize_text(text)
  print(result)
    
