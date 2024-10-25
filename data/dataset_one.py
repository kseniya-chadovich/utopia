import pandas as pd
from tokenizer import tokenize_df
from sklearn.model_selection import train_test_split

url = 'https://raw.githubusercontent.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset/refs/heads/main/Dynamically%20Generated%20Hate%20Dataset%20v0.2.3.csv'

df = pd.read_csv(url)
# print(df.head())

df_cleaned = df[['text', 'label']]
df_cleaned.dropna(subset=['text', 'label'], inplace = True)
df_cleaned['label'] = df_cleaned['label'].map ({'hate': 1, 'nothate' : 0})
print(df_cleaned.info())
print(df_cleaned.head(30))


# Split the dataset first
train_df, test_df = train_test_split(df_cleaned, test_size=0.2, stratify=df_cleaned['label'], random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df['label'], random_state=42)

# Tokenize each split
train_tokens = tokenize_df(train_df)
val_tokens = tokenize_df(val_df)
test_tokens = tokenize_df(test_df)
