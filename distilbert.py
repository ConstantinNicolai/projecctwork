from transformers import DistilBertModel, DistilBertTokenizer
import torch

# Load the DistilBERT model and tokenizer
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize a sample sentence
inputs = tokenizer("Hello, my name is ChatGPT!", return_tensors='pt')

# Forward pass through the model
outputs = model(**inputs)

# The model output is a tuple with multiple elements, we are interested in the hidden states
last_hidden_states = outputs.last_hidden_state

# Print the output shape
print(last_hidden_states.shape)
