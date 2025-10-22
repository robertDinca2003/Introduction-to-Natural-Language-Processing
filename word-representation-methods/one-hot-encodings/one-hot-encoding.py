# TO DO CREATE A GENERAL CLASS FOR THAT
import numpy as np
import pandas as pd

# Reading the corpus data
corpus = pd.read_csv('corpus.csv')
texts = np.vstack(corpus['Text'])


vocabulary = {}

# For every sentence in the corpus add every word in vocabulary and assign a positional value to it
for text in texts:
    for word in text[0].split():
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)
            
# The matrix storing the vector representation of every sentece in the corpus
# One more column for Out Of Vocabulary Words
one_hot_encodings = np.zeros(shape=(len(texts),len(vocabulary)+1))

for idx, text in enumerate(texts):
    for word in text[0].split():
        one_hot_encodings[idx][vocabulary[word]] = 1

# Reading the validation data
validation = pd.read_csv('validation.csv')
texts = np.vstack(validation['Text'])

# A new matrix to store the validation senteces in a form of vectors
validation_encodings = np.zeros(shape=(len(texts), len(vocabulary)))

for idx, text in enumerate(texts):
    for word in text[0].split():
        if word in vocabulary:
            validation_encodings[idx][vocabulary[word]] = 1
      
# Save matrixes  
np.savetxt(
    'training_embeddings.txt',
    one_hot_encodings,
    fmt='%d', 
    delimiter=',' 
)

np.savetxt(
    'validation_embeddings.txt',
    validation_encodings,
    fmt='%d', 
    delimiter=',' 
)