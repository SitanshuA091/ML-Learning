## TOKENIZATION
the foundational process in Natural Language Processing (NLP) that breaks raw, unstructured text into smaller, manageable units called tokens


### Main Types of Tokenization
- Word Tokenization: Splits text into individual words based on spaces or punctuation
- Character Tokenization: Splits text down to individual letters or characters. Eliminates out-of-vocabulary issues, but creates very long sequences that lack immediate semantic meaning.
- Subword Tokenization: Strikes a balance by breaking rare or complex words into smaller, meaningful chunks 

### Subword Tokenization Algorithms
- **Byte-Pair Encoding (BPE)**: Iteratively merges the most frequent pairs of bytes or characters in a corpus.
- **WordPiece:** Used by BERT, selecting subwords based on likelihood rather than just raw frequency.
- **SentencePiece:** An unsupervised text tokenizer that treats the input as a raw stream of characters.

## Embeddings V Tokens
Embeddings are just vectors that represents something, vectors are an array of numbers an x-y cordinate - (0.2, -0.7) is a 2D vector, 
(0.2, -0.7, 0.3) - 3D vector
how words translate to vectors that represent words -> 
```
eg. king - man + woman = queen  
``` 
intuitively or semantically its correct even though the equation is not mathematical
the words an be represented as  

**King** = (1, 9)  
**Man** = (1, 0)  
**Woman** = (9, 0)  
**queen** = (9, 9)  

so we represent the words as vectors with just 2 dimensions (embedding models give us many features/dimensions for representing singular words). the dimensions
**Tokens** are just numbers/IDs assigned to words which don't hold semantic meanings the tokens point to specific embeddings.  
since meanings of words are only undrstood in context 

```
I boarded my train
Vs
I train everyday
```

the word **train** points to same token but does'nt carry same meaning in the context of the sentence which is where Embeddings are helpful the vector embedding maybe updated for particular tokens as the surrounding context(surrounding words).

change embeddings have proven to be the most useful in training language models amongst other things chosen to represent semantic meaning