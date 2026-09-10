## Attention and Transformer Architecture
**RNNS recap in brief**
General architecture and flow ->
- in RNNS we provided an input x1 along with an initial state and the RNN produced an output, this happened at first timestep
- then hidden state of the previous timestep and the next input token and network would produce the second output token y2
- given n tokens we need to have n timesteps and map and n sequence input into an n sequence output.

**Issues with RNNs**
- Slow computation for long sequences, if there are large number of tokens (n) then computation becomes very slow.
- Vanishing or exploding gradients.
- Difficulties in accessing information from long timne ago

### TRANSFORMER STRUCTURE AND ARCHITECTURE DETAILS
- Divided into 2 micro blogs Encoder and the Decoder 
- **Encoder**
  - starts with the **input embedding**, given a sentence <em>Your cat is a lovely cat</em>, this sentence is converted into smaller tokens which can be each word for token or even further.
  - the split tokens is now mapped to some numbers based on some vocabulary size
  - these numbers are then converted into embeddings/vectors of size 512 each word is always mapped to the same input IDs, except however these embeddings will be treated as trainable parameters to learn some meaning pattern.
  - **positional encoding** - we need each word to carry some information about its position in the sentence, we need the model to treat words that are close as close and the words that are distant as distant.
  - positional encoding needs to represent a pattern that can be learned by the model.
  - after we get input embeddings(size = 512) we convert into position embeddings which we add to the input embeddings
  - How its calculated -> for even positions PE(pos, 2i) = sin(pos/10000^(2i/model)) i here is the index of the dimension in the embedding vector
  - for odd positions -> PE(pos,2i +1) = cos(pos/10000^2i/dmodel)
  - for different sentence later new positional encoding is not calculated again same vectors are used for both training as well as inference
  - cosine and sine functions are chosenm so that the model can recognize a pattern of values from continmuos functions, so relative positions are easier to see by the model
  - **Attention Layers** - **Self Attention** allows model to relate words to each other Attention(Q,K,V) = softmax(Q*K^T/sqrt(dk))*V
  - Its query, key and values because 
  - given input sequence of 6 words with dmodel(dk) =512, Q K and V are the matrices from the input sequence itself. 
  - so input matrix Q (6, 512) * K(6, 511)"s transpose => Q(6,512)*K(512, 6)/sqrt(512) 
  - resultant matrice from numerator will be in (6,6) matrice shape 
  - Q, K and W are basically the input sentence vectors stacked as once.
  - A cat is sleeping will have four vectors which are stacked into one X and Q = XWq, K = XWk, V = XWv in multi head attention which we will expand later
  - finally after applying the Q*K^T we get matrice of 6,6 which we then multiply by V which results (6,512) same as initial dimension
  - new matrice has 6 rows (6 words) which every word has now an embedding of 512 dimension
  - this embedding captures the meaning not only of that word based on vocab but also the relationship of the word with other words.
  - self attention is permutation invariant, self attention requires no params, also values along diagonal which is dot product multiplication of an embedding value by itself also its the introduction of attention in the architecture Multi head Attantion is the part after the Positional encodings
  - **Multi-Head Attention** -> given 
  some sequence length = seq, dmodel = size of embedding vector, h = no. of heads, Q, K and V with dimension(seq, dmodel) multiplied with a weight matrix Wq, Wk, Wv
  - Now the new matrice from QWq, KWk, VWv are split into smaller dimensions using the sequence deimension or dmodel dimension and resultant dimension will be dk which can be dmodel/4 (taking 4 heads or 4 splits) or seqlen dimension/no. of heads.
  - now we calculate attention from the formula -> head(i) = Attention(QWiq, KWik, VWiv), Attention is same - softmax(QK^T/sqrt(dk))*V 
  - this results will be matrices for head1, head2.. headh (h=no. of heads) of the dimension dk x seq.len.
  -  finally MultiHeadAttn(Q, K, V) = concat(headi..headh)*Wo here Wo = H(h x dk), the final multiheadttan matrice will be (seqlen x dmodel)
  - Wo is a learned weight matrix that mixes info from all heads its a linear transformation applied with the integrated heads concatenated
  - **Add & Norm Layer** :-
  - **Layer Normalization** - each value is replaced with normalized values in ranges(0, 1) given mean and variance of each vector independently which is then used with the formula of x^(i) = xi - meani/sqrt(variance^2 + gamma) gamma is multiplicative and beta is the additive learnable params which
  - In the encoder its followed by a feed forward layer followed by another add and norm layer
- **Decoder** - The Decoder components similarly has output embeddings(input like), Positional encodings, **Masked multi Head Attention** which is fed into the next multi head attention block, the same attention block is fed the output of the encoder in the forms of keys and values while query comes from the previous decoder.
  - **Masked Multi Head Attention** - The goal is to make the model casual,menaing output at certain position can only depend on the words on the previous positions. model should'nt see future words
  -  the above is achieved using deletion of values in the matrice which are the products involving word being multiplied with embeddings of a future word so embedding of word1 can only see word1's value in the other matrice similarly word2 sees word1 and word2 no further this deletes or masks the right of the resultant matrice with attention scores involing future word along the row horizontally.
  - the replacement of those values is -∞ so that after softmax activation it becomes zero.
  - in the matrice the values above principal diagonal are replaced.
  - it becomes cross attention as mask multi head attention output is fed as query whereas Key and values come from the output of encoder.
  - the masked multi head attention is the self attention of the input sentence embeddings with positional encodings of the decoder.
  - the query, key value are same as its self attn.
  - the output of MHA layer in decoder is then passed to an add and norm layer which is passed to a feed forward layer which is just a **fully connected NN**  and again its passed to the add and norm and finally to the linear layer

<em> The residual connection block is part of every block or sublayer since every layer is wrapped like Output=LayerNorm(x+Sublayer(x)) its added since for training we are adding a small adjustment to the output this makes training better.</em>
- so the output of the MHA would then go through layer normalization and then we add a small adjustment of residual x to it which then goes to the feedforward layer and then produces some output again added to the residual.

### INFERENCE AND TRAINING
- **TRAINING** - given sentence <em>I love you very much</em> on a french translation results -> Ti amo molto
- we prepend and append special tokens to show end and start like <SOS> and end <EOS>
- no matter the length of sentence the input seq len we keep same by addition of padding tokens
- the input sentence is fed to the encoder whereas the output target sentence is fed to the decoder both are conv to embeddings then applied with positional encodings
- resultant target embeddings is fed to the masked MHA in the decoder, input ones passed to the MHA layer with the decoder
- then we get the decoder output after passes to the add n norm and fc nn layers.
- the final linear layer will allow to map sequence x dmodel into sequence x vocab_size (vocab size is based on the training data).
- finally after the linear layer we apply the softmax
- as we have the target label and the calculated prediction like all models we calculate the cross entropy loss.
- as opposed to RNNs training there was no massive training for loop for one sequence instead there is a single timestep for input sequence and label

- **INFERENCE** - we pass sentence to modelit gets passed to the encoder with the special tokens and required paddings
- encoder produces output which is (seq, d_model) 
- decoder only sees <SOS> decoder sees embeddings of the special tokens and paddings only and then output from encoder is fed and as key and value which now is fed to the decoder's rest part and it feeds to the linear layer
- linear layer projects it back to our vocabulary from vocab_size from trainig data.
- softmax then is applied and choses the higher probablity tokens
- this happens at timestep 1 and continues timestep 2 but we don not compute encoder output again, output of previous timestep is appended to the input of decoder with special token it becomes ->  <SOS> `predicted_token1`.
- same happens at timestep 4 and continues till the timestep produces the <EOS> token.
- **Inference Strategies**:-
  - selecting at every timestep using softmax is the greedy strategy.
  - **Beam search** is another strategy where we selct top 8 words and evaluate all possible next words for each of them at each step keeping top 8 most plausible sequences, finally we keep best possible and probable sequence








## ARCHITECTURE EXPLAINATION
- **Input Embedding** - Input embedding will convert the raw text sentences inputs into vector into 512 dimensions
  - first we convert raw text into input IDs for every word which gives positional sense/mappings to all these words
  - each of the position ID numbers are associated with a text embedding 
  - the d_model in InputEmbedding class will be 512
  - 

- **Positional Encoding** - it wil give the special position identifying encodings for the words
   - for positional encoding we will have a matrix of seq_len x d_model 
   - embedding vectors are of d_model size and we will need the input seq_length for positions
   - Formula for positional encoding from paper:
   - for even positions - PE(pos, 2i) = sin(pos/1000^(2i/d_model))
   - for odd positions - PE(pos, 2i +1) = cos(pos/1000^(2i/d_model))
   - we will use the simplified formula with log space which will give the same term ultimately and will not effect model training
   - we will also store the pe's as buffer using `register_buffer` because these will not be training params or any learned variables, to store within a module not as atraining param we use `register_buffer`.
   - forward method to add the positional encodings to everey word inside the sentence
- **Layer Normalization** - for each param in the batch of a layer we calculate a mean and variance and then a new value for the params is added as xi(new) = xi - mean(xi)/sqrt(variance^2 + epsilon) 
  - 2 params gamma and beta are also added to this which are multiplicative and beta is additive which introduces some fluctuations since values only b/w 0 & 1 maybe too restrictive for the model
  -  bias is the additive gamma value in code
  - `def ...()->None:` here the function is also hinted at a None type return value.
