## ALIBi- Attention with Linear Bases
- Given length of training sequences **L**, L has to be equibalent to length of inference sequences.
- more context is achieved is achieved by larger L
- Transformer models (LMs) that use sinusoidal positional embeddings have very weak extrapolation abilities  
<em> **extrapolation** is a model’s ability to continue performing well as the number of input tokens during
validation increases beyond the number of tokens on which the the model was trained. </em>
- ALiBi is introduced to facilitate efficient extrapolation
- it negatively biases attention scores with a linearly decreasing penalty proportional to the distance b/w relevant key and query, it eliminates positional embeddings
- T5 bias method leads to better extrapolation than either of these. 
<em>**T5 bias** refers to relative position bias used in attention. its socre(i, j) = qikjT/sqrt(d) + b(i-j) without T5 the b does'nt get added, its added before softmax</em>
- with AliBi we take the token embeddings and feed it to attention layer and a ALiBi(relative positional bias) is added.
- while computing attention socren for each head the linearly biased attention method addes a constant non learned bias to each attention score(qi*kj, left), the softmax function in the attention layer is applied afterwards.
- `softmax(qiK^T +m *[-(i-1,..,-2, -1, 0)])` here m is the head specific slope fixed before training 
- models with 8 heads slopes used are the geometric sequence 1/2, 1/2^2, 1/2^3..1/2^8 for 16 heads we interpolate those 8 slopes by geometrically averaging every consecutive pair so it starts at 1/sqrt(2) and has the ratio of 1/sqrt(2) - 1/2^0.5, 1/2^1, 1/2^1.5..1/2^8 
- for n heads the set of slopes is the geometric sequence that starts at 2^(-8/n) and uses the same value as its ratio.
- it penalizes attention scores b/w query-key pairs, with penalty increasing as the distance b/w a key and query grows
- different heads increase penalties at different rates 
- AliBi is added to the keys and queries but not the values (same as T5 bias and rotary method)
 
### IMPLEMENTATION DETAILS
- its implemented by modifying the mask matrix by adding the linear biases to it 
- `Q,K,V → d
​QKT​→ + ALiBi bias → mask→ softmax`
- in practice query qi attends to keys 1-i this is implemented by adding a mask matrix to the query key dot product before the softmax operation is applied

