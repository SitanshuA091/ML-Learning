# SAMPLING
Sampling is the technique that bring the elements of controlled randomness in model responses while decoding(inference). whilst loading pretrained models the designed methods include arguments like `temperature`, `top_k`, `top_p` controls the model responses.  

### Greedy Decoding
---
At each autoregressive decoding step, the Transformer produces logits over the vocabulary: `z = [z1, z2..]` then softmax converts these probabilities which gives a the distribution across the vocab some prbability numbers
```python
"the"   → 0.50
"a"     → 0.25
"one"   → 0.15
...
"this"  → 0.07
```
greedy decoding would always select the `argmax(p)` amongst the probabilities whereas sampling can treat them categorically

### Applying Temperature 
temperature is the number by which the logits are divided by before the softmax is applied
starting with logits 
```
A -> 4.0
B -> 2.0
C -> 1.0
```
these logits are'nt probabilities, they're arbitrary real-value scores produced by the model's final linear layer, softmax will convert them into probablities.  
**P(A)** = e<sup>4</sup> **/** e<sup>4</sup> + e<sup>2</sup> + e<sup>1</sup>   
this results in P(A) -> 0.844, P(B) -> 0.114, P(C) -> 0.042 

**When divided by temperature**
- say temp = 2
```
original:      4.0  2.0  1.0
divided by 2:  2.0  1.0  0.5
```
now softmax([2, 1, 0.5]) becomes
```
A → 0.629
B → 0.231
C → 0.140
```
probablities became flatter bcoz the difference b/w logits were reduced previously when softmax was applied it did not favour P(A) nearly as better 
- say temp = 0.5
dividing logits
```
original:      4.0  2.0  1.0
divided by 2:  8.0  4.0  2.0
```
softmax([8, 4, 1]) produces 
```
A → 0.980
B → 0.018
C → 0.002
```
Now the prior higher value of logits absoultely dominates

when temperature T < 1 --> **differences amplified**  
T > 1 --> **differences compressed**

so temperature controls the distribution making it sharper or flatter.  
If Greedy Decoding is defaulted then temperature has no effect on which token gets selected(for positive temperatures)

### Top-K Sampling
---
Top k keeps the **k** highest scoring tokens, removes everything else and then samples acc to the probabilities  
After Softmax
```
A → 0.50
B → 0.25
C → 0.15
D → 0.07
E → 0.03
```
with k=3 we keep 
```
A → 0.50
B → 0.25
C → 0.15
D & E removed
```
the remaining probabilities are then normalized so they sum to 1
A → 0.56, B → 0.28, C → 0.16

### Top-p Sampling
Top-p(nucleus) is similar to top-k we say keep the smallest set of probability tokens whose cumulative probability reaches at least p
``` 
top_p = 0.90
   &
Token     Probability --> Cumulative
A           0.40            0.40
B           0.25            0.65
C           0.15            0.80
D           0.10            0.90
E           0.06            0.96
F           0.04            1.00
```
we keep 
```
A, B, C, D
```
as 0.40 + 0.25 + 0.15 + 0.10 = 0.90
after this E & F are discared and we renormalize A-D and sample from them 