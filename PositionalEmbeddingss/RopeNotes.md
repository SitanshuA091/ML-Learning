## ROTARY POSITIONAL EMBEDDINGS
Why positional embeddings are required - 
- Transformer models are invariant to positions by default
- The dog chased the pig will have the same representations as the pig chased the dog
- Absolute Positional embeddings - 
  - word will have some embeddings representations
  - the position of the word say dog(2) will have an embedding representation of the same dimension
  - the 2 embeddings are simply added and presented as input to the transformer layers
  - either these are learned from data or sinusoidal functions from attention paper
  PE(pos, 2i) = sin(pos/10^(42i/dmodel)
  PE(pos, 2i+1) = cos(pos/10^(42i/dmodel)
- **Issues with positional encoding**
  - position vectors for 1-512 are only there when learned from data, so Max length is bounded 
  - If a sequence has more length then issues occur with transformer processing
  - all pos embeddings are independent of each other 
- Relative pos embeddings require extra step in attention layer, changes in every step so KV caching becomes difficult.

### ROTARY POSITIONAL EMBEDDINGS
- Instead of adding up word's  embedding vector and positions apply a rotation to the vector
- to represent the position aspect we rotate the word embedding vectors by an angle θ. 
- the later the word appears position-wise in the sequence increases the angle of rotation by the times of its integer position 
- `The pig chased the dog` - in the sequence the dog vector is going to be rotated 4*θ as dog appears 4 positions later than first word in the sequence
- if more words are added after end of sequence it leaves the words before without any changes
- also preserved distances and relative positions 2 word's embedding vectors
- **Implementation**   
(<em> Assume Vector only has 2 dimensions </em>
- xm = the input/token representation at sequence position \(m\).
- m is the absoulte position, x is the vector we try to rotate\.
- \(x_m^{(1)},x_m^{(2)}\) = its two components in this simplified 2-dimensional derivation.
- \(W^{(q,k)}\) is the ordinary learned linear projection matrix used to produce either the query or the key.
- at the start \(W^{(q,k)}\) are just random initialized weights that we have while training any model  
**general equation d dimensional rotation matrix** -> 
  $$f_{\{q,k\}}(x_m, m) = R_{\Theta,m}^{d}\,W_{\{q,k\}}\,x_m
$$
- Rd has the matrice like this 
```
 _                                      _
|cosmθ1 -sinmθ1    0      0     .... 0 0 |
|sinmθ1  cosmθ1    0      0     .... 0 0 |
|   0      0     cosmθ2 -sinmθ2 .....0 0 |
|   0      0     sinmθ2  cosmθ2 .....0 0 |
|   :      :        :     :          : : |
|   0      0        0     0 ...cosmθd/2..|
|_  0      0        0     0 ...sinmθd/2._|
```
- it takes the vector splits into chunks of 2 dimensions and rotates them one at a time
- different rotation angle is applied to every vector
- simpler way with minimized computation is 
```
Rdθ,m^x = [x1, x2..]1xd *  [cosmθ1 cosmθ1 cosmθ2 cosmθ2..cosmθd/2 cosmθd/2]1xd/2 + [-x2 x1 -x4 x3 .. -xd-1 xd]1xd * [sinmθ1 sinmθ1 ...sinmθd/2 sinmθd/2]
```
- this utilizes 2 vector multiplications and one addition
- general behzvious is that when words are far closer in the sequence produce smaller dot product on avg and further apart tokens produce larger dot prouduct.
- words further apart should be less related to each other which is validated by above point.