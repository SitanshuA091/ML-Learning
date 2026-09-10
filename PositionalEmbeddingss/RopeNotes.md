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
- Relative pos embeddings require extra step in attention layer, changes in every step so KV caching becomes difficult

### ROTARY POSITIONAL EMBEDDINGS
- Instead of adding up word's  embedding vector and positions apply a rotation to the vector
- 