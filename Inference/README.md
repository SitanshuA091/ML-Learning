When we download an LLM we get a bunch of artifacts instead of an executable file, different inference engines have different ways to load n serve models.
Upon downloading an LLM model from HF the bunch of artifacts that come ith it contain a file called `model.safetensors` which contain's model weights in json formats or `the model.bin` may also contain model weights. the `config.json` file contains the model's entire architecture the number of layers, no. of experts what attention mechanism etc.
These important files live in the SSD

## INFERENCE
The steps in Inference may go through these steps
- Model Loading -> Prefill + Decoding -> Serving
Machines have a hierarchy of memory - GPU, RAM + CPU, SSD Inference engines use MMAP(Memory Map) to load the model 
- MMAP is where is the OS keeps track of the logical location of where weights are stored on the SSD when needed it loads lazily to the RAM (instead of eagerly) so that we save time loading the model.
- once the model weights are loaded in RAM the OS could evict it for other applications and once the inference engine requires to access it they can be loaded back from SSD to RAM.
- given a 15GB model is held in a 32GB RAM, communication medium between the RAM and SSD is the **PCIe Bus** which runs at 7.6GB/s on the 4.0 NvMe so unloading 5% f the weights (750MB of 16GB) at the speed would mean 107Ms of latency
- during inference weights have to be pushed to the GPU which does the heavy lifting of matrix multiplications from an input embedding
- **Quantization** takes model weights in BF16 and reduces them into FP32 and other lower precision types.
- GGUF is the most popular method used to lower the precision of model weights during the loading stage

### Inference Stages - Prefill and Decode
After Loading the model When an LLM processes a request, it does'nt generate all tokens uniformly it goes through 2 distinct stages known as prefill and decode, the prefill part is the upfront work of reading the input prompt whereas the decode stage is the iterative process of producing each output token

**Prefill Stage** - the model takes the entire input prompt and procersses tokens in parallel through the network(model) it computes the KV cache for the input sequence since whole prompt is known in advance, it utilizes the GPU's massive parallelism.  
**Decode Stage** - in this stage model generates output tokens one at a time each based on all prior tokens with KV cache available from the prefill. decoding is naturally sequential and iterative which makes it slower on per-token basis and often leaves GPU resources underutilized

- The **Time to first token (TTFT)** is essentially the latency of prefill stage. 
- prefill takes longer than each decode step bcoz it may involve processing hundreds of input tokens. 
- prefill when measured per token is more efficient
- **the time per output token(TPOT)** is determined by the decode stage latency.

