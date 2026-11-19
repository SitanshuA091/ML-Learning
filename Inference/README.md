# INFERENCE
When we download an LLM we get a bunch of artifacts instead of an executable file, different inference engines have different ways to load n serve models.
Upon downloading an LLM model from HF the bunch of artifacts that come ith it contain a file called `model.safetensors` which contain's model weights in json formats or `the model.bin` may also contain model weights. the `config.json` file contains the model's entire architecture the number of layers, no. of experts what attention mechanism etc.
These important files live in the SSD

## Inference Step - Loading Model

The steps in Inference may go through these steps
- Model Loading -> Prefill + Decoding -> Serving
Machines have a hierarchy of memory - GPU, RAM + CPU, SSD Inference engines use MMAP(Memory Map) to load the model 
- MMAP is where is the OS keeps track of the logical location of where weights are stored on the SSD when needed it loads lazily to the RAM (instead of eagerly) so that we save time loading the model.
- once the model weights are loaded in RAM the OS could evict it for other applications and once the inference engine requires to access it they can be loaded back from SSD to RAM.
- given a 15GB model is held in a 32GB RAM, communication medium between the RAM and SSD is the **PCIe Bus** which runs at 7.6GB/s on the 4.0 NvMe so unloading 5% f the weights (750MB of 16GB) at the speed would mean 107Ms of latency
- during inference weights have to be pushed to the GPU which does the heavy lifting of matrix multiplications from an input embedding
- **Quantization** takes model weights in BF16 and reduces them into FP32 and other lower precision types.
- GGUF is the most popular method used to lower the precision of model weights during the loading stage

## Inference Stages - Prefill and Decode
After Loading the model When an LLM processes a request, it does'nt generate all tokens uniformly it goes through 2 distinct stages known as prefill and decode, the prefill part is the upfront work of reading the input prompt whereas the decode stage is the iterative process of producing each output token

**Prefill Stage** - the model takes the entire input prompt and procersses tokens in parallel through the network(model) it computes the KV cache for the input sequence since whole prompt is known in advance, it utilizes the GPU's massive parallelism.  
**Decode Stage** - in this stage model generates output tokens one at a time each based on all prior tokens with KV cache available from the prefill. decoding is naturally sequential and iterative which makes it slower on per-token basis and often leaves GPU resources underutilized

- The **Time to first token (TTFT)** is essentially the latency of prefill stage. 
- prefill takes longer than each decode step bcoz it may involve processing hundreds of input tokens. 
- prefill when measured per token is more efficient
- **the time per output token(TPOT)** is determined by the decode stage latency.
- TOPT is less important than throughput for a system
- In prefill phase a single request wit a long prompt can already drive the GPU to maximum utilization, whereas in the decode phase higher utlization is achieved by batching processing multiple requests together.


### Batching Multiple Requests
serving one request at a time is inefficient as GPUs would sit idle during the lightweight decode steps. The solution is batching -processing multiple requests together so they share a single forward pass on the model.  

**Static Batching**   
- simplest approach - collect a bunch of incoming requests form a batch, and run the model to generate all their outputs together.
- the batch remains fixed untill every sequence in it is completely finished, in prefill stage all requests' prompts are processed in parallel.
- all sequences enter decode together sequences requiring only few output tokens might finish early but batch does not accept new requests untill slowest request in the batch has generrated all its tokens leading to wastage of GPU resources.
- once prefill GPU can olely focus on decode steps for that batch, which minimizes context switching overhead( the extra latency, wasted compute, and memory-transfer cost incurred when a system stops running one inference task or model context and switches to another), this tends to optimize **TPOT** 
- GPU remains underutilized for a good portion of time as new requests must wait untill the entire batch finishes before starting

**Continuos Batching**
- continuos batching is a strategy that addresses the issues with static batching by making the process dynamic. 
- instead of keeping batch membership fixed untill all sequences finish generation that slot can be immediately filled with new requests (starting its prefill) n the next iteration
- the batch size and composition can change at each decode step
- with continuos batching GPU utilization stays higher and throughput is higher
- continuous batching has new challenges - we have to manage interplay b/w prefill within the same GPU step

---

### **Request Scheduling and Continuos Batching**

**Prefill First**  
- simple way to schedule a continuous batch is a prefill-first policy 
- It means whenever a new request arrives and there is a free slot immediately run its prefill. 
- this prioritization of prefills minimizes the TTFT whereas throughput may suffer for other users they may see a hiccup in the stream and their next token is delayed as decode maybe interrupted by new prefills

**Chunked Prefill**  
- the way to mitigate issues with prefill first policy is through chunked prefilling
- This means instead of processing long prompts all at once split the prompt into chunks and intermix these chunks with decode steps.
- Using this way an ongoing decode isn't completely halted by new prefilss, its only slowed down
- it slightly increases TTFT but reduces the interruption felt by others
- chunked prefill also maximizes resource efficiency

with continuous batching of any strategy it can increase per request latency variance which is a known throughput latency trade off. pushing for maximum throughput or minimizing latency hurts each other 

## Disaggregate Serving : Seperating Prefill and Decode
disaggregating prefill and decode is an emerging research nd inference strategy where we decouples the prefill and decode phases onto different resources.  
splitting them allows each to each of prefill and decode steps to be optimized and scaled independently

Summarized Notes
- while LLM serves users in real time- it faces juggling act to get quick response and also serve as many users.
- Inference involves 2 phase nature of heavy prefill and slow decode
- Naive batching of requests can leave GPU resources idle 
- Continuous batchingand & improvement Requests Schedulinga Batching  (prefill first and chunked) optimizes resources and serves tradeoffs b/w compute (latency) and having high throughput 