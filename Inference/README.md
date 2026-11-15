When we download an LLM we get a bunch of artifacts instead of an executable file, different inference engines have different ways to load n serve models.
Upon downloading an LLM model from HF the bunch of artifacts that come ith it contain a file called `model.safetensors` which contain's model weights in json formats or `the model.bin` may also contain model weights. the `config.json` file contains the model's entire architecture the number of layers, no. of experts what attention mechanism etc.
These important files live in the SSD

## INFERENCE
The steps in Inference may go through these steps
Model Loading -> Prefill + Decoding -> Serving
Machines have a hierarchy of memory - GPU, RAM + CPU, SSD Inference engines use MMAP(Memory Map) to load the model which is the OS keeps track of the logical location of where weights are stored on the SSD when needed it loads lazily to the RAM (instead of eagerly)