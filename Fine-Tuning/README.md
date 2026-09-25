## **Fine Tuning**
Fine Tuning means training a pretrained model on new data to improve its performance on a specific task.  
LLM/Model fine tuning was to train the entire model the hidden layers b/w input & output on new data, the weights would be altered using the new data this was full **fine-tuning**.  
**Issues With Full Fine-Tuning**
- full network needs to be trained, which is computationally expensive
- storage req. for the checkpoints are expensive
- switching b/w diff full fine tuned models is difficult as models need to be loaded/unloaded

## LORA 

LORA stands for **Low Rank Adaptation** its a parameter efficient fine tuning technique, here Low Rank refers to minimum no. of rows and columns in a matrix, the rank is smaller than the dimensions of the matrix and it is basically a compact representation of a matrix.  
the matrix dimension is related to the rank of the matrix; LORA suggests adaptation through a low rank matrix, It freezes the pre-trained weights and adds a trainable piece of rank decomposition matrices into each layer of the model.  

**W<sub>n</sub> = W<sub>o</sub> + △W**

△W is constructed by multiplying B & A which are low-rank decomposed matrix where if assume  W<sub>o</sub> has shape (d X k) then:-   

Matrix **A** - (down projection), shape = r x k  
Matrix A compresses the input dimension k down to small bottleneck rank r (r << min(d, k)), A is initialized with gaussian random numbers.  


Matrix **B** - (Up projection), shaper = d x r  
It projects the r-dimensional output back up to the original output dimension d, Its initialzed to all zeros
- since B starts at zeros △W = B * A = 0 at step 0 and so fine tuning starts wth exact model behaviour
- standard weight updates △W are full rank (d x k) by having △W = B * A, △W can have a maxium rank of onlr r.  
<em> **The above is because of the matrix Rank RUle rank(B * A) <= min(rank(B), rank(A))** </em>
- LORA matrices(A + B): (r x k) + (d x r) so eg. ( 8 x 4096 ) + (4096 X 8) = 65,536 compared to full rank 4096 x 4096 = 16.7M which is a ~99% reduction

**Forward Pass**:
- The output of B*A is scaled by a scaling factor α (alpha)and r (rank) as follows: (α / r), where r is the intrinsic dimension, typically ranging from 1 to 64 
 $$h = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$

$\alpha$ is a constant scaling hyperparameter

LoRA is just one of several efficient fine-tuning approaches. A notable variant is Quantization LoRA (**QLoRA**), which combines high-precision computation with low-precision storage.

## QLORA
QLORA - Quantized LoRA is an efficient finetuning approach that reduces memory further while finetuning models