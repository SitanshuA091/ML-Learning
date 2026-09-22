## Distributed Data Parallelism
- Distributed training is required when GPU memory may not be enough
- we are forced to use a small batch size because bigger batch size givves OOM errors
- training dataset maybe too large

Training setup needs to be scaled  
**Vertical Scaling**
- 1 x GPU server - 8GB RAM, 4GB GPU memory ----> 1 X GPU Server -  64GB RAM and 32GB GPU memory
no code change upgrade hardware setup to more compute and memory.

**Horizontal Scaling**
- 1 X GPU Server(8GB RAM , 5GB GPU memory) ----> 4x GPU Servers - 8GB RAM 4GB GPU Memory

## HORIZONTAL SCALING IN DETAIL
- **Data Parallelism** - If model can fit within a single GPu we can distribute trainming on multiple servers (each containing one or more GPUs) with each GPU processing a subset of the entire dataset in parallel and synchronizing gradients during backpropogation.
- **Model Parallelism** - If Model cannot fit within a single GPU then we need to break the model into smaller layers and let each GPU process a part of the forward/backprop step during gradient descent
- data Parallelism and model parallelism can be ciombined to create a a hybrid option

**Gradient Accumalation** 
given ypred = x1w1 + x2w2 + b
goal becomes to use stochastic grad descent to find values of params w1, w2 and b such that MSE loss b/w ypred and ytaget is minimized.
argmin(ypred-ytarget)^2
(w1,w2,b)  
pytorch will create a computational graph taking inputs x1 and x2 and params of weights and the bias b which produces the prediction which is compared against the target, model runs backpropogatioon to minimizethe loss b/w the target and the predictions, `loss.backward()` will calculate the gradient w.r.t param. `optimizer.step()` will update the value of each param using the gradient - param(new) = param(old) - a * grad, ,`optimizer.zero()`- zeroes out all gradients next we run forward on next data items with all steps.
this is running without grad accumalation, with gradient accumalation we accumalate gradients from say prior 2 steps and update w1, w2, b with gradients every 2 items, we do not do optimizer.zero() say on odd data items and because of that g1 stays and then g2 is calculated and while updation of w1 we use g1+g2 both.

## DDP TRAINING
<em>**node** - GPU server</em>
- begining of training model's weights are initialized(randomly) on one node and sent to all other nodes (broadcasted)
- each node trains on a subset of data with no overlap with only forward and backward step.
- every few batches gradients of each node are accumalated on one node and sent back to all other nodes (reduce)
- each node then updates params of its local model with gradients received using its own optimizer using `optimzer.step()`.
- sequence of reduce and boradcast combined together is called all-reduce.
- cycle continues
simple eg. four GPU nodes we have 8 = batch size 4 batches are trained now G1, G2..G4 does not update params with gradients at all it accumalates across 8 items grad1 and same for other gpu nodes, then grad1, grad2..grad4 are broadcasted to one node (say G3) with all reduce it gives a global gradient then params are updated with this global gradient.

**COLLECTIVE COMMUNICATION AND POINT TO POINT COMM'N** 
- collective communication allows to model the comm'n pattern b/w groups of nodes
- it basically means communication operation where multiple GPUs/processes participate together, rather than one GPU sending data directly to another
- In point to point comm'n you'd send the file iteratively to each of the friend one by one.(given 1MB/s file size - 5Mb time takes 35 secs)
- Collective communication intrdouces **broadcast** opeartor, it assigns a unique ID to each node knownn as RANK, which in case of 7 GPUs once a GPU has recieved the gradient accumalated they can themselves send to nodes without them which reduces time as simultaneously file/graident value is shared by multiple servers to nodes whcih did'nt have them yet.
- collective communication we explot interconnectivity b/w nodes to avoid idle times and reduce total communication time.
**FAILOVER: If One Node crashes**
- Restart entire cluster which would mean training would restart, all params and computation would be lost.better approach is to use checkpointing(saving model weights on a shared disk every few iterations(eg every epoch))
- using the chckpoint `model.pt, we need to have shared storage for the checkpoints since pytorch needs the checkpoints to initialize the weights to any node.
**Demo on Paperspace**
- create a network 
- use that network while creating machine instances (choose Ml-in-a-box no specific OS).
- we also need drive space from paperspace(same cloud provider) ideally to have that shared storage across machines to load weights from saved checkpoints.
- use !torchrun --nproc-per-node=4 train.py --model Moe --epochs 5, --nproc-per-node is inbuilt arg in torch.
- `torchrun` will use two env variables local_rank and global_rank 
  - local rank is the ID of the GPU on the local computer, its useful when we want to print only on 1 GPU per system
  - global rank (RANK) - indicates globally unique ID among all nodes in the cluster, its useful when we want to use a specific GPU to perform an operation amongst all others.(eg. to save checkpoints or to initialize services like W&B)
### **PRACTICAL CODE TIPS and Template**
- Use `local_rank` and `rank` from configs
- Introduce a new param in Dataloader `sample = DistributedDataSampler(train_dataset, shuffle = True)` and keep shuffle = False in Dataloader params.
- create a logic to preload a checkpoint - the latest one, with state_dict and all..
- initialize W&B in if global_rank==0
- wrap the model(our class) in an instance of `DistributedDataParallel()` (torch docs)
- training loop (for one epoch) we have same model needs to be called like model.module.encode or any other method bcoz the model is an instance of ddp class from pytorch.
- use `init_processgroup(backend = 'nccl')`
- use `torch.save()` in the if globsl_rank ==0 only on one GPU server we wanna save checkpoints.

**DDp in Pytorch**
- Pytorch synchronizes gradients when loss.backward is called 
- we can use `no_sync` from `DistributedDataParallel` to avoid pytorch accumalating gradients for every few steps inside the no_sync context.
- use it with model(DDP class's instance) - `with model.no_sync():`.
- the all-reduce op of sending accumalated gradients happens as soon as gradients are calculated for a set of params in one layer which leads to less Idle time due to communication overhead.