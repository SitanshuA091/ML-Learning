# Mixture Of Experts
Mixture of Experts (MoE) is an architecture that uses a set of expert networks to solve tasks, while a gating network determines which experts to use for each input.  
The Moe Model uses a sparse subset of experts to make predictions, imroving efficiency and scalability

## COMPONENTS
- **Expert Netowrks** - individual networks/models that process the inputs can be as simple as FFNs, or MLPs
- **Gating Network** - Decides which experts should process each input
- **Combination of outputs** - outputs of selected networks are combined based on a weighted sum where weights are given by the gating network

## Formulation 
```
input : x  
consider N experts
each network maps the input x to an output y for expert i.
```
**output**:  
y<sub> i </sub> = f<sub> i </sub>(x) 

**Gating Network**:  
it computes a set of scores **g** for each expert these scores indicate how much attention each expert should recieve for a given input x:  
y<sub> i </sub> = σ (w<sub> i </sub> <sup> T </sup> x )  
**σ** - sigmoid activation function (ensures scores  lie b/w 0 and 1) 

**Top-k Selection** :  
from the gating network top k experts are selected, experts with highest gating scores will be active-   
**{ g<sub>i <sub>1</sub></sub>(x), g<sub>i <sub>2</sub></sub>(x).......g<sub>i <sub>k</sub> </sub> (x) }**  

**weighted sum of experts**  
y<sub> final = $\sum_{ k=1}^{ K}$ w<sub>i k </sub> * f <sub>i k </sub>(x)
w<sub>ik</sub> is gating score for expertik and f<sub>ik</sub> is output of the expert
- to ensure weights sum to 1 the gating scores for the experts are normalized
- normalization step ensures weighted sum is properly scaled

