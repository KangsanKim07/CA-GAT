# CA-GAT : Closeness Aware Graph Attention Network
2021 Spring &lt;CS492:Graph Machine Learning and Mining> Final Individual Project </br>
**Best Project Award** </br>


Implementation of Closeness Aware Graph Attention Network (CA-GAT) </br>
Base code from [@dongkwan-kim/SuperGAT](https://github.com/dongkwan-kim/SuperGAT/tree/20d0e9729c47c03a94937e199b150b713256bc9d) 
(I edited [SuperGAT/layer.py](https://github.com/KangsanKim07/CA-GAT/blob/main/SuperGAT/layer.py)) </br>
[Youtube Video](https://youtu.be/mzlcdxgNsQE)

## Introduction
- Adjacent nodes are not always important, and 2-hop, 3-hop nodes can be important </br>
![image](https://user-images.githubusercontent.com/59245409/120894466-74420100-c653-11eb-9eaa-b49c0b25c845.png) </br>
For node A, node E can be important as other neighbors, and node D might not be that much important.
But GAT just receives messages only from neighbors.
Let’s make new GAT that also gets messages from important nodes even they are not adjacent to target node.

- Overfitting of GAT comes from the lack of enough supervision data </br>
GAT uses only classification loss to train attention mechanism thus it's easy to overfit and it needs more data to train
If we give more information about each node to model, attention model can compute attention weights  better
Which information should we give? How can we give information to the attention mechanism?

## CA-GAT
- Idea : Use closeness information to identify important nodes </br>
Compute Personalized PageRank scores of each node, and in GAT layer, get messages from adjacent nodes and also nodes whose scores are bigger than threshold τ. 
If node u has bigger score of v than τ, make fake edge (u, v).
Node u receives message from adjacent nodes(containing fake neighbors, not adjacent but important).
<img src="https://user-images.githubusercontent.com/59245409/120894620-5a54ee00-c654-11eb-9dac-aa40399c9fbd.png" width="600">
<img src="https://user-images.githubusercontent.com/59245409/120894621-5d4fde80-c654-11eb-80d1-cd9a85472c0f.png" width="600">

## Dataset
- Cora
- CiteSeer
- Amazon/Photo
- WikiCS
- PPI

## Finding the best threshold
- Threshold has a big influence on the accuracy </br>
It decides how many fake edges are made.
If it is too big, model is almost same with original GAT.
If it is too small, too many edges are made even they aren’t from important nodes.
<img src="https://user-images.githubusercontent.com/59245409/120894933-c7b54e80-c655-11eb-9a4e-5916d299c002.png" width="800">


## Result
<img src="https://user-images.githubusercontent.com/59245409/120894847-72793d00-c655-11eb-8657-ca34f61d3884.png" width="600">
<img src="https://user-images.githubusercontent.com/59245409/120894992-00552800-c656-11eb-923a-1d5a909a0b43.png" width="400">

## Variants
### CA-GAT<sub>ADD</sub>
<img src="https://user-images.githubusercontent.com/59245409/120895127-8f624000-c656-11eb-889b-dcbbefa8e083.png" width="600">
<img src="https://user-images.githubusercontent.com/59245409/120895130-925d3080-c656-11eb-8a13-942a598d4476.png" width="600">

### CA-GAT<sub>MULT</sub>
<img src="https://user-images.githubusercontent.com/59245409/120895138-97ba7b00-c656-11eb-824b-fa096afcbdbb.png" width="600">
<img src="https://user-images.githubusercontent.com/59245409/120895140-99843e80-c656-11eb-8cca-0fe117e1e790.png" width="600">

### Result
<img src="https://user-images.githubusercontent.com/59245409/120895150-a43ed380-c656-11eb-986f-fb8146ea716f.png" width="600">

