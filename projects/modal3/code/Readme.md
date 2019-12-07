# Planning Peg-Solitaire based on images
Using a Gumbel-Softmax Categorical-Autoencoder a 33bit state-encoding has been learned.\
Analysing the found encoding shows that every Peg has been mapped to one bit:
<img src="image/sae_test2_9.png">

Following, these state-encodings where used to learn a 8bit action encoding:
<img src="image/aae_test_15.png">

Finally, using PU-learning, a state and transition classifier was trained.\
These networks decide if a state or transition is legal or illegal.

Corresponding Papers:\
[Classical Planning in Deep Latent Space: Bridging the Subsymbolic-Symbolic Boundary](https://arxiv.org/abs/1705.00154)\
[Towards Stable Symbol Grounding with Zero-Suppressed State AutoEncoder](https://arxiv.org/abs/1903.11277)

## Executing the code
All the above networks can be trained and their performance can be controlled in the *analysis.ipynb* jupyter-notebook.

A random Peg-Solitaire game can be simulated executing the *simulate.ipynb* notebook.\
Following parameters can be set:
- *randomSteps*: defines the number of moves executed
- *randomPath*: with this optional parameter the action number to be executed can be defined (this action has to be legal)
- *showAll*: setting this parameter to *True* shows all possible successor at each step

Three random steps:
<img src="image/path_0_2.png">
<img src="image/path_1_2.png">
<img src="image/path_2_0.png">

The given game simulation code can be easily modified to run a depth/breadth first search on the transition tree.