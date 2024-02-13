The purpose of this project is to create a chess engine. The engine will leverage transformers in the hope the 
attention mechanism will help the engine identify the what of a given state is most relevant to choosing the optimal
move. Then, ultimately the engine will be trained by reinforcement learning, playing better and better versions of 
itself. Borrowing from object detection, I hope to mimic structures (like DINO) which use the feature map of a classifier
backbone to inform an object detector. Specifically, the backbone will consider convolutions of the state of the chess
board, and be trained to identify good moves and bad moves (including legal vs illegal), or perhaps to approximate the
evaluation of another engine. The features learned in this backbone will then be used to inform the agent policy


# Backbone
...While classical convolutions may be helpful, it is likely they will restrict effectiveness for various piece (e.g.
those that move diagonally) as most feature maps will concentrate more locally on the board than is appropriate. This
may be addressed automatically considering a transformer based model, so here convolutions generally refers to the effect
of the shrinking, increasingly concentrated feature map, opposed to a strictly local window condensing information.

# Agent