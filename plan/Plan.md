The purpose of this project is to create a chess engine. The engine will follow the process of "DeepChess" (David, O. 2016*) providing an end-to-end learning agent capable of developing a position evaluation without manual feature design.

DeepChess is made up of 3 stages:
1. Autoencoder training to produce a feature extracter to understand chess positions. 
1. Valuation training to enable the feature extracter to understand good positions from bad.
1. Application of the position evaluator to in alpha beta tree search to determine best moves.

I would like to reproduce the first 2 stages, but hope to replace the alpha beta search with a more simple action head (some nn layers). Potentially results will be less powerful than DeepChess, but I still expect a capable agent enabled by a deep understanding of the board. Further, I will likely leverage the full observation space of the PettingZoo chess environment, which includes 8 stacked frames of the current position (and 7 previous), opposed to the representation of the board used by DeepChess which only included the current position.


* David, O., Netanyahu, N., & Wolf, L. (2016). DeepChess: End-to-End Deep Neural Network for Automatic Learning in Chess. Springer International Publishing.