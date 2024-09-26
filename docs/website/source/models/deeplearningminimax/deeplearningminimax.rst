Deep Learning MiniMax
========================
This is an attempt to reproduce the product provided by
:cite:t:`David_2016`. In short, automated feature engineering supports a
chess board classifier which is used as the heuristic for game tree search. The
Feature engineering and classifier are achieved through supervised learning,
and the game tree search is implemented with AI search techniques.

Supervised Learning
^^^^^^^^^^^^^^^^^^^^^
The goal is to create a model capable of classifying a given position, or chess
board observation, as either winning or losing. More specifically, when
comparing two observations it should be able to deem which is winning and which
is losing. Then, incorporate this scheme as a heuristic evaluation for a classic
minimax game tree search. This effort is guided by :cite:t:`David_2016`.

**Autoencoding-** To start, a randomly initialized model, of any shape, will
know nothing about chess. It will know nothing about strategy, have no 
expectation of how a game might play out, and not understand the interdependence
of pieces on the board. This will make classifying one board state as better
than another quite difficult. So, first a feature extractor, something that
does understand the intricacies of the game, will be useful to inform our
classifier.

Autoencoding is one way to create a competent feature extractor, without relying
on a lot of manual feature engineering. In short, the original observation
serves as input and the objective label. In between, a model which allows for
less and less information to be passed after each layer, and its mirror image.
The model then encodes the observation in a way meaningful enough that the
mirror image can decode and rebuild the original observation. If the observation
is rebuilt well, the features captured must be useful and relevant
representations of the game.

To help train, we initially avoid forcing the model to reconstitute the
observation from the fully condensed form. Instead, for each batch, we take a
slice of the model to first train the larger layers, closer to the size of the
observation. Then we train on the batch again, including the next layer of the
model, and so on until the entire model is included. This reinforces the idea
of gradually more abstract concepts being captured by the feature extractor as
the observation is processed deeper within the model.

.. figure:: figures/autoencoder.png
    :scale: 100%
    :align: center

    Autoencoding training process per batch.


**Classification-**

MiniMax Search Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^
