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

.. _autoencoder:
.. figure:: figures/autoencoder.png
    :scale: 100%
    :align: center

    Autoencoding training process per batch.

The reconstruction of the board can occur in multiple ways. Initially, the
feature extractors were trained using mean squared error as a loss, as
illustrated in :numref:`autoencoder`. Specifically, the decoder creates an
output for each element of the input, and regresses toward the input's true (1)
or false (0) states. More naturally, and effectively, the problem is structured
as a classification for each element, as either true or false. So, ultimately
cross entropy loss is used, where two outputs are created for each input element
to represent the false state and true state, and error is calculated based on
the probability of the correct state. The improvement can be seen below in
:numref:`autoencoder_loss_diff`.

.. _autoencoder_loss_diff:
.. figure:: figures/autoencoder_loss_diff.png
    :align: center

    Autoencoding training results mean squared error loss (orange) vs. cross 
    entropy loss (green).

**Classification-** The classifier should aid in a minimax algorithm, which
would search the game tree by simulating moves and considering their result.
Classically, the consideration would be a heuristic valuation of the board 
state, e.g. a count of my pieces minus my opponents pieces. In this case, the
classifier will serve as the heuristic. Moreover, rather than provide an
independent value of a board state, it will compare board states relative to one
another, still allowing for board states to be ranked. The exact reason for this
design choice is unclear in :cite:t:`David_2016`, but its conceivable the
problem is simplified trying to train something capable of recognizing one thing
better than another, opposed to an absolute understanding of value.

The task then, given a board state which ended in a win and one that ended in a
loss, pass each board through the feature extractor, and pass the features
through a classifier which identifies which board wins and which loses. This
process is illustrated in :numref:`classifier`, using cross entropy loss.

.. _classifier:
.. figure:: figures/classifier.png
    :scale: 100%
    :align: center

    Classifier training process per batch.

The impact of the loss change for the feature extractor is further made clear in
the training results of the classifier (see :numref:`classifier_ae_loss_diff`).
Where the original feature exactor supported a classifier in achieving an
accuracy of nearly 97%, the switch led to an accuracy of nearly 99%.

.. _classifier_ae_loss_diff:
.. figure:: figures/classifier_ae_loss_diff.png
    :align: center

    Classifier training results using feature extractor trained on mean squared
    error loss (purple) vs. feature extractor trained on cross entropy loss
    (yellow).

MiniMax Search Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^
With a classifier capable of comparing two board states, a search can be
performed over the game space for the best moves. With infinite time and power,
of course the entire game space is searchable. In reality, the search is limited
to as many moves ahead as possible. To bolster the effectiveness of the search,
two additional techniques are included in the minimax search, alpha-beta pruning
and iterative deepening.

The minimax search is straightforward. On the agents turn, considering all legal
moves, it can simulate what the board would look like with each of those moves.
The classifier can then provide a sorting of those simulated boards, such that
the best, maximal, move can be made. A better analysis still is, for all those
boards simulated, simulate what the board would look like after the opponent
makes any legal move. Where the opponent would like to minimize the agent's
success, the sort enabled by the classifier can reasonably determine the
best move for the opponent the same way it does for the agent. Now, the agent
is instead determining its best, maximal, move based on boards which include the
opponents optimal, minimizing, response. So it goes, for as many moves back and
forth as compute power and time allow. The agent will do its best to find the
maximal move of the minimal moves of the maximal moves... of the minimal moves.

The exponential nature of this search should be clear. Supposing there are 10
legal moves for each player in any state (sometimes there are fewer, often there
are many more), then every consideration of the agents moves then the opponents
would be 100 (10\ :sup:`2`) boards. Every move thereafter, for either side,
adds another zero. To look 6 moves ahead, 3 for the agent, 3 for its opponent,
is 1,000,000 boards. Quickly, it becomes impossible to fully calculate the
best move. With this motivation, we include alpha-beta pruning. This helps
avoid searching branches known to be suboptimal based on other branches already
explored. In short, the agent should not waste time fantasizing about a winning
sequence of moves that an opponent minimizing its success would never allow. "I
have checkmate in 20 only if the opponent does nothing productive in that time"
is a line of thought that need not be explored. Since the classifier allows the
agent to sort board states by an estimate of most to least winning, and vice
versa, it should first happen upon the most relevant branches and be able to
quickly dismiss the rest.

However, since the sort based on the classifier is only a best estimate based on
the classifier's training, the idea of best and worst board states may shift as
the agent simulates a greater number of moves (simulates deeper). Then, to help
ensure the sort does provide the most relevant branches first, the search can be
done iteratively deeper. This lets less deep simulations do a pre-sort for more
deep simulations, while costing relatively little time given the exponential
nature of the search. So if the pre-sort can help prune more branches than
otherwise, it will actually save time overall. To accomplish this, for each
board state (B) reached in the search, part of the search will be over that
board's actions. Returned from that part of the search will be the expected
board state reached after some depth. This expected board state is stored as a
heuristic observation. The next time B is reached in a search it will be able
to use the heuristic observations as a peek into the future to sort its legal
actions, prior to searching through them, better allowing alpha and beta to
prune the search tree. Again, expected board states will be returned from the
partial search over this action from this board, and if the depth of the search
from this board is deeper than it was previous the returned expected board state
will replace the current heuristic observation for this action.

Future Work
==============
Currently, the agent plays very poorly. When playing against humans who know
little more than the rules of chess, it appears capable of drawing. Otherwise,
it is quick to hang pieces for no apparent counter play. Perhaps more
significant, it rarely, if ever, captures pieces hung by its human opponent.
This makes reckless play on the part of the human opponent quite rewarding. When
creating the dataset on which this version is trained, the advice of
:cite:t:`David_2016` was taken to omit captures on the basis, "capture moves are
misleading as they mostly result in a transient advantage since the other side
is likely to capture back right away." However, since the board classification
is based on the end result of the game, I am not confident it is necessary to
exclude captures on such a basis, as the classifier will see many examples where
captures do not lead to victory. Further, I am suspicious the agent has a blind
spot for capturing pieces as a strong strategy for furthering its objective.
Then, another version of the agent should be created where the dataset does not
exclude captures.

Another great issue for this agent is its slow performance in the minimax
algorithm. :cite:t:`David_2016` does not provide metrics on their performance
as far as depth or timings, so it is difficult to set expectations exactly.
Regardless, as any depth greater than 3 is near unplayable for those of us with
finite life spans, it is hoped improvements can be found. Initial attempts at
a hash table were unsuccessful at creating a difference in timing, so this
should be revisited.