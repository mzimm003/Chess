Environment Setup
======================

To ensure the reinforcement learning system setup is fully functional I start by
attempting to train an agent on TicTacToe. Environments, policies, and
algorithms have all been wrapped, but otherwise remain unchanged. As such, it is
expected a policy can be learned quickly given the relatively small state space
that exists for the game of tic-tac-toe. The environment is provided by the
Gymnasium Petting Zoo library. The observation is used, but the action mask is
not.

As of February 1, multiple training runs have been initiated, with varied
learning rates and for varied lengths, all to roughly the same effect. The win
rate of the policy would indicate no learning is taking place, as, after the
first iteration, the win rate is completely unchanged. With a stocastic sampling
exploration method, as provided by the PPO algorithm, this should be impossible.
Otherwise, loss would indicate the policy is in fact training, as it smoothly
falls to zero. Below are charts to demonstrate these findings. Why does policy
seem to update from iteration 1 to 2, but then never again? Notably, mean reward
quickly reaches -1 (absolute minimum). Two equally poor policies should be
committing illegal moves as they trade first move, reaping -1 reward themselves
leaving the other with 0, provides the expectation mean reward should be about
-0.5, and improve from their until a new opponent is generated.

Charts as of 02/01/2024:

.. figure:: figures/0201.png

    Not working results.

Eureaka moment strikes as of 02/02/2024! Every episode run provides an "Illegal
move" warning on the second step. Utilizing the IDE's debugger, I find Petting
Zoo's wrapper environments do not update the selected agent after the step,
despite the base environment prompting as much. This means, player_1 takes the
first action, and the second action, hence the second step always resulting in
an illegal move. To remedy this, I've attempted to modify the code base of
Petting Zoo's BaseWrapper class to include the following::

    @property
    def agent_selection(self) -> str:
        return self.env.unwrapped.agent_selection
    
    @agent_selection.setter
    def agent_selection(self, new_val: str):
        self.env.unwrapped.agent_selection = new_val

This has allowed the environment to appropriately pass the action back and forth
between players. While illegal moves are still being made, far fewer are being
reported, and training metrics are more stocastic as expected. The initial run
after this fix looks promising, and perhaps true training can finally commence.
Below are the results of the inital training after the fix (yellow) compared to
a couple runs done prior to the fix which demonstrate the peculiarities
described on 02/01/24:

.. figure:: figures/0202.png

    New results.

With this I can confidently work with new models and hyperparameter tuning
sessions in pursuit of various improvements in size and speed. Exciting times!