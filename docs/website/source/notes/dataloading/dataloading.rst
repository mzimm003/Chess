.. _dataloading:
Chess Dataset
===============

Source
-----------
According to DeepChess :cite:p:`David_2016`, chess games can be collected from
http://computerchess.org.uk/ccrl/404/. These are games played by chess engines
and are available for download. In downloading all games, I have available over
2 million games. 

Processing
--------------
I started with processing a few games, then a month's worth (~300K), to ensure
games would be processed correctly. The download from computerchess.org.uk
provides games in algebraic notation, along with some metadata. Algebraic
notation is useful for human readability, but is not well suited for the
environment supplied by Petting Zoo. Then I first had to translate moves from
algebraic notation to something the chess learning environment would understand
(an action space of nearly 5,000 moves, representing some combination of row,
column, and trajectory). Next, to gain data for the first and second steps of
training my model, what I really need are the observations of the board from
these games I've downloaded. So, with translated actions, I simply play the game
through the learning environment and save the observations made. This allowed me
to build a database for the first two steps of training, except that 2 million
games came with its own challenges as I reached hardware limitations. So, I
additionally had to make considerations for RAM management and introduce
multiprocessing to make the fullest use of my CPU.

Translation
^^^^^^^^^^^^^
For clarity, I will refer to the move made in the games being translated which
are provided in algebraic notation as ANM, and moves representing actions in the
Petting Zoo learning environment as LEA. The largest challenge in translation is
that algebraic notation is relative to the current state of the board and
describes where a piece is going, where the action space of the Petting Zoo
chess environment is absolute (agnostic to the board state) and therefore relies
on where the piece is. Several means of calculation were considered, most of
which doubled efforts the learning environment was already making in producing
the observations, so I believe the most optimal meant starting with the
available LEA legal moves and eliminating those inconsistent with the ANM. This
meant identifying the player making the move, the piece, potentially additional
piece context, and the square on which the piece would end up, all from the ANM.
Then, from the LEA legal moves I select all that end at the same square as the
ANM. Of those, I identify the piece making the LEA, comparing the origin of the
move to the piece positions according to a learning environment observation of
the board, and further refine the LEA legal moves to those made by the correct
piece. For the most part, this is sufficient to identify the only LEA legal
action the ANM could be describing, but in some cases two of the same piece
could move to the same square. This is where the extra piece context would come
into play, which is simply an extra bit of information in the ANM describing
either the origin column or row, whichever would clear up ambiguity between
multiple pieces. Finally, there is also the rare move of a pawn's
underpromotion, which is distinct in the LEAs as it is in ANM, thus
straightforward to consider.

Gaining Observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^
With move to action translation setup, gaining observations becomes as simple as
playing the game through the learning environment and recording the observations
along the way. With a list of ANMs for any game, the moves are fed to the
translator one at a time, and the returned action is fed to the environment, and
a new state of the board can be observed. It is important in the case of the
Petting Zoo environment that the player's perspective also be recorded, as the
observation is relative to the player.

Multiprocessing and RAM Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once the games were setup so some simple parsing of the
http://computerchess.org.uk/ccrl/404/ provided "pgn" file fed moves to the
environment and translator, test files of 5 games, and 14K games both performed
well. Better still once I introduced multiprocessing surrounding the observation
recording. Once I attempted to run the full file of all games (1M+)
computerchess had to offer, new problems quickly popped up. Parsing a 15MB file
(14K games) proved much different than a 1.2GB file. To make a more efficient
parser I made an iterator which relies on a pandas DataFrame to parse one game
from another, which can be fed to a multiprocess mapping for some quick
formatting. The iterator was particularly helpful in keeping memory cost down.
Any other method of trying to parse the pgn file in multiple processes required
a copy of the file to exist for each process, quickly consuming memory, or a
master copy which would lock as any process accessed it, defeating a lot of the
speed otherwise gained. Then observations were processed and saved to the disk,
while metadata would accumulate in the multiprocessing of the environments.
Again this works well and works fast for 14K games, but for 1M+ games, simply
the metadata kept in memory would eventually become too much for my available
32GB of RAM. Finally then, it was necessary that even the metadata be split, and
a master file be created to keep track of which file would have information for
which game/observation. In a similar vein, I found even my observation saving
scheme would be thwarted by the massive number of games, as only so many files
fit into a single folder (some post-hoc research on ext4 suggests potentially
10M with a special configuration), so observations would have to be split,
collecting them into a folder per game.

Result
--------------
The lessons learned in handling and generating a massive data set have been
many. Ultimately, I was successful in managing the use of my computer's resource
such that it was capable of producing observations for all 1M+ games in a timely
manner. Introducing a proper iterator and multiprocessing to the pgn parsing
process brought its time from 18hrs to 17 minutes, a 6350% improvement. Further,
these timings are likely overstated as the act of time estimates (via the tqdm
library) themselves was found to slow things considerably, not allowing for
full, constant utilization of the CPU. In all, total run time was 6 hours,
producing the following:

    ========================= =================
    Item                      Value
    ========================= =================
    Games Processed           1,109,910
    Observations Produced     61,839,519
    Metadata Size             32 GB
    Observations Size         711 GB
    ========================= =================