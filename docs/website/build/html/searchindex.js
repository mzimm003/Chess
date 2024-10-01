Search.setIndex({"docnames": ["data/data", "index", "models/deeplearningminimax/deeplearningminimax", "models/deepmind/actionvalue", "models/deepmind/behaviouralcloning", "models/deepmind/statevalue", "models/generative/generative", "models/models", "models/reinforcementlearning/reinforcementlearning", "notes/dataloading/dataloading", "notes/environmentsetup/environmentsetup", "notes/evaluator/evaluator", "notes/hyperparametertuning/hyperparametertuning", "notes/modelsize/modelsize", "notes/notes", "notes/trainingsetup/trainingsetup", "plan/plan", "references"], "filenames": ["data/data.rst", "index.rst", "models/deeplearningminimax/deeplearningminimax.rst", "models/deepmind/actionvalue.rst", "models/deepmind/behaviouralcloning.rst", "models/deepmind/statevalue.rst", "models/generative/generative.rst", "models/models.rst", "models/reinforcementlearning/reinforcementlearning.rst", "notes/dataloading/dataloading.rst", "notes/environmentsetup/environmentsetup.rst", "notes/evaluator/evaluator.rst", "notes/hyperparametertuning/hyperparametertuning.rst", "notes/modelsize/modelsize.rst", "notes/notes.rst", "notes/trainingsetup/trainingsetup.rst", "plan/plan.rst", "references.rst"], "titles": ["Data", "Chess Bot", "Deep Learning MiniMax", "Action Value", "Behavioural Cloning", "State Value", "Generative", "Models", "Reinforcement Learning", "Chess Dataset", "Environment Setup", "Chess Board Evaluation", "Hyperparameter tuning", "Model Size", "Notes", "Algorithm", "Plan", "References"], "terms": {"goal": [0, 2, 15], "databas": [0, 7, 9], "support": [0, 2, 7, 17], "our": [0, 2, 7], "chess": [0, 2, 7, 12, 14, 15, 16, 17], "bot": [0, 7], "i": [0, 2, 9, 10, 11, 12, 13, 15, 16], "have": [0, 2, 7, 9, 10, 13, 15], "larg": [0, 11, 12, 13, 15], "number": [0, 2, 9, 12, 15], "exampl": [0, 2, 15], "from": [0, 2, 9, 10, 11, 12, 13, 15, 16], "strong": [0, 2, 7, 11], "player": [0, 2, 7, 9, 10, 13], "further": [0, 2, 9, 11, 12, 13, 15, 16], "import": [0, 9, 12, 13], "easili": [0, 13], "understood": 0, "machin": [0, 7, 17], "Then": [0, 2, 9, 11, 12, 15], "adequ": 0, "obtain": 0, "million": [0, 9, 15], "board": [0, 2, 9, 12, 13, 14, 15, 16], "posit": [0, 2, 9, 11, 15, 16], "reach": [0, 2, 9, 10], "high": [0, 11, 13], "elo": 0, "where": [0, 2, 9, 11, 15], "outcom": 0, "game": [0, 2, 9, 10, 11, 15], "known": [0, 2], "can": [0, 2, 9, 10, 11, 12, 13, 15], "effect": [0, 2, 10, 15], "repres": [0, 2, 9, 11, 13, 15], "seri": 0, "ye": 0, "statement": 0, "mostli": [0, 2, 13], "regard": 0, "each": [0, 2, 9, 11, 12, 13, 15], "piec": [0, 2, 9, 13], "its": [0, 2, 9, 11, 12, 13, 15], "within": [0, 2], "8x8": 0, "To": [0, 2, 9, 10, 11, 12, 13, 15], "achiev": [0, 2, 7, 12], "so": [0, 2, 7, 9, 11, 13, 15], "mani": [0, 2, 7, 9, 11], "comput": [0, 2, 9], "rate": [0, 10, 11, 13], "list": [0, 9], "record": [0, 9], "top": [0, 15], "portabl": 0, "notat": [0, 9], "pgn": [0, 9, 15], "format": [0, 9], "These": [0, 9, 13], "ar": [0, 2, 7, 9, 10, 11, 12, 13, 15], "freeli": 0, "avail": [0, 9, 15], "download": [0, 9], "compress": [0, 15], "get": [0, 13], "necessari": [0, 2, 9, 12, 13], "requir": [0, 9], "recreat": 0, "plai": [0, 2, 9], "per": [0, 2, 9, 12, 15], "file": [0, 9, 15], "pettingzoo": [0, 16, 17], "tbg": [0, 17], "21": [0, 17], "includ": [0, 2, 10, 12, 15, 16], "environ": [0, 9, 14, 15, 16], "thi": [0, 2, 9, 10, 11, 12, 13, 15, 16], "provid": [0, 2, 9, 10, 12, 13, 15, 16], "observ": [0, 2, 7, 10, 13, 15, 16], "state": [0, 2, 7, 9, 10, 11, 13, 15], "specif": [0, 2, 13, 15], "found": [0, 2, 9, 11], "here": [0, 11, 12, 13], "In": [0, 2, 9, 11, 13, 17], "short": [0, 2, 13], "set": [0, 2, 9, 11, 13, 15], "represent": [0, 2, 16], "also": [0, 9, 11, 12, 13], "refer": [0, 1, 9], "channel": [0, 13], "an": [0, 2, 9, 10, 11, 12, 13, 15, 16], "matrix": 0, "zero": [0, 2, 10, 13], "ones": 0, "": [0, 2, 9, 10, 11, 12, 13, 15, 17], "respect": [0, 11], "inform": [0, 2, 9, 11, 12, 15, 17], "like": [0, 2, 9, 11, 13, 16], "white": 0, "pawn": [0, 9], "There": [0, 7], "few": [0, 9, 11, 13], "worth": [0, 9], "clarifi": 0, "document": [0, 1], "0": [0, 2, 10, 11, 12, 13, 15], "castl": 0, "queensid": 0, "non": 0, "all": [0, 2, 9, 10, 11, 12, 15], "1": [0, 2, 9, 10, 11, 13, 15], "kingsid": 0, "2": [0, 2, 9, 10, 11, 12, 13, 15, 16, 17], "black": [0, 17], "3": [0, 2, 16, 17], "4": [0, 2, 11, 12, 13], "5": [0, 9, 10, 11, 15], "50": 0, "move": [0, 2, 9, 10, 11, 13, 16], "rule": [0, 2], "count": [0, 2], "One": 0, "flatten": [0, 15], "dimens": 0, "6": [0, 2, 9, 11], "7": [0, 13, 15, 16], "8": [0, 12, 13, 16], "knight": 0, "9": [0, 11], "bishop": 0, "10": [0, 2, 17], "rook": 0, "11": 0, "queen": 0, "12": 0, "king": 0, "13": 0, "oppon": [0, 2, 10], "14": 0, "15": 0, "16": 0, "17": [0, 9], "18": [0, 13], "19": 0, "ha": [0, 2, 10, 11, 15], "been": [0, 9, 10, 11, 13, 15], "seen": [0, 2, 11, 12], "befor": [0, 12], "20": [0, 2, 11, 13], "110": 0, "repeat": 0, "most": [0, 2, 9, 11, 12, 13, 15], "recent": 0, "least": [0, 2], "creat": [0, 2, 7, 9, 11, 12, 13, 15, 16], "matter": 0, "submit": 0, "describ": [0, 9, 10, 13, 15], "save": [0, 2, 9, 12], "appropri": [0, 10], "label": [0, 2, 11], "while": [0, 2, 9, 10, 11], "refin": [0, 9], "concentr": [0, 12, 13], "more": [0, 1, 2, 7, 9, 10, 11, 12, 13, 14, 15, 16], "relev": [0, 2, 11], "datapoint": 0, "On": [0, 2, 11], "advic": [0, 2], "david": [0, 2, 17], "et": [0, 2], "al": [0, 2], "dnw16": [0, 2, 9, 15, 16, 17], "avoid": [0, 2, 11, 12, 13], "veri": [0, 2, 11], "earli": [0, 12, 13], "first": [0, 2, 9, 10, 11, 13, 15, 16], "immedi": 0, "after": [0, 2, 10, 11], "captur": [0, 2], "which": [0, 2, 9, 10, 11, 12, 13, 15, 16], "end": [0, 2, 9, 12, 15, 16, 17], "draw": [0, 2], "measur": [0, 11, 15], "valu": [0, 2, 7, 9, 12, 13, 15], "process": [0, 2, 11, 14, 15, 16, 17], "300": 0, "000": [0, 2, 9, 11], "size": [0, 2, 9, 10, 14, 15], "gb": [0, 9], "875": 0, "812": 0, "69": 0, "consider": [0, 2, 9], "make": [0, 2, 9, 11, 13], "difficult": [0, 2], "pars": [0, 9, 15], "fit": [0, 9, 15], "onto": 0, "ram": 0, "onc": [0, 9, 11], "address": [0, 12], "difficulti": 0, "over": [0, 2, 9, 11, 13, 15], "1m": [0, 9], "even": [0, 9, 12, 13, 15], "when": [0, 2, 15], "limit": [0, 2, 9, 15], "custom": 0, "iter": [0, 2, 9, 10, 15], "leverag": [0, 16], "python": [0, 17], "panda": [0, 9], "librari": [0, 9, 10], "combin": [0, 9, 11], "multiprocess": [0, 15], "map": [0, 9, 13], "ensur": [0, 2, 9, 10, 11], "quickli": [0, 2, 9, 10], "effici": [0, 9], "us": [0, 2, 9, 10, 11, 12, 15, 16], "power": [0, 2, 16], "memori": [0, 9], "hard": [0, 15], "drive": [0, 15], "accumul": [0, 9], "up": [0, 9, 11, 16], "some": [0, 2, 7, 9, 11, 13, 15, 16], "specifi": 0, "my": [0, 2, 9, 11, 13, 14, 15], "case": [0, 2, 9, 11], "master": [0, 9], "keep": [0, 9, 12], "track": [0, 9], "contain": [0, 15], "metadata": [0, 9], "e": [0, 2, 12, 15, 17], "g": [0, 2], "label_file_1": 0, "200": 0, "procedur": [0, 13], "allow": [0, 2, 9, 10, 11], "scale": 0, "depend": [0, 15], "suppli": [0, 9], "space": [0, 2, 9, 10, 11, 12, 15, 16], "second": [0, 9, 10, 11], "issu": [0, 2, 13], "locat": 0, "load": [0, 12], "one": [0, 2, 9, 12], "time": [0, 2, 9, 10, 11, 12, 15], "next": [0, 2, 9, 11, 15], "complic": 0, "somewhat": [0, 12], "idea": [0, 2, 11], "gpu": [0, 15], "train": [0, 2, 9, 10, 12, 16], "model": [0, 1, 2, 9, 10, 11, 14], "much": [0, 9, 10, 11, 12, 15], "faster": [0, 15, 17], "than": [0, 2, 9, 13, 15, 16], "cpu": [0, 9, 15], "mean": [0, 2, 9, 10, 13, 15], "prudent": 0, "enabl": [0, 2, 16], "batch": [0, 2, 15], "usag": 0, "stabl": [0, 12], "indic": [0, 10, 15], "must": [0, 2, 11, 12], "serv": [0, 2], "order": [0, 11, 13], "select": [0, 1, 7, 9, 10, 11], "randomli": [0, 2, 11, 15], "split": [0, 9, 13, 15], "those": [0, 2, 9, 15], "sort": [0, 2], "two": [0, 2, 9, 10, 11, 15], "should": [0, 2, 10, 11, 12, 13, 15], "fail": 0, "complet": [0, 10, 11, 12], "take": [0, 2, 10, 11, 15], "lock": [0, 9], "paus": 0, "handl": [0, 9, 11], "open": 0, "close": [0, 11], "oldest": 0, "sinc": [0, 2, 11, 13], "pytorch": [0, 17], "ayh": [0, 17], "24": [0, 10, 17], "dataload": [0, 15], "being": [0, 2, 9, 10, 11, 13], "special": [0, 9], "made": [0, 2, 9, 10, 11, 15, 16], "how": [0, 2, 12, 13], "store": [0, 2], "dict": 0, "detail": [0, 12, 13], "repositori": 0, "plan": 1, "data": [1, 7, 9, 11, 12], "note": [1, 13, 15], "learn": [1, 7, 9, 10, 11, 16, 17], "about": [1, 2, 10, 11], "construct": 1, "section": [1, 15], "attempt": [2, 9, 10, 11, 13, 15], "reproduc": [2, 15, 16], "product": 2, "autom": 2, "featur": [2, 11, 12, 13, 15, 16], "engin": [2, 9, 16], "classifi": 2, "heurist": 2, "tree": [2, 16], "The": [2, 9, 10, 11, 12, 13, 15, 16], "through": [2, 9, 11, 13, 15, 17], "implement": 2, "ai": 2, "techniqu": 2, "capabl": [2, 9, 11, 12, 13, 16], "given": [2, 10, 11], "either": [2, 9, 15], "win": [2, 10, 11], "lose": [2, 11], "compar": [2, 9, 10, 12, 13], "abl": [2, 11], "deem": 2, "incorpor": 2, "scheme": [2, 9], "evalu": [2, 14, 16], "classic": 2, "effort": [2, 9], "guid": 2, "autoencod": [2, 11, 13, 15, 16], "start": [2, 9, 10, 15], "initi": [2, 10, 11, 13, 15], "ani": [2, 7, 9], "shape": [2, 12, 13], "know": [2, 12], "noth": 2, "It": [2, 9, 13], "strategi": 2, "expect": [2, 10, 11, 13, 16], "might": [2, 12, 13], "out": [2, 13], "understand": [2, 9, 16], "interdepend": 2, "better": [2, 9, 13, 15], "anoth": [2, 9], "quit": [2, 11], "extractor": [2, 11, 12, 13, 15], "someth": [2, 9], "doe": [2, 10], "intricaci": 2, "wai": [2, 9, 15], "compet": 2, "without": [2, 11, 16], "reli": [2, 9, 13], "lot": [2, 9], "manual": [2, 16], "origin": [2, 9, 15], "input": [2, 15], "object": 2, "between": [2, 9, 10, 11, 13, 15], "less": [2, 13, 14, 15, 16], "pass": [2, 10, 11, 15], "layer": [2, 11, 12, 13, 15, 16], "mirror": [2, 15], "imag": 2, "encod": 2, "meaning": [2, 15], "enough": [2, 15], "decod": [2, 15], "rebuild": 2, "If": 2, "rebuilt": 2, "well": [2, 9, 11, 12, 13], "help": [2, 9], "we": [2, 7, 12, 13], "forc": 2, "reconstitut": 2, "fulli": [2, 10, 11, 15], "condens": [2, 15], "form": 2, "instead": [2, 15], "slice": [2, 13], "larger": [2, 11, 12, 13], "closer": 2, "again": [2, 9, 10, 13], "until": [2, 10], "entir": [2, 15], "reinforc": [2, 7, 10, 17], "gradual": 2, "abstract": [2, 12, 13, 15], "concept": 2, "deeper": [2, 15], "reconstruct": [2, 15], "occur": [2, 12], "multipl": [2, 7, 9, 10], "were": [2, 9, 11], "squar": [2, 9, 12, 13, 15], "error": [2, 12, 15], "loss": [2, 10, 11, 12, 15], "illustr": 2, "fig": 2, "output": [2, 11, 15], "element": [2, 13], "regress": 2, "toward": 2, "true": [2, 10, 13], "fals": [2, 13], "natur": [2, 13], "problem": [2, 9], "structur": [2, 11], "classif": 2, "ultim": [2, 9, 11, 13, 15], "cross": [2, 11], "entropi": [2, 11], "calcul": [2, 9, 11, 13], "base": [2, 10, 11], "probabl": 2, "correct": [2, 9], "improv": [2, 9, 10, 13, 15], "below": [2, 7, 10, 11, 12, 13, 15], "result": [2, 10, 11, 12, 13, 16], "orang": 2, "v": 2, "green": 2, "aid": [2, 15], "would": [2, 9, 10, 11, 13, 16], "simul": 2, "consid": [2, 9, 12, 13], "valuat": [2, 16], "minu": 2, "moreov": 2, "rather": 2, "independ": 2, "rel": [2, 9, 10, 13, 15], "still": [2, 9, 10, 16], "rank": 2, "exact": [2, 15], "reason": [2, 12], "design": [2, 16], "choic": 2, "unclear": 2, "conceiv": 2, "simplifi": [2, 12], "try": [2, 9, 15], "recogn": 2, "thing": [2, 9, 12], "oppos": [2, 11, 13, 16], "absolut": [2, 9, 10], "task": 2, "identifi": [2, 9], "purpl": 2, "yellow": [2, 10], "current": [2, 9, 13, 16], "agent": [2, 10, 16, 17], "poorli": 2, "against": [2, 11], "human": [2, 9], "who": 2, "littl": 2, "appear": 2, "otherwis": [2, 9, 10], "quick": [2, 9], "hang": 2, "appar": 2, "counter": 2, "perhap": [2, 10], "signific": 2, "rare": [2, 9], "ever": [2, 12], "hung": 2, "reckless": 2, "part": [2, 9, 13, 15], "reward": [2, 10], "dataset": [2, 11, 12, 14, 15], "version": [2, 15], "wa": [2, 9, 11, 15], "taken": [2, 15], "omit": 2, "basi": [2, 7], "mislead": 2, "thei": [2, 10, 12, 13], "transient": 2, "advantag": [2, 12, 13], "other": [2, 9, 10, 12, 13, 17], "side": 2, "back": [2, 10, 13], "right": [2, 12], "awai": 2, "howev": [2, 11, 12], "am": [2, 11], "confid": [2, 10], "exclud": 2, "see": [2, 7, 12, 13, 15], "do": [2, 10, 13], "lead": [2, 11, 12, 15], "victori": 2, "suspici": 2, "blind": 2, "spot": 2, "great": 2, "slow": [2, 9, 13], "perform": [2, 9, 11, 13], "metric": [2, 10], "far": [2, 10, 13], "depth": [2, 12, 14, 15], "exactli": 2, "regardless": 2, "greater": [2, 12], "unplay": 2, "u": 2, "finit": 2, "life": 2, "span": 2, "hope": [2, 12, 16], "hash": 2, "tabl": [2, 13], "unsucces": [], "differ": [2, 9, 11, 12, 15], "revisit": 2, "come": [3, 4, 5, 6, 8, 9, 12], "soon": [3, 4, 5, 6, 8], "\u00bd": [7, 14], "\u00bc": [7, 14], "\u215b": [7, 14], "\u00be": [7, 14], "\u215c": [7, 14], "\u215d": [7, 14], "\u215e": [7, 14], "_": [7, 14], "\u00b5": [7, 14], "\u03c9": [7, 14], "\u00aa": [7, 14], "\u00ba": [7, 14], "\u00b9": [7, 14], "\u00b2": [7, 14], "\u00b3": [7, 14], "With": [2, 7, 9, 10, 12, 15], "abil": [7, 12, 13], "full": [7, 9, 16], "best": [2, 7, 11, 12, 16], "world": 7, "creation": 7, "own": [7, 9], "approach": 7, "viabl": 7, "explor": [2, 7, 10, 12], "deep": [7, 11, 16, 17], "minimax": 7, "futur": [7, 13], "work": [7, 9, 10, 13], "action": [2, 7, 9, 10, 16], "behaviour": 7, "clone": 7, "gener": [7, 9, 10, 11], "accord": 9, "deepchess": [9, 11, 15, 16, 17], "collect": 9, "http": [9, 15, 17], "computerchess": [9, 15], "org": [9, 15, 17], "uk": [9, 15], "ccrl": [9, 15], "404": [9, 15], "month": 9, "300k": 9, "correctli": 9, "algebra": 9, "along": [9, 12, 15], "readabl": 9, "suit": 9, "pet": [9, 10, 13, 15], "zoo": [9, 10, 13, 15], "had": 9, "nearli": [2, 9], "row": 9, "column": 9, "trajectori": 9, "step": [9, 10, 12], "what": [2, 9, 12, 13, 15], "realli": 9, "need": [2, 9, 11], "ve": [9, 10], "simpli": [9, 11, 13, 15], "me": 9, "build": 9, "except": 9, "came": [9, 11], "challeng": 9, "hardwar": 9, "addition": 9, "introduc": [9, 13], "fullest": 9, "For": [9, 11, 13, 15], "clariti": 9, "anm": 9, "lea": 9, "largest": [9, 11, 13], "go": [9, 11, 12], "agnost": 9, "therefor": [9, 15], "sever": [9, 11, 12], "doubl": 9, "alreadi": [2, 9, 11, 12], "produc": [9, 11, 15, 16], "believ": 9, "optim": [2, 9, 11, 12], "meant": [9, 15], "legal": [2, 9], "elimin": 9, "inconsist": 9, "potenti": [9, 11, 12, 16], "addit": [2, 9], "context": 9, "same": [2, 9, 10, 11], "Of": 9, "suffici": [9, 11, 12], "onli": [2, 9, 11, 15, 16], "could": [9, 12], "extra": 9, "bit": [9, 11, 13], "whichev": 9, "clear": [2, 9], "ambigu": 9, "final": [9, 10, 11], "underpromot": 9, "distinct": 9, "thu": [9, 13], "straightforward": [2, 9, 15], "setup": [9, 12, 13, 14], "becom": [2, 9, 15], "simpl": [9, 11, 16], "fed": 9, "return": [2, 9, 10], "new": [9, 10], "perspect": 9, "test": [9, 15], "14k": 9, "both": [9, 11, 15], "surround": [9, 15], "run": [9, 10, 11, 13], "offer": 9, "pop": 9, "15mb": 9, "prove": [9, 11, 13], "2gb": 9, "parser": 9, "datafram": 9, "particularli": 9, "cost": [2, 9, 15], "down": 9, "method": [9, 10], "copi": [9, 15], "exist": [9, 10], "consum": 9, "access": 9, "defeat": [9, 12], "speed": [9, 10, 15], "disk": [9, 15], "fast": [9, 11], "kept": 9, "eventu": 9, "too": [9, 11], "32gb": 9, "similar": [9, 11, 12, 15], "vein": [9, 11], "thwart": 9, "massiv": 9, "singl": [9, 13], "folder": 9, "post": 9, "hoc": 9, "research": [9, 15], "ext4": 9, "suggest": [9, 13], "10m": 9, "configur": 9, "them": [2, 9], "lesson": [9, 11], "success": [2, 9, 15], "resourc": [9, 13], "manner": 9, "proper": 9, "brought": 9, "18hr": 9, "minut": 9, "6350": 9, "overst": 9, "act": 9, "estim": [2, 9, 12], "via": 9, "tqdm": 9, "themselv": [9, 10], "constant": [9, 11], "util": [9, 10, 15], "total": [9, 15], "hour": [9, 15], "follow": [9, 10, 16], "item": [9, 15], "109": 9, "910": 9, "61": 9, "839": 9, "519": 9, "32": 9, "711": 9, "system": [10, 17], "function": [10, 11], "tictacto": 10, "polici": 10, "algorithm": [10, 14], "wrap": 10, "remain": [10, 11], "unchang": 10, "As": [10, 11, 15], "small": [10, 12, 15], "tic": 10, "tac": 10, "toe": 10, "gymnasium": 10, "mask": 10, "februari": 10, "vari": [10, 12], "length": 10, "roughli": 10, "place": 10, "stocast": 10, "sampl": [10, 12, 13, 15], "ppo": 10, "imposs": [2, 10], "fact": 10, "smoothli": 10, "fall": 10, "chart": [10, 12], "demonstr": [10, 11, 12], "find": [2, 10, 15], "why": 10, "seem": [10, 11, 12], "updat": 10, "never": [2, 10, 11], "notabl": 10, "minimum": 10, "equal": [10, 12], "poor": 10, "commit": 10, "illeg": 10, "trade": 10, "reap": 10, "leav": 10, "02": 10, "01": 10, "2024": [10, 17], "Not": 10, "eureaka": 10, "moment": [10, 12], "strike": 10, "everi": [2, 10, 11], "episod": 10, "warn": 10, "id": 10, "debugg": 10, "wrapper": 10, "despit": [10, 15], "prompt": 10, "player_1": 10, "henc": 10, "alwai": [10, 13], "remedi": 10, "modifi": 10, "code": 10, "basewrapp": 10, "class": 10, "properti": 10, "def": 10, "agent_select": 10, "self": 10, "str": 10, "env": 10, "unwrap": 10, "setter": 10, "new_val": 10, "forth": [2, 10], "fewer": [2, 10, 12, 15], "report": 10, "fix": 10, "look": [2, 10, 14], "promis": 10, "commenc": 10, "init": 10, "coupl": [10, 15], "done": [2, 10, 11, 13], "prior": [2, 10], "peculiar": 10, "hyperparamet": [10, 14, 15], "tune": [10, 11, 14, 15], "session": 10, "pursuit": 10, "variou": 10, "excit": [10, 11], "phase": 11, "determin": [2, 11, 12, 16], "connect": [11, 15], "network": [11, 15, 17], "neural": [11, 15, 17], "diagram": 11, "tweak": 11, "readi": 11, "assum": 11, "subset": [11, 15], "suspect": [11, 15], "flip": 11, "gain": 11, "caus": [11, 13], "overfit": 11, "intend": 11, "alpha": [2, 11, 16], "beta": [2, 11, 16], "search": [11, 16], "highli": 11, "comparison": [11, 13], "assign": 11, "likelihood": 11, "categori": 11, "lean": 11, "stage": [11, 16], "long": 11, "adher": 11, "consist": 11, "neuron": 11, "activ": [11, 15], "restrict": 11, "especi": [11, 12], "qualit": 11, "view": 11, "share": 11, "adam": [11, 12], "topologi": 11, "paramet": 11, "sensit": 11, "proven": [11, 15], "switch": [2, 11], "maxim": [2, 11], "schedul": 11, "shrink": [11, 12, 13], "epoch": [11, 12, 13], "fine": [11, 12], "possibl": [2, 11], "cours": [2, 11], "mai": [2, 11, 13], "valid": [11, 15], "accuraci": [2, 11, 13], "1e": 11, "2e": 11, "5e": 11, "shown": 11, "figur": [11, 13], "unstabl": [11, 13], "rest": [2, 11, 15], "96": [11, 17], "face": 11, "pick": 11, "That": 11, "curv": 11, "sgd": 11, "surprisingli": 11, "decent": 11, "balanc": 11, "low": 11, "chanc": [11, 13], "diverg": 11, "obviou": 11, "point": [11, 12, 13, 15], "converg": [11, 12], "reduc": 11, "continu": [11, 13], "slowli": 11, "decai": [11, 12], "did": [11, 13], "enhanc": 11, "though": 11, "40": [11, 13], "experi": [11, 12, 13], "aggress": 11, "95": 11, "75": 11, "multipli": 11, "similarli": 11, "97": [2, 11], "mark": [11, 17], "chang": [2, 11], "30": 11, "hold": [11, 13], "25": 11, "unfortun": [11, 12, 13, 15], "fluctuat": 11, "none": 11, "98": 11, "due": [11, 13], "random": [11, 15], "therebi": 11, "local": 11, "maximum": 11, "halt": 11, "latest": 11, "now": [2, 12, 15], "adapt": 12, "width": [12, 15], "chosen": 12, "saddl": 12, "later": [12, 13], "just": 12, "higher": 12, "increas": 12, "compens": 12, "instabl": 12, "straight": 12, "forward": [12, 13], "smaller": [12, 13, 15], "256": [12, 13], "0001": [12, 13], "control": 12, "extract": [12, 15, 16], "infer": 12, "late": 12, "simpler": 12, "think": [12, 13], "cannot": [12, 15], "line": [2, 12], "corner": 12, "edg": 12, "etc": 12, "whether": 12, "intermedi": [12, 13], "answer": 12, "propos": 12, "level": [12, 13], "emphas": 12, "emphasi": [12, 13], "4096": [12, 13], "2048": [12, 13], "1024": [12, 13], "512": [12, 13], "128": [12, 13], "across": [12, 13], "finer": [12, 13], "previous": 13, "chose": 13, "slightli": 13, "000025": 13, "disadvantag": 13, "consequ": 13, "discuss": 13, "abov": 13, "lr": 13, "pictur": 13, "replac": [2, 13, 16], "bottom": 13, "kei": 13, "takeawai": 13, "suffer": 13, "complex": 13, "nuanc": 13, "vanish": 13, "gradient": 13, "manag": 13, "extent": [13, 15], "normal": 13, "wast": [2, 13], "overli": 13, "concern": 13, "relationship": 13, "finest": 13, "yet": 13, "lack": 13, "jump": 13, "beyond": 13, "medium": 13, "smallest": 13, "option": 13, "wors": 13, "establish": 13, "conduct": 13, "purpos": [13, 16], "project": [13, 16], "give": 13, "steadi": 13, "graph": [13, 17], "superb": 13, "score": 13, "precis": 13, "recal": 13, "decept": 13, "exception": 13, "111x8x8": 13, "tensor": 13, "111": 13, "group": 13, "mayb": 13, "64": 13, "cell": 13, "empti": 13, "turn": [2, 13], "imbal": 13, "affect": 13, "benefici": 13, "becaus": 13, "sens": [13, 15], "A": [2, 14], "curat": 14, "emul": 15, "reproduct": 15, "develop": [15, 16], "68": 15, "plenti": 15, "cite": 15, "around": 15, "7m": 15, "05": 15, "34m": 15, "reserv": 15, "fraction": 15, "2m": 15, "substanti": 15, "profil": 15, "under": 15, "bother": 15, "bottleneck": 15, "solid": 15, "confirm": 15, "suspicion": 15, "advanc": [15, 17], "script": 15, "show": 15, "call": 15, "dwarf": 15, "subsampl": 15, "choos": 15, "cover": 15, "comprehens": 15, "hopefulli": 15, "impact": [2, 15], "11m": 15, "huge": 15, "100": [2, 15], "dai": 15, "week": 15, "identif": 15, "solut": 15, "begin": 15, "feedforward": 15, "altern": 15, "linear": 15, "8x8x111": 15, "104": 15, "variabl": 15, "logit": 15, "increasingli": 15, "aim": 15, "By": 15, "preserv": 15, "destroi": 15, "togeth": 15, "float": 15, "differenti": 15, "propag": 15, "good": 16, "bad": 16, "applic": 16, "head": 16, "nn": 16, "stack": 16, "frame": 16, "previou": [2, 16], "jason": 17, "ansel": 17, "edward": 17, "yang": 17, "horac": 17, "he": 17, "natalia": 17, "gimelshein": 17, "animesh": 17, "jain": 17, "michael": 17, "voznesenski": 17, "bin": 17, "bao": 17, "peter": 17, "bell": 17, "berard": 17, "evgeni": 17, "burovski": 17, "geeta": 17, "chauhan": 17, "anjali": 17, "chourdia": 17, "Will": 17, "constabl": 17, "alban": 17, "desmaison": 17, "zachari": 17, "devito": 17, "elia": 17, "ellison": 17, "feng": 17, "jiong": 17, "gong": 17, "gschwind": 17, "brian": 17, "hirsh": 17, "sherlock": 17, "huang": 17, "kshiteej": 17, "kalambarkar": 17, "laurent": 17, "kirsch": 17, "lazo": 17, "mario": 17, "lezcano": 17, "yanbo": 17, "liang": 17, "yinghai": 17, "lu": 17, "ck": 17, "luk": 17, "bert": 17, "maher": 17, "yunji": 17, "pan": 17, "christian": 17, "puhrsch": 17, "matthia": 17, "reso": 17, "saroufim": 17, "marco": 17, "yukio": 17, "siraichi": 17, "helen": 17, "suk": 17, "suo": 17, "phil": 17, "tillet": 17, "eikan": 17, "wang": 17, "xiaodong": 17, "william": 17, "wen": 17, "shunt": 17, "zhang": 17, "xu": 17, "zhao": 17, "keren": 17, "zhou": 17, "richard": 17, "zou": 17, "ajit": 17, "mathew": 17, "gregori": 17, "chanan": 17, "peng": 17, "wu": 17, "soumith": 17, "chintala": 17, "dynam": 17, "bytecod": 17, "transform": 17, "compil": 17, "29th": 17, "acm": 17, "intern": 17, "confer": 17, "architectur": 17, "program": 17, "languag": 17, "oper": 17, "volum": 17, "asplo": 17, "april": 17, "url": 17, "asset": 17, "pytorch2": 17, "pdf": 17, "doi": 17, "1145": 17, "3620665": 17, "3640366": 17, "omid": 17, "nathan": 17, "netanyahu": 17, "lior": 17, "wolf": 17, "automat": 17, "page": 17, "88": 17, "springer": 17, "publish": 17, "2016": 17, "dx": 17, "1007": 17, "978": 17, "319": 17, "44781": 17, "0_11": 17, "j": 17, "terri": 17, "benjamin": 17, "nathaniel": 17, "grammel": 17, "jayakumar": 17, "ananth": 17, "hari": 17, "ryan": 17, "sullivan": 17, "lui": 17, "santo": 17, "clemen": 17, "dieffendahl": 17, "carolin": 17, "horsch": 17, "rodrigo": 17, "perez": 17, "vicent": 17, "gym": 17, "multi": 17, "34": 17, "15032": 17, "15043": 17, "2021": 17, "farama": 17, "exactor": 2, "led": 2, "99": 2, "infinit": 2, "searchabl": 2, "realiti": 2, "ahead": 2, "bolster": 2, "prune": 2, "deepen": 2, "analysi": 2, "minim": 2, "respons": 2, "goe": 2, "exponenti": 2, "suppos": 2, "sometim": 2, "often": 2, "thereaft": 2, "add": 2, "motiv": 2, "branch": 2, "suboptim": 2, "fantas": 2, "sequenc": 2, "checkmat": 2, "thought": 2, "vice": 2, "versa": 2, "happen": 2, "upon": 2, "dismiss": 2, "worst": 2, "shift": 2, "let": 2, "pre": 2, "actual": 2, "overal": 2, "accomplish": 2, "b": 2, "peek": 2, "partial": 2, "unsuccess": 2}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"data": [0, 13, 15], "engin": 0, "gather": 0, "preprocess": 0, "The": 0, "stat": 0, "tricki": 0, "bit": 0, "chess": [1, 9, 11], "bot": 1, "plai": 1, "A": 1, "deep": 2, "learn": [2, 8, 12, 13], "minimax": 2, "supervis": 2, "search": 2, "algorithm": [2, 15], "futur": 2, "work": 2, "action": 3, "valu": [3, 5], "behaviour": 4, "clone": 4, "state": 5, "gener": 6, "model": [7, 12, 13, 15], "reinforc": 8, "dataset": 9, "sourc": 9, "process": 9, "translat": 9, "gain": 9, "observ": 9, "multiprocess": 9, "ram": 9, "manag": 9, "result": 9, "environ": 10, "setup": 10, "board": 11, "evalu": 11, "train": [11, 13, 15], "hyperparamet": [11, 12], "tune": 12, "batch": 12, "size": [12, 13], "rate": 12, "all": 13, "curv": 13, "By": 13, "measur": 13, "peculiar": 13, "qualit": 13, "analysi": 13, "auto": [13, 15], "encod": [13, 15], "valid": 13, "best": 13, "note": 14, "plan": 16, "refer": 17}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinxcontrib.bibtex": 9, "sphinx": 58}, "alltitles": {"Data": [[0, "data"], [15, "data"]], "Data Engineering": [[0, "data-engineering"]], "Data Gathering": [[0, "data-gathering"]], "Data Preprocessing": [[0, "data-preprocessing"]], "The Stats": [[0, "the-stats"]], "The Tricky Bit": [[0, "the-tricky-bit"]], "Chess Bot": [[1, "chess-bot"]], "Play A Bot": [[1, "play-a-bot"]], "Action Value": [[3, "action-value"]], "Behavioural Cloning": [[4, "behavioural-cloning"]], "State Value": [[5, "state-value"]], "Generative": [[6, "generative"]], "Models": [[7, "models"]], "Reinforcement Learning": [[8, "reinforcement-learning"]], "Chess Dataset": [[9, "chess-dataset"]], "Source": [[9, "source"]], "Processing": [[9, "processing"]], "Translation": [[9, "translation"]], "Gaining Observations": [[9, "gaining-observations"]], "Multiprocessing and RAM Management": [[9, "multiprocessing-and-ram-management"]], "Result": [[9, "result"]], "Environment Setup": [[10, "environment-setup"]], "Chess Board Evaluation": [[11, "chess-board-evaluation"]], "Training": [[11, "training"]], "Hyperparameters": [[11, "hyperparameters"]], "Hyperparameter tuning": [[12, "hyperparameter-tuning"]], "Batch Size and Learning Rate": [[12, "batch-size-and-learning-rate"]], "Model Size": [[12, "model-size"], [13, "model-size"]], "All Model Learning Curves": [[13, "all-model-learning-curves"]], "Learning Curves By Size": [[13, "learning-curves-by-size"]], "Measurement peculiarities": [[13, "measurement-peculiarities"]], "Qualitative analysis": [[13, "qualitative-analysis"]], "Auto-encoded Training Data": [[13, "auto-encoded-training-data"]], "Auto-encoded Validation Data": [[13, "auto-encoded-validation-data"]], "Auto-encoded Training Data - Best Models": [[13, "auto-encoded-training-data-best-models"]], "Auto-encoded Validation Data - Best Models": [[13, "auto-encoded-validation-data-best-models"]], "Notes": [[14, "notes"]], "Algorithm": [[15, "algorithm"]], "Model": [[15, "model"]], "Auto Encoder Training": [[15, "auto-encoder-training"]], "Plan": [[16, "plan"]], "References": [[17, "references"]], "Deep Learning MiniMax": [[2, "deep-learning-minimax"]], "Supervised Learning": [[2, "supervised-learning"]], "MiniMax Search Algorithm": [[2, "minimax-search-algorithm"]], "Future Work": [[2, "future-work"]]}, "indexentries": {}})