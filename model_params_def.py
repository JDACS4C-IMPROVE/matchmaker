from improvelib.utils import str2bool

preprocess_params = [

]
train_params = [
    {"name": "num_cores",
     "type": int,
     "default": 8,
     "help": "num_cores",
    },
    {"name": "arch",
     "type": str,
     "default": "./architecture.txt",
     "help": "Architecute file to construct MatchMaker layers",
    },
    {"name": "inDrop",
     "type": float,
     "default": 0.0001,
     "help": "inDrop",
    },
    {"name": "drop",
     "type": float,
     "default": 0.5,
     "help": "drop",
    },
]
infer_params = [
    {"name": "num_cores",
     "type": int,
     "default": 8,
     "help": "num_cores",
    },
]