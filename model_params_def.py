from improvelib.utils import str2bool

preprocess_params = [

]
train_params = [
    {"name": "num_cores",
     "type": int,
     "default": 8,
     "help": "num_cores",
    },
    {"name": "inDrop",
     "type": float,
     "default": 0.0001,
     "help": "inDrop",
    },
    {"name": "DSN_1",
     "type": str,
     "default": "2048-4096-2048",
     "help": "layers of Drug Synergy Network 1",
    },
    {"name": "DSN_2",
     "type": str,
     "default": "2048-4096-2048",
     "help": "layers of Drug Synergy Network 2",
    },
    {"name": "SPN",
     "type": str,
     "default": "2048-1024",
     "help": "layers of Synergy Prediction Network",
    },
]
infer_params = [
    {"name": "num_cores",
     "type": int,
     "default": 8,
     "help": "num_cores",
    },
]