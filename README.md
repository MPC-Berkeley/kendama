# Cup-and-Ball (Kendama) Game Simulation

This repository contains the source code from our work on Learning to Play Cup-and-Ball with Noisy Camera Observations. We use dm_control and MuJoCo for the simulation environment. 

![](https://media2.giphy.com/media/WsvZkkNLGjUdpBAKux/giphy.gif)

Full text of our paper can be found here:
```
@misc{bujarbaruah2020learning,
    title={Learning to Play Cup-and-Ball with Noisy Camera Observations},
    author={Monimoy Bujarbaruah and Tony Zheng and Akhil Shetty and Martin Sehr and Francesco Borrelli},
    year={2020},
    eprint={2007.09562},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```
## Installation
(Optional) Load a new virtual environment
```
python3 -m venv kendama_env
source kendama_env/bin/activate
```

1) Install MuJoCo according to the instructions from https://github.com/openai/mujoco-py.

2) Install dm_control with the develop option to modify files without having to reinstall every time.
```
$ git clone https://github.com/deepmind/dm_control
$ cd dm_control
$ python setup.py develop
```

3) Add the content from this repository to dm_control
```
$ git clone https://github.com/MPC-Berkeley/kendama
$ cp -r kendama/controls dm_control/
$ cp -r kendama/suite dm_control/suite
```
Note that dm_control/suite already contains \__init__.py and this will overwrite it. The only additional content is the line to import the kendama model:
```
from dm_control.suite import kendama_catch_simulation
```

4) Install dependencies
```
python3 -m pip install -r kendama/requirements.txt
```

## Getting Started

Run the simulation from the dm_control/controls folder using
```
python3 run_kendama_simulation.py
```