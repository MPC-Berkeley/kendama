# Cup-and-Ball (Kendama) Game Simulation

This repository contains the source code from our work on Learning to Play Cup-and-Ball with Noisy Camera Observations. We use dm_control and MuJoCo for the simulation environment. 

![](https://media2.giphy.com/media/WsvZkkNLGjUdpBAKux/giphy.gif)

Full text of our paper can be found [here](https://arxiv.org/abs/2007.09562):

Bujarbaruah, M., Zheng, T., Shetty, A., Sehr, M., & Borrelli, F. (2020). Learning to Play Cup-and-Ball with Noisy Camera Observations. arXiv preprint arXiv:2007.09562.

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

If you are running into an error where `mjdata.h` cannot be found, modify line 36 from `setup.py` to make sure the file path for MuJoCo is correct:
```
DEFAULT_HEADERS_DIR = '~/.mujoco/mujoco200/include'.format
```

3) Add the content from this repository to dm_control
```
$ git clone https://github.com/MPC-Berkeley/kendama
$ cp -r kendama/controls dm_control/
$ cp -r kendama/suite dm_control/suite
```
Note that dm_control/suite already contains `__init__.py` and this will overwrite it. The only additional content is the line to import the kendama model:
```
from dm_control.suite import kendama_catch_simulation
```

## Getting Started

Run the simulation from the dm_control/controls folder using
```
python3 run_kendama_simulation.py
```

## License
MIT License

Copyright (c) 2020 Model Predictive Control (MPC) Laboratory

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
