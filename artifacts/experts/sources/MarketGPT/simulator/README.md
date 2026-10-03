The codes in this folder are the work of the authors of the papers 

* [ABIDES-Gym: Gym Environments for Multi-Agent Discrete Event Simulation and Application to Financial Markets](https://arxiv.org/pdf/2110.14771.pdf)
* [ABIDES: Towards High-Fidelity Market Simulation for AI Research](https://arxiv.org/abs/1904.12066)

The original code can be found at: https://github.com/jpmorganchase/abides-jpmc-public

I selectively chose and implemented parts of the ABIDES platform that were relevant for this project. The most important elements were the exchange agent and all order book functionality. These two components interact with the model through an interface (notebooks/simulate.ipynb) which includes error correction, data collection, etc. I also ported over some code to build custom agents so that I could run simulations with the model + background agents but I have not tested or built out the interface for this yet. Other elements like the simulation kernel and latency model have also been ported but not tested or implemented at this time.