self-driven-car-pytorch
========================

This repository is a demo of implementation of self-driving car using [Open.AI](https://openai.com/) [Universe](https://github.com/openai/universe) and [PyTorch](http://pytorch.org/) to support [Can I do AI?..](https://www.slideshare.net/IzzetMustafaiev/can-i-do-ai) presentation at [JavaDay 2017](http://javaday.org.ua/izzet-mustafaiev-can-i-do-ml/) and here is a [video](https://youtu.be/qejP06Hesbk).

Prerequisite
------------------------

[Open.AI](https://openai.com/) [Universe](https://github.com/openai/universe) dependencies should be installed.

Usage
------------------------

Before to use we need to install [Anaconda](https://conda.io) based environment:

    conda env create -f env.yml

Then simply execute to run bot:

    source activate sd-car-pytorch
    python racer-ac.py

During training there are `score` and `reward` metrics will be pushed using [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) into *runs* folder. During training or later statistics can be analysed using [Tensorboard](https://github.com/tensorflow/tensorboard) at http://localhost:6006.

    tensorboard --logdir runs 

Model
------------------------    

There is also a trained model persisted in `racer-ac.pth.tar` that can be loaded automatically for testing or continue to learn.

Credits
------------------------    

* PyTorch reinforcement [examples](https://github.com/pytorch/examples/tree/master/reinforcement_learning)
* OpenAI Universe [starter agent](https://github.com/openai/universe-starter-agent)
* Siraj Raval on [Self Driving Car](https://github.com/llSourcell/Self-Driving-Car-Demo)

TODO
------------------------    
* Adopt A3C model, however it looks like ACKTR is more promising algorithm according to [https://arxiv.org/abs/1708.05144](https://arxiv.org/abs/1708.05144)
* Adopt experience replay 
* Add architecture diagram
* Add runs comparison
* Try to train on GPU on [FloydHub](https://www.floydhub.com/) or other GPU cloud.