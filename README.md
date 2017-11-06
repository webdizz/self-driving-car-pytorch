self-driven-car-pytorch
========================

This repository is a demo of implementation of self-driving car using [Open.AI](https://openai.com/) [Universe](https://github.com/openai/universe) and [PyTorch](http://pytorch.org/) to support [Can I do AI?..](https://www.slideshare.net/IzzetMustafaiev/can-i-do-ai) presentation.

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

During training there are `score` and `reward` metrics will be pushed to InfluxDB running in Docker container along with Grafana to visualize learning process. 
There is also a dashboard used for this `grafana-dashboard.json`.
To run monitoring execute bellow command from project folder:

    docker-compose up

Credits
------------------------    

* PyTorch reinforcement [examples](https://github.com/pytorch/examples/tree/master/reinforcement_learning)
* OpenAI Universe [starter agent](https://github.com/openai/universe-starter-agent)
* Siraj Raval on [Self Driving Car](https://github.com/llSourcell/Self-Driving-Car-Demo)

TODO
------------------------    
* Adopt LSTM to remember previous actions and env states
* Investigate more MaxPooling and Dropout layers
* Adopt A3C model