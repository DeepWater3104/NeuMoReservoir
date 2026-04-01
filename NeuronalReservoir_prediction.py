from neuron import h as nrn
from NeuronalReservoir import neuronalreservoir
from neuron.units import ms, mV
import numpy as np
import math
nrn.load_file("stdrun.hoc")

class neuronalreservoir_prediction(neuronalreservoir):
    def __init__(self, cell, prng, params):
        self.exc_num_synchro_syns = params['task']['exc_num_synchro_syns']
        self.exc_num_syn          = params['task']['exc_num_syn']
        self.exc_syn_weight       = params['task']['exc_syn_weight']
        super().__init__(cell, prng, params)
