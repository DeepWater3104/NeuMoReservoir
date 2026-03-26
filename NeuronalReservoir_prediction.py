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


if __name__ == '__main__':
    from DataGenerator import sin_datagenerator, MackeyGlass_datagenerator
    from sklearn.metrics import mean_squared_error

    prng = np.random.default_rng(1234)
    #prng = np.random.default_rng(123)

    freq = 0.01
    sinwave = sin_datagenerator(params, freq, prng)
    #mgdata = MackeyGlass_datagenerator(params, prng)

    import sys
    sys.path.append("./cells/cell1/")
    import os
    from neuron_simulation import run_nrnivmodl, extract_template_name, get_hoc_morph_for_emodel_folder, check_line_in_file
    import logging
    
    relativepath_cell1 = "./cells/cell1/"
    
    # Check if the mecahnisms folder exsists
    if not(os.path.exists(relativepath_cell1 + "mechanisms")):
        logging.error("No mechanisms directory found.")
    # Run the command
    run_nrnivmodl(relativepath_cell1)
    #from neuron import rxd
    
    # Get the hoc file
    hoc_path, morph_path = get_hoc_morph_for_emodel_folder(relativepath_cell1)
    #Load the standard hoc file and the custom hoc file for the model
    nrn.load_file('stdrun.hoc')
    nrn.load_file(hoc_path.as_posix())
    #Extract the template name from the hoc file, and create a cell instance
    method_name = extract_template_name(hoc_path.as_posix())
    #Based on the number of arguments in the template, initialize the cell.
    if check_line_in_file(hoc_path.as_posix(), "gid = $1"):
        cell = getattr(nrn, method_name)(0, relativepath_cell1 + "morphology",morph_path.name )
    else:
        cell = getattr(nrn, method_name)( relativepath_cell1 + "morphology",morph_path.name )


    reservoir = neuronalreservoir_prediction(cell, sinwave, prng, params)
    #reservoir = neuronalreservoir(cell, mgdata, prng, params)

    #print('"       mean interval of input data            "', np.mean(np.abs(np.diff(mgdata.get_inputdata()))))
    print('"       mean interval of input data            "', np.mean(np.abs(np.diff(sinwave.get_inputdata()))))
    print('"resolution of data-synapse correspondence(exc)"', reservoir.datagenerator.exc_dataresolution)
    print('"resolution of data-synapse correspondence(inh)"', reservoir.datagenerator.inh_dataresolution)
    reservoir.generate_response()

    output_transient = reservoir.readout_transient()

    reservoir.optimize()
    print(reservoir.W[reservoir.num_states])

    output_aftertraining_training = reservoir.readout_training()
    output_aftertraining_test = reservoir.readout_test()

    print('" training MSE "', mean_squared_error(output_aftertraining_training, reservoir.datagenerator.trainingdata_target))
    print('" test MSE     "', mean_squared_error(output_aftertraining_test, reservoir.datagenerator.testdata_target))


    # plot membrane pontential
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(16,6))
    ax1 = fig.add_subplot(4, 1, 1)
    #v_rec_list = np.empty((0, 0))
    #for v in reservoir.v_rec_list:
    #    v_rec_list = np.concatenate(( v_rec_list, np.array(v.to_python()).reshape(len(v.to_python()),1) ), axis=1)

    #print(np.shape(v_rec_list))

    #ax1.pcolor(np.array(reservoir.t_rec.to_python()), v_rec_list)
    #ax1.pcolor(v_rec_list)
    for v_rec in reservoir.v_rec_list:
        ax1.plot(reservoir.t_rec, v_rec)
    ax1.set_ylabel('V[mV]')
    ax1.get_xaxis().set_visible(False)

    ax2 = fig.add_subplot(4, 1, 2)
    exc_sum_syncurrent = reservoir.get_sum_syncurrent('exc')
    inh_sum_syncurrent = reservoir.get_sum_syncurrent('inh')
    ax2.plot(reservoir.t_rec, exc_sum_syncurrent, label='sum of exc synaptic current')
    ax2.plot(reservoir.t_rec, inh_sum_syncurrent, label='sum of inh synaptic current')
    ax2.set_ylabel('synaptic current')
    ax2.legend()
    ax2.get_xaxis().set_visible(False)

    ax3 = fig.add_subplot(4, 1, 3)
    ax3.plot(reservoir.datagenerator.t, reservoir.datagenerator.get_inputdata(), label='input data')
    ax3.set_ylabel('input')
    ax3.get_xaxis().set_visible(False)

    ax4 = fig.add_subplot(4, 1, 4)
    #ax4.plot(reservoir.datagenerator.t[int(reservoir.datagenerator.len_transientdata):], reservoir.datagenerator.get_targetdata(), label='ground truth')
    #ax4.plot(reservoir.datagenerator.t[int(reservoir.datagenerator.len_transientdata):int(reservoir.datagenerator.len_transientdata+reservoir.datagenerator.len_trainingdata)], output_aftertraining_training, label='prediction on training set')
    #ax4.plot(reservoir.datagenerator.t[int(reservoir.datagenerator.len_transientdata+reservoir.datagenerator.len_trainingdata):], output_aftertraining_test, label='prediction on test set')
    ax4.plot(reservoir.datagenerator.t[int(reservoir.datagenerator.len_transientdata):], reservoir.datagenerator.get_targetdata())
    ax4.plot(reservoir.datagenerator.t[int(reservoir.datagenerator.len_transientdata):int(reservoir.datagenerator.len_transientdata+reservoir.datagenerator.len_trainingdata)], output_aftertraining_training)
    ax4.plot(reservoir.datagenerator.t[int(reservoir.datagenerator.len_transientdata+reservoir.datagenerator.len_trainingdata):], output_aftertraining_test)
 
    ax4.set_ylim(min(reservoir.datagenerator.get_inputdata()), max(reservoir.datagenerator.get_inputdata()))
    ax4.set_ylabel('output')
    ax4.set_xlabel('Time[ms]')
    ax4.legend()

    ax1.set_xlim((reservoir.datagenerator.len_transientdata+reservoir.datagenerator.len_trainingdata/2)*reservoir.bin_width, (reservoir.datagenerator.len_transientdata+reservoir.datagenerator.len_trainingdata+reservoir.datagenerator.len_testdata)*reservoir.bin_width )
    ax2.set_xlim((reservoir.datagenerator.len_transientdata+reservoir.datagenerator.len_trainingdata/2)*reservoir.bin_width, (reservoir.datagenerator.len_transientdata+reservoir.datagenerator.len_trainingdata+reservoir.datagenerator.len_testdata)*reservoir.bin_width )
    ax3.set_xlim((reservoir.datagenerator.len_transientdata+reservoir.datagenerator.len_trainingdata/2)*reservoir.bin_width, (reservoir.datagenerator.len_transientdata+reservoir.datagenerator.len_trainingdata+reservoir.datagenerator.len_testdata)*reservoir.bin_width )
    ax4.set_xlim((reservoir.datagenerator.len_transientdata+reservoir.datagenerator.len_trainingdata/2)*reservoir.bin_width, (reservoir.datagenerator.len_transientdata+reservoir.datagenerator.len_trainingdata+reservoir.datagenerator.len_testdata)*reservoir.bin_width )

    plt.tight_layout()
    plt.savefig("output.png")
