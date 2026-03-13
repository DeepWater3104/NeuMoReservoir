from neuron import h as nrn
from neuron.units import ms, mV
import numpy as np
import math
nrn.load_file("stdrun.hoc")

class neuronalreservoir():
    def __init__(self, cell, datagenerator, prng, params):
        # instantialize neuron
        self.cell = cell
        #nrn.v_init = -70 * mV
        nrn.celsius = 36
        #nrn.dt = 0.005 * ms

        self.datagenerator = datagenerator

        self.bin_width   = params['bin_width']
        self.num_states = params['num_states']
        self.record_target = params['record_target']

        self.exc_num_syns         = params['exc_num_syns']
        self.exc_num_synchro_syns = params['exc_num_synchro_syns']
        self.exc_syn_weight       = params['exc_syn_weight']
        self.exc_syn_tau1         = params['exc_syn_tau1']
        self.exc_syn_tau2         = params['exc_syn_tau2']
        self.syn_mechanisms         = params['syn_mechanisms']

        self.inh_num_syns         = params['inh_num_syns']
        self.inh_num_synchro_syns = params['inh_num_synchro_syns']
        self.inh_syn_weight       = params['inh_syn_weight']
        self.inh_syn_tau1         = params['inh_syn_tau1']
        self.inh_syn_tau2         = params['inh_syn_tau2']

        self.reg = params['reg']
        self.cell = cell

        self.v_rec_list = []
        self.excsyncurrent_rec_list = []
        self.inhsyncurrent_rec_list = []

        self.prng = prng
        self.W = self.prng.random(self.num_states+1)# readout weight

        self.create_synapses()
        self.connect_synapses()
        self.create_records()


    def create_synapses(self):
        if self.syn_mechanisms=='ionotropic':
            self.datagenerator.create_synapses(self.cell)
        elif self.syn_mechanisms=='ionotropic_and_metabotropic':
            self.datagenerator.create_synapses(self.cell, self.ip3, self.cyt)

    def connect_synapses(self):
        self.datagenerator.connect_synapses()


    def resister_inputevent_toNetCon(self):
        self.datagenerator.resister_inputevent_toNetCon()

    def create_records(self):
        self.t_rec = nrn.Vector().record(nrn._ref_t)
        if self.record_target == 'potential':
            total_length = 0
            cumulative_length_dict = []
            for sec in self.cell.all:
                cumulative_length = {'min':total_length, 'max':total_length+sec.L}
                cumulative_length_dict.append(cumulative_length)
                total_length += sec.L

            for rec in range(self.num_states):
                rec_loc = total_length * self.prng.random()

                for index, sec in enumerate(self.cell.all):
                    if cumulative_length_dict[index]['min'] <= rec_loc and rec_loc < cumulative_length_dict[index]['max']:
                        rec_prop = (rec_loc - cumulative_length_dict[index]['min']) / (cumulative_length_dict[index]['max'] - cumulative_length_dict[index]['min'])
                        v = nrn.Vector().record(sec(rec_prop)._ref_v)
                        self.v_rec_list.append(v)
            
            for syn in self.datagenerator.exc_syn_list:
                syncurrent = nrn.Vector().record(syn._ref_i)
                self.excsyncurrent_rec_list.append(syncurrent)

            for syn in self.datagenerator.inh_syn_list:
                syncurrent = nrn.Vector().record(syn._ref_i)
                self.inhsyncurrent_rec_list.append(syncurrent)
        elif self.record_target == 'calcium_rxd':
           node_list = calcium[cyt].nodes
           selected_nodes_array = self.prng.choice(node_list, size=self.num_states, replace=False)
           selected_nodes_list = selected_nodes_array.tolist()
           for node in selected_nodes_list:
                v = nrn.Vector().record(node._ref_concentration)
                self.v_rec_list.append(v)
        elif self.record_target == 'calcium_acum':
            total_length = 0
            cumulative_length_dict = []
            for sec in self.cell.all:
                cumulative_length = {'min':total_length, 'max':total_length+sec.L}
                cumulative_length_dict.append(cumulative_length)
                total_length += sec.L

            while len(self.v_rec_list) < self.num_states:
                rec_loc = total_length * self.prng.random()

                for index, sec in enumerate(self.cell.all):
                    if cumulative_length_dict[index]['min'] <= rec_loc and rec_loc < cumulative_length_dict[index]['max']:
                        rec_prop = (rec_loc - cumulative_length_dict[index]['min']) / (cumulative_length_dict[index]['max'] - cumulative_length_dict[index]['min'])
                        if hasattr(sec(rec_prop), '_ref_cai'):
                            v = nrn.Vector().record(sec(rec_prop)._ref_cai)
                            self.v_rec_list.append(v)
                        else:
                            break



    def bin_average(self, v_rec, t_rec):
        # t_rec[0] is not always 0.0
        v_rec = np.array(v_rec)
        t_rec = np.array(t_rec)
        bin_index = 0
        temp_sum = np.zeros(self.datagenerator.len_transientdata + self.datagenerator.len_trainingdata+self.datagenerator.len_testdata)
        #temp_num = np.zeros(self.datagenerator.len_transientdata + self.datagenerator.len_trainingdata+self.datagenerator.len_testdata)

        for t_index, t in enumerate(t_rec):
            bin_index = math.floor(t/self.bin_width)
            if (self.datagenerator.len_transientdata + self.datagenerator.len_trainingdata+self.datagenerator.len_testdata) <= bin_index:
                break

            if (t_index+1) == len(t_rec):
                temp_sum[bin_index] += v_rec[t_index] * (len(self.datagenerator.get_inputdata()) * self.bin_width - t)
            else:
                temp_sum[bin_index] += v_rec[t_index] * (t_rec[t_index+1]-t)



            #temp_sum[math.floor(t/self.bin_width)] += v_rec[t_index]
            #temp_num[math.floor(t/self.bin_width)] += 1

        #output = np.where(temp_num == 0, -1,  temp_sum / self.bin_width) # insert -1 when they had no t_rec in a bin
        output = temp_sum / self.bin_width

        return np.transpose(output)


    def get_all_state_variables(self):
        # count number of data points in a single bin
        bin_avg_list = np.column_stack([self.bin_average(v_rec.to_python(), self.t_rec.to_python()) for v_rec in self.v_rec_list])
        
        return bin_avg_list

    def get_transient_state_variables(self):
        bin_avg_list = np.column_stack([self.bin_average(v_rec.to_python(), self.t_rec.to_python()) for v_rec in self.v_rec_list])
        
        return bin_avg_list[:self.datagenerator.len_transientdata]
    

    def get_training_state_variables(self):
        # take average within a bin which is only after generated transient period
        bin_avg_list = np.column_stack([self.bin_average(v_rec.to_python(), self.t_rec.to_python()) for v_rec in self.v_rec_list])

        # exclude transient data
        bin_avg_list = bin_avg_list[self.datagenerator.len_transientdata:self.datagenerator.len_transientdata+self.datagenerator.len_trainingdata]
        
        return bin_avg_list

    def get_test_state_variables(self):
        bin_avg_list = np.column_stack([self.bin_average(v_rec.to_python(), self.t_rec.to_python()) for v_rec in self.v_rec_list])
        bin_avg_list = bin_avg_list[self.datagenerator.len_transientdata+self.datagenerator.len_trainingdata:]

        return bin_avg_list

    def readout_transient(self):
        state_vars = np.concatenate( (self.get_transient_state_variables(), np.ones((self.datagenerator.len_transientdata,1))), axis=1 )
        return state_vars @ self.W


    def readout_training(self):
        state_vars = np.concatenate( (self.get_training_state_variables(), np.ones((self.datagenerator.len_trainingdata,1))), axis=1 )
        return state_vars @ self.W

    def readout_test(self):
        state_vars = np.concatenate( (self.get_test_state_variables(), np.ones((self.datagenerator.len_testdata,1))), axis=1 )
        return state_vars @ self.W



    def optimize(self):
        state_vars = np.concatenate( (self.get_training_state_variables(), np.ones((self.datagenerator.len_trainingdata,1))), axis=1 )
        self.W = np.linalg.inv(np.transpose(state_vars) @ state_vars + self.reg*np.eye(self.num_states+1)) @ np.transpose(state_vars) @ self.datagenerator.trainingdata_target

    def generate_response(self):
        nrn.finitialize(-65 * mV)
        # resister input event
        self.resister_inputevent_toNetCon()
        #print('transient period duration', self.datagenerator.len_transientdata*self.bin_width, 'ms')
        #print('run for', len(self.datagenerator.get_inputdata()) * self.bin_width, 'ms')
        nrn.continuerun( len(self.datagenerator.get_inputdata()) * self.bin_width * ms)

    def get_sum_syncurrent(self, typesyn):
        t_rec = self.t_rec.to_python()

        sum_syncurrent = np.zeros(np.size(t_rec))
        if typesyn == 'exc':
            syn_rec_list = []
            for syn_rec in self.excsyncurrent_rec_list:
                syn_rec_list.append(syn_rec.to_python())
            for t_index, t in enumerate(t_rec):
                for syn_rec in syn_rec_list:
                    sum_syncurrent[t_index] += syn_rec[t_index]

        elif typesyn == 'inh':
            syn_rec_list = []
            for syn_rec in self.inhsyncurrent_rec_list:
                syn_rec_list.append(syn_rec.to_python())
            for t_index, t in enumerate(t_rec):
                for syn_rec in syn_rec_list:
                    sum_syncurrent[t_index] += syn_rec[t_index]
        
        
        return sum_syncurrent



    #def closedloop_run(self):


if __name__ == '__main__':
    import NeuronalReservoir_params
    from DataGenerator import sin_datagenerator, MackeyGlass_datagenerator
    from sklearn.metrics import mean_squared_error
    params = NeuronalReservoir_params.params

    prng = np.random.default_rng(1234)
    #prng = np.random.default_rng(123)

    freq = 0.01
    sinwave = sin_datagenerator(params, freq, prng)
    #mgdata = MackeyGlass_datagenerator(params, prng)

    # CA1 pyramidal
    from CA1pyramidal import CA1Pyramidal
    cell = CA1Pyramidal()

    # purkinje
    #from Purkinje import Purkinje
    #cell = Purkinje()

    # CA3 pyramidal
    #from CA3pyramidal import CA3Pyramidal
    #cell = CA3Pyramidal(STDP=False)


    reservoir = neuronalreservoir(cell, sinwave, prng, params)
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
