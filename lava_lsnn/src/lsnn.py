# Import Process level primitives
from lava.proc.lif.process import LIF, AbstractLIF
from lava.proc.lif.models import AbstractPyLifModelFloat
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi
from lava.proc.dense.process import Dense

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.sub.model import AbstractSubProcessModel


from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.ports.reduce_ops import ReduceSum

from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU


from lava.utils.weightutils import SignMode

import os
import numpy as np
import typing as ty

from lava.magma.core.process.process import LogConfig
from numpy import ndarray


class LIF_mod(AbstractLIF):
    """Leaky-Integrate-and-Fire (LIF) neural Process.

    LIF dynamics abstracts to:
    u[t] = u[t-1] * (1-du) + a_in         # neuron current
    v[t] = v[t-1] * (1-dv) + u[t] + bias  # neuron voltage
    s_out = v[t] > vth                    # spike if threshold is exceeded
    v[t] = 0                              # reset at spike

    Parameters
    ----------
    shape : tuple(int)
        Number and topology of LIF neurons.
    u : float, list, numpy.ndarray, optional
        Initial value of the neurons' current.
    v : float, list, numpy.ndarray, optional
        Initial value of the neurons' voltage (membrane potential).
    du : float, optional
        Inverse of decay time-constant for current decay. Currently, only a
        single decay can be set for the entire population of neurons.
    dv : float, optional
        Inverse of decay time-constant for voltage decay. Currently, only a
        single decay can be set for the entire population of neurons.
    bias_mant : float, list, numpy.ndarray, optional
        Mantissa part of neuron bias.
    bias_exp : float, list, numpy.ndarray, optional
        Exponent part of neuron bias, if needed. Mostly for fixed point
        implementations. Ignored for floating point implementations.
    vth : float, optional
        Neuron threshold voltage, exceeding which, the neuron will spike.
        Currently, only a single threshold can be set for the entire
        population of neurons.

    Example
    -------
    >>> lif = LIF(shape=(200, 15), du=10, dv=5)
    This will create 200x15 LIF neurons that all have the same current decay
    of 10 and voltage decay of 5.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        u: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        v: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        du: ty.Optional[float] = 0,
        dv: ty.Optional[float] = 0,
        bias_mant: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        bias_exp: ty.Optional[ty.Union[float, list, np.ndarray]] = 0,
        vth: ty.Optional[float] = 10,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
        **kwargs,
    ) -> None:
        
        super().__init__(
            shape=shape,
            u=u,
            v=v,
            du=du,
            dv=dv,
            bias_mant=bias_mant,
            bias_exp=bias_exp,
            name=name,
            log_config=log_config,
            **kwargs,
        )

        self.vth = Var(shape=(1,), init=vth)
        self.s_in = InPort(shape=shape,reduce_op=ReduceSum)
        s_out = OutPort(shape=shape)
        self.current_out = OutPort(shape=shape)

    def print_vars(self):
        """Prints all variables of a LIF process and their values."""

        sp = 3 * "  "
        print("Variables of the LIF:")
        print(sp + "u:    {}".format(str(self.u.get())))
        print(sp + "v:    {}".format(str(self.v.get())))
        print(sp + "du:   {}".format(str(self.du.get())))
        print(sp + "dv:   {}".format(str(self.dv.get())))
        print(sp + "vth:  {}".format(str(self.vth.get())))   
            
@implements(proc=LIF_mod, protocol=LoihiProtocol)
@requires(CPU)
@tag("float_pt")
class PyModLifModel(AbstractPyLifModelFloat):

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    current_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vth: float = LavaPyType(float, float)
        
    def spiking_activation(self):
        """Spiking activation function for LIF."""
        return self.v > self.vth
    
    def run_spk(self):
        super().run_spk()
        self.current_out.send(self.u)

class LsnnNet(AbstractProcess):
    def __init__(self, 
                **params) -> None:
        
        super().__init__()
        
        dense_input_weights = params.get("input_weights", np.ones((1,1)))
        dense_recurrent_weights = params.get('recurrent_weights', np.ones((1,1)))
        dense_output_weights = params.get('output_weights', np.ones((1,1)))

        self.dense_input_weights = Var(shape=dense_input_weights.shape, init=dense_input_weights)
        self.dense_output_weights = Var(shape=dense_output_weights.shape, init=dense_output_weights)
        self.dense_recurrent_weights = Var(shape=dense_recurrent_weights.shape, init=dense_recurrent_weights)

        self.s_in  = InPort(shape = (dense_input_weights.shape[1],), reduce_op = ReduceSum)
        self.a_out  = OutPort(shape = (dense_output_weights.shape[0],))
        
        
        #parameter for forward lif neuron group
        f_u = params.get("f_u",0)
        f_v = params.get("f_v",0)
        f_dv = params.get("f_dv",0)
        f_du = params.get("f_du",0)
        f_bias_exp = params.get("f_bias_exp",0)
        f_bias_mant = params.get("f_bias_mant",1)
        f_vth = params.get("f_vth",1)

        self.input_shape = Var(shape=dense_input_weights.shape)
        self.output_shape = Var(shape=dense_output_weights.shape)

        self.f_u = Var(shape=(dense_input_weights.shape[0],), init=f_u)
        self.f_v = Var(shape=(dense_input_weights.shape[0],), init=f_v)
        self.f_dv = Var(shape=(1,), init=f_dv)
        self.f_du = Var(shape=(1,), init=f_du)
        self.f_vth = Var(shape=(1,), init=f_vth)
        self.f_bias_exp = Var(shape=(dense_input_weights.shape[0],), init=f_bias_exp) 
        self.f_bias_mant = Var(shape=(dense_input_weights.shape[0],), init=f_bias_mant)
        
        #parameter for backword lif neuron group
        b_u = params.get("b_u",0)
        b_v = params.get("b_v",0)
        b_dv = params.get("b_dv",1)
        b_du = params.get("b_du",0)
        b_bias_exp = params.get("b_bias_exp",0)
        b_bias_mant = params.get("b_bias_mant",1)
        b_vth = params.get("b_vth",10000)

        self.b_u = Var(shape=(dense_input_weights.shape[0],), init=b_u)
        self.b_v = Var(shape=(dense_input_weights.shape[0],), init=b_v)
        self.b_dv = Var(shape=(1,), init=b_dv)
        self.b_du = Var(shape=(1,), init=b_du)
        self.b_vth = Var(shape=(1,), init=b_vth)
        self.b_bias_exp = Var(shape=(dense_input_weights.shape[0],), init=b_bias_exp) 
        self.b_bias_mant = Var(shape=(dense_input_weights.shape[0],), init=b_bias_mant)

        self.log_config = params.get("log_config",0)                                       
        

@implements(proc=LsnnNet, protocol=LoihiProtocol)
class LsnnNetModel(AbstractSubProcessModel):

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc):
        """Builds sub Process structure of the Process."""
        
        # Instantiate child processes
        # The input shape is a 2D vector (shape of the weight matrix).
        input_shape = proc.proc_params.get("input_shape", (1,1))
        output_shape = proc.proc_params.get("output_shape", (1,1))

        self.in_dense = Dense(weights=proc.proc_params.get("dense_input_weights", np.ones((1,1))))

        self.f_lif = LIF_mod(shape=(input_shape[1],),
                             u = proc.proc_params.get("f_u", (1,)),
                             v = proc.proc_params.get("f_v", (1,)),
                             dv = proc.proc_params.get("f_dv", 0),
                             du = proc.proc_params.get("f_du", 0),
                             bias_exp= proc.proc_params.get("f_bias_exp", 0),
                             bias_mant=proc.proc_params.get("f_bias_mant", 1),
                             vth=proc.proc_params.get("f_vth", 1),
                             log_config= proc.proc_params.get("log_config", 0)
                             )
        
        self.r_dense = Dense(weights=proc.proc_params.get("dense_recurrent_weights", np.ones((1,1))),
                             sign_mode = SignMode.INHIBITORY)
        ### Add minus sign at dense output

        self.b_lif = LIF_mod(shape=(output_shape[0],),
                            u = proc.proc_params.get("b_u", (0,)),
                            v = proc.proc_params.get("b_v", (0,)),
                            dv = proc.proc_params.get("b_dv", 0),
                            du = proc.proc_params.get("b_du", 0),
                            bias_exp= proc.proc_params.get("b_bias_exp", 0),
                            bias_mant=proc.proc_params.get("b_bias_mant", 1),
                            vth=proc.proc_params.get("b_vth", 10000),
                            log_config= proc.proc_params.get("log_config", 0)
                        )
        
        self.out_dense = Dense(weights=proc.proc_params.get("dense_output_weights", np.ones((1,1))))



        proc.in_ports.s_in.connect(self.in_dense.in_ports.s_in)

        self.out_dense.out_ports.a_out.connect(proc.out_ports.a_out)


        self.f_lif.in_ports.s_in.connect_from([self.in_dense.out_ports.a_out,self.r_dense.out_ports.a_out])

        self.f_lif.out_ports.s_out.connect(self.out_dense.in_ports.s_in)

        self.f_lif.out_ports.current_out.connect(self.b_lif.in_ports.s_in)
        
        self.b_lif.out_ports.current_out.connect(self.r_dense.in_ports.s_in)
        

        # proc.vars.in_weights.alias(self.in_dense.vars.weights)

        proc.vars.f_u.alias(self.f_lif.vars.u)
        proc.vars.f_v.alias(self.f_lif.vars.v)
        proc.vars.f_bias_mant.alias(self.f_lif.vars.bias_mant)
        proc.vars.f_du.alias(self.f_lif.vars.du)
        proc.vars.f_dv.alias(self.f_lif.vars.dv)
        proc.vars.f_vth.alias(self.f_lif.vars.vth)

        proc.vars.dense_recurrent_weights.alias(self.r_dense.vars.weights)

        proc.vars.b_u.alias(self.b_lif.vars.u)
        proc.vars.b_v.alias(self.b_lif.vars.v)
        proc.vars.b_bias_mant.alias(self.b_lif.vars.bias_mant)
        proc.vars.b_du.alias(self.b_lif.vars.du)
        proc.vars.b_dv.alias(self.b_lif.vars.dv)
        proc.vars.b_vth.alias(self.b_lif.vars.vth)

        proc.vars.dense_output_weights.alias(self.out_dense.vars.weights)