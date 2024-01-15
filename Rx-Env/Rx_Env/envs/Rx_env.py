import gymnasium as gym
import numpy as np
from qiskit_dynamics import Solver, DynamicsBackend
#from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import Operator
from qiskit import pulse
from qiskit.quantum_info import state_fidelity, random_statevector,Statevector, DensityMatrix
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics import Solver


#from qutip import fidelity
# Configure to use JAX internally
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
from qiskit_dynamics.array import Array
from gym.spaces import Box
from gym import spaces
Array.set_default_backend("jax")
from matplotlib import pyplot as plt
from qiskit_dynamics.pulse import InstructionToSignals


X_op = Operator.from_label('X')
    

# Custom Gym environment for Rx(π/2) gate optimization
class RxEnv(gym.Env):
    def __init__(self, backend=None,):
        super().__init__()
        self.backend = backend
        self.simulator = None
        self.action_space = 2
        # self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.state = 0
        self.start=0
        self.end=7
        self.step_size=0.01
        
        
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(-1.0,  1.0, shape=(3,), dtype=float),
                "target": spaces.Box(-1.0,  1.0, shape=(3,), dtype=float),
            }
        )

        self.action_space = spaces.Box(0.0,  2.0, shape=(), dtype=np.float32)

        # self.action_space= spaces.Dict(
        #     {
        #         "real_amplitude": Box(low=0.0, high=6.0, shape=(), dtype=np.float32),
                
        #         "img_amplitude": Box(low=0.0, high=6.0, shape=(), dtype=np.float32),
            
        #         "phase": Box(low=0.0, high=6.0, shape=(), dtype=np.float32),
            
        #     }
        # )


        self.start_state = np.array([1, 0])
        self.target_state =np.array([0, 1])
        
    def step(self, action):
        # Strength of the Rabi-rate in GHz.
        r = 0.1

        # Frequency of the qubit transition in GHz.
        w = 5.

        # Sample rate of the backend in ns.
        dt = 1 / 4.5

        # Define gaussian envelope function to approximately implement an sx gate.
        amp = 1. / 1.75
        sig = 0.6985/r/amp
        T = 4*sig
        duration = int(T / dt)
        beta = 2.0

        with pulse.build(name="sx-sy schedule") as sxp:
            pulse.play(pulse.Drag(duration, action, sig / dt, beta), pulse.DriveChannel(0))

        self.show=sxp.draw()

        #The code below generates a list of schedules specifying experiments on qubit 0    
        converter = InstructionToSignals(dt, carriers={"d0": w})

        signals = converter.get_signals(sxp)


        # construct operators
        X = Operator.from_label('X')
        Z = Operator.from_label('Z')

        drift = 2 * np.pi * w * Z/2
        operators = [2 * np.pi * r * X/2]

        # construct the solver
        hamiltonian_solver = Solver(
            static_hamiltonian=drift,
            hamiltonian_operators=operators,
            rotating_frame=drift,
            rwa_cutoff_freq=2 * 5.0,
            hamiltonian_channels=['d0'],
            channel_carrier_freqs={'d0': w},
            dt=dt
        )
        
        # Start the qubit in its ground state.
        y0 = self.start_state

        sol = hamiltonian_solver.solve(t_span=[0., 2*T], y0=y0, signals=sxp, atol=1e-8, rtol=1e-8)
        self.current_state = sol.y[-1]  
        # compute fidelity
        fid = state_fidelity(self.current_state, self.target_state, validate=True)
        
        observation=self._get_obs()

        if fid==1:
            reward = 1
            done = True

        return observation, reward, done, {'fidelity': round(fid, 3)}
        

    def reset(self, seed=None, options=None):
        """
        Resets relevant variables to their initial state

        Return:
            touple: initial state and auxilarry information

        """

        self.start_state = random_statevector(2)
        self.target_state = self.start_state.evolve(X_op)
        self.current_state = self.start_state

        fid = state_fidelity(self.current_state, self.target_state, validate=True)
        return self._get_obs(), {'fidelity':fid}#returning initial state and auxilarry information
    
    def render(self):
        """
        Return:
            text drawing of the current circuit represented by the QuantumCircuit object

        """
        return self.show
    
    
    def _get_obs(self):#returning the observation
        density_matrix_start = DensityMatrix(self.current_state)
        density_matrix_target = DensityMatrix(self.target_state)
        
        return {"agent": self.toBloch(np.array(density_matrix_start)), "target": self.toBloch(np.array(density_matrix_start))}

    def toBloch(self,matrix):#converting density matrix to bloch vector
        [[a, b], [c, d]] = matrix
        x = complex(c + b).real
        y = complex(c - b).imag
        z = complex(a - d).real
        return np.array([x, y, z])
    
    
    # def sample(self):
    #     """
    #     Return:
    #         action (dictionary): a specific action in the environment action space

    #     """
    #     action = {}
    #     action['amp']=np.random.choice(self.create_amplitudes())+np.random.choice(self.create_amplitudes())*1j
    #     action['phase']=np.random.choice(self.create_phase())
    #     return action


    
