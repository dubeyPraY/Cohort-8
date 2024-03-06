#This is a custum gym environment for the optimization of Rx(π/2) gate

import gymnasium as gym
import numpy as np
from qiskit_dynamics import Solver
from qiskit.quantum_info import Operator
from qiskit import pulse
from qiskit.quantum_info import state_fidelity, random_statevector, DensityMatrix
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics import Solver
from gym import spaces
from qiskit_dynamics.array import Array
from qiskit_dynamics.pulse import InstructionToSignals
from math import exp

#setting up jax
import jax
jax.config.update("jax_enable_x64", True)
Array.set_default_backend("jax")

X_op = Operator.from_label('X')
    
# Custom Gym environment for Rx(π/2) gate optimization
class RxEnv(gym.Env):
    def __init__(self, backend=None,):
        super().__init__()
        self.backend = backend
        self.simulator = None
        self.action_space = 2
        self.state = 0
        self.start=0
        self.end=7
        self.step_size=0.01
        self.seg=0
        
        
        self.observation_space = spaces.Dict(
            {
                "start": spaces.Box(-1.0,  1.0, shape=(3,), dtype=np.float32),
                "target": spaces.Box(-1.0,  1.0, shape=(3,), dtype=np.float32),
                "current": spaces.Box(-1.0,  1.0, shape=(3,), dtype=np.float32)
            }
        )

        self.action_space = spaces.Box(-1.0,  1.0, shape=(), dtype=np.float32)

        self.start_state = np.array([1, 0])
        self.target_state =np.array([0, 1])
        
    def step(self, action):
        """
        Perform a step in the environment.

        Args:
            action: The action to take in the environment.

        Returns:
            observation: The current observation of the environment.
            reward: The reward obtained from the environment.
            done: A boolean indicating whether the episode is done.
            truncated: A boolean indicating whether the episode was truncated.
            info: Additional information about the step.

        """

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
        
        sol = hamiltonian_solver.solve(t_span=[0., 2*T], y0=y0, signals=sxp)
        self.current_state = sol.y[-1]  
        
        # compute fidelity
        try:
            fid = state_fidelity(self.current_state, self.target_state, validate=False)
            
        except:
            fid=-1
        reward = fid
        observation=self._get_obs()
        done = False

        if fid==1:
            reward = 100
            
        elif fid>0.999:
            reward=10*exp(fid)
            
        self.seg+=1
        truncated=False
        if self.seg>20:
            truncated=True
        
        if done or truncated:
            print('fid= ',fid, "reward= ", reward)
            
        observation=self._get_obs()
        
        return observation, reward, done, truncated, {'fidelity': round(fid, 3)}
        

    def reset(self, seed=None, options=None):
        """
        Resets relevant variables to their initial state

        Args:
            seed (int): Seed value for random number generation (default: None)
            options (dict): Additional options for resetting (default: None)
            
        Returns:
            tuple: Initial state and auxiliary information
            
        """
        
        # Set the start state to a random statevector
        self.start_state = random_statevector(2, seed)
        
        # Evolve the start state using the X_op operator to get the target state
        self.target_state = self.start_state.evolve(X_op)
        
        # Set the current state to the start state
        self.current_state = self.start_state
        
        # Reset the segment counter
        self.seg = 0
        
        # Calculate the fidelity between the current state and the target state
        fid = state_fidelity(self.current_state, self.target_state, validate=True)
        
        # Return the initial state and auxiliary information
        return self._get_obs(), {'fidelity': fid}
    
    def render(self):
        """
            Have to meaningfully define this function to show pulses
        
        Return:
            text drawing of the current circuit represented by the QuantumCircuit object

        """
        return self.show
    
    
    def _get_obs(self):
        """
        Get the observation for the current state.

        Returns:
            dict: A dictionary containing the observations for the start, target, and current states.
        """

        # Create DensityMatrix objects for the start, target, and current states
        density_matrix_start = DensityMatrix(self.start_state)
        density_matrix_target = DensityMatrix(self.target_state)
        density_matrix_current = DensityMatrix(self.current_state)
        
        # Convert the density matrices to Bloch vectors using the toBloch method
        start_observation = self.toBloch(np.array(density_matrix_start))
        target_observation = self.toBloch(np.array(density_matrix_target))
        current_observation = self.toBloch(np.array(density_matrix_current))
        
        # Return a dictionary containing the observations for the start, target, and current states
        return {"start": start_observation, "target": target_observation, "current": current_observation}

    def toBloch(self, matrix):
        """
        Converts a density matrix to a Bloch vector.

        Parameters:
        matrix (list of lists): The density matrix to be converted.

        Returns:
        numpy.ndarray: The Bloch vector representation of the density matrix.
        """

        # Extract the elements of the matrix
        [[a, b], [c, d]] = matrix

        # Calculate the Bloch vector components
        x = complex(c + b).real
        y = complex(c - b).imag
        z = complex(a - d).real

        # Return the Bloch vector as a numpy array
        return np.array([x, y, z])
        
