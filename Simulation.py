import importlib
import sys

# Import modules
from DynamicalSystems import SpringMassDamper, DoubleIntegrator, FirstOrderSystem
from Controller import SpikingController, NetworkBuilder
from Visualiser import Visualiser
import DataModels as dm
import numpy as np
import matplotlib.pyplot as plt

# Reload modules to get latest changes
importlib.reload(sys.modules['DataModels'])
importlib.reload(sys.modules['DynamicalSystems'])
importlib.reload(sys.modules['Controller'])
importlib.reload(sys.modules['Visualiser'])

# Re-import after reload to get updated classes
from DynamicalSystems import SpringMassDamper, DoubleIntegrator, FirstOrderSystem, create_plant
from Controller import SpikingController, NetworkBuilder
from Visualiser import Visualiser
import DataModels as dm
import time
from DataModels import Config, read_data_from_yaml


def run_simulation(config_file: str):
    print(f"Loading configuration from: {config_file}")


    config = read_data_from_yaml(config_file, Config)

    # Output configuration details
    print(f"Running simulation with {config.spiking_network.n_neurons} neurons for {config.simulation.simulation_time} seconds")
    print(f"Total neurons: {config.spiking_network.n_neurons}")
    print(f"Plant type: {config.plant.type}")
    print(f"Connectivity: {config.spiking_network.connectivity}")
    print(f"Simulation time: {config.simulation.simulation_time} seconds")
    print(f"Controller parameters: gamma={config.spiking_network.controller.gamma}, Lp={config.spiking_network.controller.Lp}, Lf={config.spiking_network.controller.Lf}, mu={config.spiking_network.controller.mu}, lambda_ridge={config.spiking_network.controller.lambda_ridge}")
    print(f"Random seeds: Base seed={config.simulation.base_seed} (each controller gets base_seed + index)")
    print(f"Spike force variability: {'Enabled' if config.spiking_network.controller.spike_force.variability else 'Disabled'}")
    if config.spiking_network.controller.spike_force.variability:
        print(f"Spike force distribution: N({config.spiking_network.controller.spike_force.mean:.3f}, {config.spiking_network.controller.spike_force.std:.3f}²)")
    print(f"Measurement noise amplitude: {config.plant.measurement_noise_amplitude:.1e} ({'Disabled' if config.plant.measurement_noise_amplitude == 0 else 'Enabled'})")
    

    plant = create_plant(config)
    controllers, adjacency_matrix, ext_in, ext_out = NetworkBuilder.create_network(config)
    # for c in controllers:
    #     print(c.m_in)
    # controllers[-1].reference_tracking_cost_enable = 'all'
    # controllers[-1].Q, controllers[-1].M = controllers[-1].set_cost_matrices()
    # Time the simulation

    start_time = time.time()
    y, spikes, y_ref, voltages, thresholds, _ = NetworkSimulator.simulate_network(config, plant, controllers, adjacency_matrix, ext_in, ext_out)
    print("Simulation completed in {:.2f} seconds".format(time.time() - start_time))

    Visualiser.plot_all(
        config,
        controllers,
        y,
        y_ref,
        spikes,
        voltages=voltages,
        thresholds=thresholds
    )
    

class NetworkSimulator:

    @staticmethod
    def get_reference(config: dm.Config):
        """
        Generate the reference signal for the simulation.

        Parameters:
        -----------
        config : dm.Config
            Configuration object containing simulation parameters

        Returns:
        --------
        np.ndarray
            Array containing the reference signal
        """
        T = config.simulation.simulation_time
        Ts = config.plant.sampling_time
        N = int(T / Ts)

        reference_signal_type = config.simulation.reference_signal.signal_type

        # Check if step reference signal is enabled
        if reference_signal_type == 'step':

            # Create a reference signal that flips sign at specified intervals
            ref_sign_flips = config.simulation.reference_signal.step.sign_flips
            amplitude = config.simulation.reference_signal.step.amplitude

            y_ref = np.ones((N, 1)) * amplitude  # Start with a constant reference signal
            flip_indices = [N // (ref_sign_flips + 1) * i for i in range(1, ref_sign_flips + 1)]
            for idx in flip_indices:
                y_ref[idx:] *= -1
            return y_ref
        
        # Check if sine reference signal is enabled
        elif reference_signal_type == 'sine':
            # Create a sine wave reference signal
            amplitude = config.simulation.reference_signal.sine.amplitude
            frequency = config.simulation.reference_signal.sine.frequency
            phase = config.simulation.reference_signal.sine.phase
            offset = config.simulation.reference_signal.sine.offset

            t = np.arange(0, T, Ts)
            y_ref = amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset
            return y_ref.reshape(-1, 1)

    @staticmethod
    def simulate_network(config: dm.Config, plant, controllers, adjacency_matrix, ext_in, ext_out):

        spike_force_multipliers = ext_out[0]
        # print(spike_force_multipliers, np.linalg.norm(spike_force_multipliers, ord=1))
        # spike_force_multipliers *= 0.1
        
        # Normalise by number of neurons
        # spike_force_multipliers /= np.linalg.norm(spike_force_multipliers, ord=1)
        # Use regular neurons' input dimension for reference signal
        y_ref = NetworkSimulator.get_reference(config) #+ np.random.normal(0, 0.01, (int(config.simulation.simulation_time / config.plant.sampling_time), 1))
        # Simulation parameters
        T = config.simulation.simulation_time # Total time in seconds
        Ts = config.plant.sampling_time  # Sampling time
        N = int(T / Ts)  # Number of time steps
        print(f"Number of time steps: {N}")
        
        # Create separate RNG for measurement noise to ensure reproducibility
        base_seed = config.simulation.base_seed
        measurement_noise_amplitude = config.plant.measurement_noise_amplitude
        noise_rng = np.random.default_rng(base_seed + 2000) if measurement_noise_amplitude > 0 else None
        
        # Simulation loop
        total_neurons = len(controllers)
        y = np.zeros((N, 1))  # Output (position)
        y[0] = plant.y
        spikes = np.zeros((N, total_neurons))  # To store spikes from all controllers
        voltages = np.zeros((N, total_neurons))  # To store voltages from all controllers
        thresholds = np.zeros((N, total_neurons))  # To store thresholds from all controllers
        Sigma_inv_tr = np.zeros((N, total_neurons))  # To store Sigma inverse trace from all controllers
        u_prev = np.zeros((total_neurons, 1))  # Previous control outputs from all controllers

        for k in range(N):            
            # Get control inputs from all controllers based on connectivity setting
            u_current = np.zeros((total_neurons, 1))
            for i, controller in enumerate(controllers):
                # Get the current controller's input based on its connectivity
                c_in_plant = y[k].reshape(-1, 1) if controller.m_in_plant > 0 else None
                connectivity_in_spikes = adjacency_matrix[:, i] if controller.m_in_spikes > 0 else None
                c_in_spikes = u_prev[connectivity_in_spikes == 1] if controller.m_in_spikes > 0 else None
                if config.spiking_network.controller.combine_spike_channels == 'all' and c_in_spikes is not None:
                    # Sum all spikes into a single channel
                    c_in_spikes = np.sum(c_in_spikes, axis=0).reshape(-1, 1)
                elif config.spiking_network.controller.combine_spike_channels == 'sign' and c_in_spikes is not None:
                    # Sum spikes into two channels based on the sign of the controller's output
                    spike_multipliers_i = np.copy(spike_force_multipliers)  # Exclude current controller's multiplier
                    spike_multipliers_i[i] = 0  # Set current controller's multiplier to zero
                    # 1) Find spikes from neurons with positive output
                    positive_spikes = u_prev[spike_multipliers_i > 0]
                    # 2) Find spikes from neurons with negative output
                    negative_spikes = u_prev[spike_multipliers_i < 0]
                    # 3) Combine them into two channels
                    c_in_spikes = np.vstack((
                        np.sum(positive_spikes, axis=0).reshape(-1, 1),
                        np.sum(negative_spikes, axis=0).reshape(-1, 1)
                    ))
                    # Remove spikes from the current controller

                if c_in_plant is not None and c_in_spikes is not None:
                    # Combine plant input and spikes
                    c_in = np.vstack((c_in_plant, c_in_spikes))
                elif c_in_plant is not None:
                    c_in = c_in_plant
                elif c_in_spikes is not None:
                    c_in = c_in_spikes

                # If the controller has a reference signal,
                # repeat it for each input channel and for the prediction horizon
                # if controller.m_in_plant > 0:
                #     # y_ref_i = np.ones_like(c_in) * y_ref[k]
                #     y_ref_i = np.vstack((np.ones_like(c_in_plant) * y_ref[k], np.zeros_like(c_in_spikes)))
                #     y_ref_i = np.repeat(y_ref_i, controller.Lf, axis=0)
                # else:
                #     y_ref_i = None

                if config.spiking_network.controller.reference_tracking_cost.enable != 'none':
                    # print(y_ref[k], y_ref[k:controller.Lf])
                    # y_ref_i = np.vstack((
                    #     np.ones((controller.m_in_plant, 1)) * y_ref[k], # Repeat 
                    #     np.zeros((controller.m_in_spikes, 1))
                    # ))
                    y_ref_i = np.vstack((
                        np.ones((controller.m_in_plant, 1)),
                        np.zeros((controller.m_in_spikes, 1))
                    ))
                    y_ref_i = np.repeat(y_ref_i, controller.Lf, axis=0) # Repeat the same reference for the prediction horizon. Must be fixed



                    ref_horizon = y_ref[k:k + controller.Lf]
                    if len(ref_horizon) < controller.Lf:
                        # Pad with last value if at the end of the simulation
                        ref_horizon = y_ref[k]*np.ones((controller.Lf, 1))

                    # Create the full reference horizon that matches y_ref_i structure
                    ref_plant = np.tile(ref_horizon, (controller.m_in_plant, 1))  # Repeat for plant inputs
                    ref_spikes = np.zeros((controller.m_in_spikes * controller.Lf, 1))  # Zeros for spike inputs
                    ref_horizon_full = np.vstack((ref_plant, ref_spikes))

                    y_ref_i *= ref_horizon_full
                    
                else:
                    y_ref_i = None
                

                    
                # Step the controller with the current inputs
                u, Voltage, Threshold = controller.step(k, c_in, y_ref_i) 

                # Store the control output and other data
                u_current[i] = u
                voltages[k, i] = Voltage.flatten()
                thresholds[k, i] = Threshold
                Sigma_inv_tr[k, i] = np.trace(np.linalg.inv(controller.Sigma))
            
            # Update previous outputs for next iteration
            u_prev = u_current.copy() 
            # Apply spike force variability and combine control inputs from all neurons
            u_combined = np.sum([spike_force_multipliers[i] * u_current[i] for i in range(total_neurons)])
            if k < N - 1:
                # Update plant state
                y_clean = plant.step(u_combined.reshape(-1, 1))
                # Add measurement noise if enabled
                if measurement_noise_amplitude > 0:
                    noise = measurement_noise_amplitude * noise_rng.normal(0, 1, y_clean.shape)
                    y[k+1] = y_clean + noise
                else:
                    y[k+1] = y_clean

            # Store data from all controllers
            spikes[k] = u_current.flatten()

        return y, spikes, y_ref, voltages, thresholds, Sigma_inv_tr