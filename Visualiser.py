import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import matplotlib.animation as animation
import networkx as nx

import DataModels as dm 
from Controller import SpikingController, NetworkBuilder


class Visualiser:

    @staticmethod
    def plot_all(
        config: dm.Config,
        controllers: list,
        y: np.ndarray,
        y_ref: np.ndarray,
        spikes: np.ndarray,
        voltages: np.ndarray = None,
        thresholds: np.ndarray = None
    ):
        """
        Plot all simulation results
        """
        highlight_range = config.plotting.highlight_range if config.plotting.highlight_range else None
        show_legends = config.plotting.show_legends
        # Plot the system response
        if config.plotting.plot_response:
            Visualiser.plot_response(
                config, y, y_ref, spikes, highlight_range, show_legends
            )


        # Animate the system response if enabled
        if config.plotting.save_animations:
            response_filename = "response.gif"
            network_filename = "network.gif"
        else:
            response_filename = None
            network_filename = None

        if config.plotting.animate_response:
            Visualiser.animate_response(
                config, y, y_ref, spikes, show_legends=show_legends,
                save_as=response_filename, fps=30
            )
        
        # Animate the network if enabled
        if config.plotting.animate_network:
            Visualiser.animate_network(
                config, spikes, ext_signals=y,
                layout_func=nx.spring_layout,
                fade_tau=3, dt=config.plant.sampling_time,
                figsize=(6, 6),
                on_color="tab:blue", off_color="lightgray",
                ext_on_color="tab:red", ext_off_color="lightcoral",
                save_as=network_filename, fps=30
            )

        if config.plotting.plot_variance:
            Visualiser.plot_epistemic_variance(config, controllers)

        # Compare predictions for all controllers
        if config.plotting.compare_predictions:
            Visualiser.compare_predictions(config, controllers, y, spikes)

        # Plot the connectivity matrix
        if config.plotting.plot_matrices.plot_connectivity_matrices:
            Visualiser.plot_connectivity_matrices(config)

        # Plot the covariance matrices
        if config.plotting.plot_matrices.plot_covariance_matrices:
            Visualiser.plot_covariance_matrices(config, controllers)

        # Plot the gain matrices
        if config.plotting.plot_matrices.plot_gain_matrices:
            Visualiser.plot_gain_matrices(config, controllers)

        # Plot voltage and threshold for each controller
        if config.plotting.plot_voltage_threshold and voltages is not None and thresholds is not None:
            Visualiser.plot_voltage_threshold(config, voltages, thresholds)

        

    @staticmethod
    def plot_response(
        config: dm.Config,
        y: np.ndarray,
        y_ref: np.ndarray,
        spikes: np.ndarray,
        highlight_range: tuple = None,
        show_legends: bool = True
    ):
        
        plant_type = config.plant.type
        n_neurons = config.spiking_network.n_neurons
        # Plotting the results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # Position subplot
        N = len(y)
        Ts = config.plant.sampling_time
        ax1.plot(np.arange(N) * Ts, y_ref, label='Reference Signal', linestyle='--', color='red')
        ax1.plot(np.arange(N) * Ts, y, label='Position', color='black', linewidth=2)
        plant_name = plant_type.replace('_', ' ').title()
        ax1.set_title(f'{plant_name} System Simulation ({n_neurons} Controllers)')
        ax1.set_ylabel('Position' if plant_type != 'first_order' else 'Output')
        if show_legends:
            ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add highlight if specified
        if highlight_range is not None:
            start_time, end_time = highlight_range
            ax1.axvspan(start_time, end_time, alpha=0.3, color='grey', label='Highlighted Range')
            ax2.axvspan(start_time, end_time, alpha=0.3, color='grey', label='Highlighted Range')

        # Spike raster plot subplot for all controllers
        time_axis = np.arange(N) * Ts
        colors = plt.cm.tab10(np.linspace(0, 1, n_neurons))
        
        spike_detected = False
        for i in range(n_neurons):
            spike_times = time_axis[spikes[:, i] > 0]
            if len(spike_times) > 0:
                ax2.eventplot([spike_times], lineoffsets=0.1 + i*0.8/n_neurons, 
                            linelengths=0.8/n_neurons, colors=[colors[i]], 
                            linewidths=2)
                spike_detected = True
        
        if spike_detected:
            ax2.set_ylim(0, 1)
            # ax2.set_ylabel('Spikes')
        else:
            ax2.text(0.5, 0.5, 'No spikes detected', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_ylim(0, 1)
            # ax2.set_ylabel('Spikes')

        ax2.set_title('Spike Raster Plot')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Time (s)')
        ax2.set_yticks([])  # Remove y-ticks from spike raster plot

        plt.tight_layout()
        plt.show()

        # Create zoomed plot if highlight range is specified
        if highlight_range is not None:
            start_time, end_time = highlight_range
            
            # Find indices corresponding to the highlight range
            start_idx = int(start_time / Ts)
            end_idx = int(end_time / Ts)
            start_idx = max(0, start_idx)
            end_idx = min(N, end_idx)
            
            if start_idx < end_idx:
                # Create zoomed plot
                fig_zoom, (ax1_zoom, ax2_zoom) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                
                # Zoomed time axis
                time_zoom = np.arange(start_idx, end_idx) * Ts
                
                # Position subplot (zoomed)
                ax1_zoom.plot(time_zoom, y_ref[start_idx:end_idx], 
                            label='Reference Signal', linestyle='--', color='red')
                ax1_zoom.plot(time_zoom, y[start_idx:end_idx], 
                            label='Position', color='black', linewidth=2)
                ax1_zoom.set_title(f'{plant_name} System Simulation')
                ax1_zoom.set_ylabel('Position' if plant_type != 'first_order' else 'Output')
                if show_legends:
                    ax1_zoom.legend()
                ax1_zoom.grid(True, alpha=0.3)
                
                # Spike raster plot (zoomed)
                spike_detected_zoom = False
                for i in range(n_neurons):
                    # Get spike times within the zoom range
                    spike_mask = (spikes[start_idx:end_idx, i] > 0)
                    spike_times_zoom = time_zoom[spike_mask]
                    
                    if len(spike_times_zoom) > 0:
                        ax2_zoom.eventplot([spike_times_zoom], lineoffsets=0.1 + i*0.8/n_neurons, 
                                        linelengths=0.8/n_neurons, colors=[colors[i]], 
                                        linewidths=2, label=f'Controller {i+1}')
                        spike_detected_zoom = True
                
                if spike_detected_zoom:
                    ax2_zoom.set_ylim(0, 1)
                    # ax2_zoom.set_ylabel('Spikes')
                else:
                    ax2_zoom.text(0.5, 0.5, 'No spikes detected in range', 
                                ha='center', va='center', transform=ax2_zoom.transAxes)
                    ax2_zoom.set_ylim(0, 1)
                    # ax2_zoom.set_ylabel('Spikes')
                
                ax2_zoom.set_title('Spike Raster Plot')
                ax2_zoom.grid(True, alpha=0.3)
                ax2_zoom.set_xlabel('Time (s)')
                ax2_zoom.set_yticks([])  # Remove y-ticks from zoomed spike raster plot
                
                plt.tight_layout()
                plt.show()


    @staticmethod
    def plot_connectivity_matrices(
        config: dm.Config
    ):
        # Create a figure with three subplots where each is a heatmap of the connectivity matrices
        adjacency, ext_in, ext_out = NetworkBuilder.create_connectivity(config.spiking_network)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['Adjacency Matrix', 'External Input Connectivity', 'External Output Connectivity']
        matrices = [adjacency, ext_in, ext_out]
        for ax, title, matrix in zip(axes, titles, matrices):
            im = ax.imshow(matrix, cmap='viridis', aspect='auto')
            ax.set_title(title)
            ax.set_xlabel('Neurons')
            ax.set_ylabel('Neurons' if title != 'External Input Connectivity' else 'Plant Output')
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.grid(False)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def compare_predictions(
        config: dm.Config,
        controllers: list,
        y: np.ndarray,
        spikes: np.ndarray
    ):
        # Comparing plant predictions for all controllers
        Ts = config.plant.sampling_time
        predictions = [np.array(controller.predictions) for controller in controllers]
        # true_outputs = np.array([[pred[0] for pred in controller.predictions] for controller in controllers])
        n_neurons = len(predictions)
        n_neurons = config.spiking_network.n_neurons
        connectivity = config.spiking_network.connectivity
        Lf = config.spiking_network.controller.Lf
        show_legends = config.plotting.show_legends


        colors = plt.cm.tab10(np.linspace(0, 1, n_neurons))
        n_cols = min(n_neurons, 3)  # Maximum 3 columns
        n_rows = (n_neurons + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), 
                                sharex=True, sharey=True)
        
        # Ensure axes is always a 2D array for consistent indexing
        if n_neurons == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n_neurons):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            preds = predictions[i]
            # ax.plot(np.arange(len(preds)-1) * Ts, y[1:], 
            #         label='True Output', color='black', linestyle='--', linewidth=2)
            ax.plot(np.arange(len(preds)-1) * Ts, y[1:],
                    label='True Output', color='black', linestyle='--', linewidth=2)
            ax.plot(np.arange(len(preds)-1) * Ts, preds[:-1, 0, 0], # Lf-1 is correct
                    label=f'Controller {i+1} Predictions', color=colors[i])
            # ax.plot(np.arange(len(preds)) * Ts, preds[:-1, Lf-1, 0], 
            #         label=f'Controller {i+1} Predictions2', color='red')
            
            # Add controller label
            ax.set_title(f'Controller {i+1} - One-Step-Ahead Predictions')
            ax.set_ylabel('Predicted Output')
            if show_legends:
                ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_neurons, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

        # Comparing spike predictions for all controllers (only for connected case)
        if connectivity != 'none' and n_neurons > 1:

            if config.spiking_network.controller.combine_spike_channels == None:
                # Create a figure where each controller shows its predictions of all other controllers
                fig, axes = plt.subplots(n_neurons, n_neurons-1, figsize=(5*(n_neurons-1), 4*n_neurons), 
                                        sharex=True, sharey=True)
                
                # Handle the case where we have only 2 neurons (1D array of axes)
                if n_neurons == 2:
                    axes = axes.reshape(n_neurons, -1)

                # If no spike channel combination, we plot each controller's prediction of each other controller's spikes
                for i in range(n_neurons):
                    other_controller_idx = 0
                    for j in range(n_neurons):
                        if i != j:  # Skip self
                            ax = axes[i, other_controller_idx]
                            
                            # Get the actual spikes from controller j
                            actual_spikes = spikes[:, j]
                            preds = predictions[i]
                            
                            
                            # The spike prediction for controller j is at position Lf + other_controller_idx*Lf
                            # This corresponds to the position in the input vector for controller j's previous output
                            spike_pred_idx = controllers[i].m_in_plant*Lf + other_controller_idx*Lf
                            
                            ax.plot(np.arange(len(actual_spikes)) * Ts, actual_spikes, 
                                    label=f'Controller {j+1} Actual Spikes', color='black', linestyle='-', linewidth=2)
                            
                            if spike_pred_idx < preds.shape[1]:
                                pred_length = min(len(preds), len(actual_spikes))
                                ax.plot(np.arange(pred_length) * Ts, preds[:pred_length, spike_pred_idx+Lf-1, 0], 
                                        label=f'Controller {i+1} Predictions', color='#1E90FF') #colors[i])
                            
                            #ax.set_title(f'Controller {i+1} Predicting Controller {j+1} Spikes')
                            #ax.set_ylabel('Spikes')
                            if show_legends:
                                ax.legend()
                            ax.grid(True, alpha=0.3)
                            
                            other_controller_idx += 1
                
                plt.tight_layout()
                plt.show()
            elif config.spiking_network.controller.combine_spike_channels == 'sign':
                # If spike channels are combined by sign, we plot the predictions of positive and negative spikes
                adjacency, ext_in, ext_out = NetworkBuilder.create_connectivity(config.spiking_network)
                spike_force_multipliers = ext_out[0]
                fig, axes = plt.subplots(n_neurons, 2, figsize=(10, 4*n_neurons), 
                                        sharex=True, sharey=True)
                for i in range(n_neurons):
                    # Get the actual spikes that controller i observes
                    multipliers_i = np.copy(spike_force_multipliers)  # Exclude current controller's multiplier
                    multipliers_i[i] = 0  # Set current controller's multiplier to zero

                    spikes_in_pos = spikes[:, multipliers_i > 0]
                    spikes_in_neg = spikes[:, multipliers_i < 0]
                    
                    true_spikes_pos = np.sum(spikes_in_pos, axis=1)
                    true_spikes_neg = np.sum(spikes_in_neg, axis=1)

                    preds = predictions[i]

                    # # The spike prediction for controller i is at position Lf + i*Lf
                    spike_pred_idx_pos = controllers[i].m_in_plant*Lf
                    spike_pred_idx_neg = controllers[i].m_in_plant*Lf + Lf

                    ax_pos = axes[i, 0]
                    ax_neg = axes[i, 1]
                    ax_pos.plot(np.arange(len(true_spikes_pos)) * Ts, true_spikes_pos, 
                                label=f'Controller {i+1} Actual Positive Spikes', color='black', linestyle='-', linewidth=2)
                    ax_neg.plot(np.arange(len(true_spikes_neg)) * Ts, true_spikes_neg,
                                label=f'Controller {i+1} Actual Negative Spikes', color='black', linestyle='-', linewidth=2)
                    if spike_pred_idx_pos < preds.shape[1]:
                        # Use only the first Lf predictions to match the actual spikes length
                        pred_length = min(len(preds), len(true_spikes_pos))
                        ax_pos.plot(np.arange(pred_length) * Ts, preds[:pred_length, spike_pred_idx_pos+Lf-1, 0], 
                                    label=f'Controller {i+1} Positive Predictions', color=colors[i])
                        ax_neg.plot(np.arange(pred_length) * Ts, preds[:pred_length, spike_pred_idx_neg+Lf-1, 0], 
                                    label=f'Controller {i+1} Negative Predictions', color=colors[i])
                    ax_pos.set_title(f'Controller {i+1} Predicting Positive Spikes')
                    ax_neg.set_title(f'Controller {i+1} Predicting Negative Spikes')
                    ax_pos.set_ylabel('Spikes')
                    ax_neg.set_ylabel('Spikes')
                    if show_legends:
                        ax_pos.legend()
                        ax_neg.legend()
                    ax_pos.grid(True, alpha=0.3)
                    ax_neg.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

            elif config.spiking_network.controller.combine_spike_channels == 'all':
                # If spike channels are combined into one, we plot the predictions of combined spikes
                n_cols = min(n_neurons, 2)  # Maximum 2 columns
                n_rows = (n_neurons + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 4*n_rows), 
                                        sharex=True, sharey=True)
                
                # Ensure axes is always a 2D array for consistent indexing
                if n_neurons == 1:
                    axes = np.array([[axes]])
                elif n_rows == 1:
                    axes = axes.reshape(1, -1)
                elif n_cols == 1:
                    axes = axes.reshape(-1, 1)
                
                for i in range(n_neurons):
                    row = i // n_cols
                    col = i % n_cols
                    ax = axes[row, col]
                    
                    # Get the actual spikes that controller i observes
                    actual_spikes = np.sum(spikes, axis=1)
                    preds = predictions[i]

                    # The spike prediction for controller i is at position Lf
                    spike_pred_idx = controllers[i].m_in_plant*Lf
                    
                    ax.plot(np.arange(len(actual_spikes)) * Ts, actual_spikes, 
                            label=f'Controller {i+1} Actual Spikes', color='black', linestyle='-', linewidth=2)
                    if spike_pred_idx < preds.shape[1]:
                        pred_length = min(len(preds), len(actual_spikes))
                        ax.plot(np.arange(pred_length) * Ts, preds[:pred_length, spike_pred_idx+Lf-1, 0], 
                                label=f'Controller {i+1} Predictions', color=colors[i])
                    if show_legends:
                        ax.set_title(f'Controller {i+1} Predicting Combined Spikes')
                        ax.set_ylabel('Spikes')
                        ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # Hide unused subplots
                for i in range(n_neurons, n_rows * n_cols):
                    row = i // n_cols
                    col = i % n_cols
                    ax = axes[row, col]
                    ax.axis('off')
                    
                plt.tight_layout()
                plt.show()

        else:
            print("Spike predictions not plotted (no connections or single neuron)")

    @staticmethod
    def plot_voltage_threshold(
        config: dm.Config,
        voltages: np.ndarray, 
        thresholds: np.ndarray):
        

        n_neurons = config.spiking_network.n_neurons
        Ts = config.plant.sampling_time
        n_cols = min(n_neurons, 3)  # Maximum 3 columns
        n_rows = (n_neurons + n_cols - 1) // n_cols
        colors = plt.cm.tab10(np.linspace(0, 1, n_neurons))
        N = voltages.shape[0]
        show_legends = config.plotting.show_legends

        # Plot voltages and thresholds for all controllers
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), 
                                sharex=True, sharey=True)
        
        # Ensure axes is always a 2D array for consistent indexing
        if n_neurons == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i in range(n_neurons):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Add controller label
            ax.plot(np.arange(N) * Ts, voltages[:, i], label=f'C{i+1} Voltage', color=colors[i])
            ax.plot(np.arange(N) * Ts, thresholds[:, i], label=f'C{i+1} Threshold', 
                    linestyle='--', color=colors[i])
            ax.set_title(f'Controller {i+1} - Voltage and Threshold')
            ax.set_ylabel('Voltage / Threshold')
            if show_legends:
                ax.legend()
            ax.grid()
        
        # Hide unused subplots
        for i in range(n_neurons, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_covariance_matrices(
        config: dm.Config,
        controllers: list,
    ):
        indices = config.plotting.plot_matrices.indices
        
        # Filter controllers based on indices parameter
        if indices is None:
            # Show all controllers
            selected_controllers = controllers
            selected_indices = list(range(len(controllers)))
        else:
            # Show only specified indices
            selected_controllers = [controllers[i] for i in indices if i < len(controllers)]
            selected_indices = [i for i in indices if i < len(controllers)]
        
        if not selected_controllers:
            print("No valid controller indices specified for covariance matrix plotting")
            return
            
        # Σ / Ψ heat-maps for selected controllers
        n_neurons = len(selected_controllers)

        fig, axes = plt.subplots(n_neurons, 3, figsize=(15, 4*n_neurons))
        if n_neurons == 1:
            axes = axes.reshape(1, -1)

        for plot_idx, (controller_idx, controller) in enumerate(zip(selected_indices, selected_controllers)):
            # Sigma heatmap
            im1 = axes[plot_idx,0].imshow(controller.Sigma, cmap='viridis', aspect='auto')
            axes[plot_idx,0].set_title(f'Controller {controller_idx+1} - $\\Sigma$')
            fig.colorbar(im1, ax=axes[plot_idx,0], shrink=0.8)
            
            # Psi heatmap
            im2 = axes[plot_idx,1].imshow(controller.Psi, cmap='viridis', aspect='auto')
            axes[plot_idx,1].set_title(f'Controller {controller_idx+1} - $\\Psi$')
            fig.colorbar(im2, ax=axes[plot_idx,1], shrink=0.8)
            
            # Sigma inverse heatmap
            im3 = axes[plot_idx,2].imshow(np.linalg.inv(controller.Sigma), cmap='viridis', aspect='auto')#, vmin=-3, vmax=3)
            axes[plot_idx,2].set_title(f'Controller {controller_idx+1} - $\\Sigma^{{-1}}$')
            fig.colorbar(im3, ax=axes[plot_idx,2], shrink=0.8)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_gain_matrices(
        config: dm.Config,
        controllers: list,
    ):
        n_neurons = config.spiking_network.n_neurons
        n_cols = min(n_neurons, 3)  # Maximum 3 columns
        n_rows = (n_neurons + n_cols - 1) // n_cols

        controller_gains = [controller.get_gain_matrices()[-1] for controller in controllers]

        # Controller gain matrix heatmaps for all controllers
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Ensure axes is always a 2D array for consistent indexing
        if n_neurons == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i in range(n_neurons):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Plot gain matrix heatmap
            K = controller_gains[i]
            im = ax.imshow(K, cmap='viridis', aspect='auto') #, vmin=-np.abs(K).max(), vmax=np.abs(K).max()
            
            # Add controller label
            ax.set_title(f'Controller {i+1} - Gain Matrix K')
            ax.set_xlabel('Input Dimension')
            ax.set_ylabel('Output Dimension')
            fig.colorbar(im, ax=ax, shrink=0.8)
        
        # Hide unused subplots
        for i in range(n_neurons, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def animate_network(config, spikes, ext_signals=None,
                             layout_func=nx.spring_layout,
                             fade_tau=1, dt=1.0, figsize=(6, 6),
                             on_color="tab:blue", off_color="lightgray",
                             ext_on_color="tab:red", ext_off_color="lightcoral",
                             save_as=None, fps=30):
        """
        Show (or save) an animated raster of a spiking network in its structural context.
        -------------------------------------------------------------------------------
        Parameters
        ----------
        adjacency : (N,N) array
            Adjacency / weight matrix (non-zero → directed edge i→j).
        spikes : (T,N) array
            Binary (or 0/float) matrix – 1 when neuron j fires at step t.
        layout_func : callable, default spring layout
            Any NetworkX layout taking G -> {node: (x,y)}.
        fade_tau : float
            Exponential decay time-constant (in simulation steps) for the colour fade.
        dt : float
            Simulation time step (for an on-figure clock). 1.0 = “step”, 0.001 = “ms”, …
        figsize : tuple
            Figure size passed to `plt.subplots`.
        on_color / off_color : str or rgba
            Bright colour on a spike / base colour when fully faded.
        ext_conn : (M,N) array, optional
            External connectivity matrix (external signal m → neuron n).
        ext_signals : (T,M) array, optional
            External signal activity over time.
        ext_on_color / ext_off_color : str or rgba
            Colors for external signal nodes when active/inactive.
        save_as : str | None
            If not None, path ('.mp4', '.gif', '.avi' …) where the animation is stored.
        fps : int
            Frames per second for saving.
        """
        # --------------------------------------------------------------------- set-up
        T, N = spikes.shape
        
        adjacency, ext_in, ext_out = NetworkBuilder.create_connectivity(network_params=config.spiking_network)
        plot_external = False

        # Handle external signals
        if (ext_in is not None and ext_signals is not None and plot_external) and (np.sum(ext_in) > 0):
            ext_in = np.asarray(ext_in)
            ext_signals = np.asarray(ext_signals)
            M = ext_in.shape[0]  # Number of external signals
            if ext_signals.shape != (T, M):
                raise ValueError(f"ext_signals shape {ext_signals.shape} doesn't match expected ({T}, {M})")
        else:
            M = 0
            ext_signals = np.zeros((T, 0))
        
        # Create extended graph with external nodes
        # External nodes are numbered N, N+1, ..., N+M-1
        total_nodes = N + M
        extended_inter = np.zeros((total_nodes, total_nodes))
        extended_inter[:N, :N] = adjacency  # Internal connections
        
        if M > 0:
            # Add edges from external nodes to neurons
            for m in range(M):
                for n in range(N):
                    if ext_in[m, n]:
                        extended_inter[N + m, n] = 1
        
        G = nx.from_numpy_array(extended_inter, create_using=nx.DiGraph)
        pos = layout_func(G)                       # {node: (x, y)}
        
        # Separate neuron and external node positions
        neuron_pos = {i: pos[i] for i in range(N)}
        ext_pos = {i: pos[N + i] for i in range(M)} if M > 0 else {}

        # pre-compute constant drawing primitives
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_axis_off()
        
        # Draw edges (only between neurons, and from external to neurons)
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.6, alpha=.4, arrows=False)

        # Convert colors to rgba
        neuron_on_rgba = np.array(to_rgba(on_color))
        neuron_off_rgba = np.array(to_rgba(off_color))
        ext_on_rgba = np.array(to_rgba(ext_on_color))
        ext_off_rgba = np.array(to_rgba(ext_off_color))
        
        # Create scatter plots for neurons
        neuron_rgba = np.tile(neuron_off_rgba, (N, 1))
        neuron_scat = ax.scatter(*zip(*[neuron_pos[i] for i in range(N)]),
                                s=180, c=neuron_rgba, edgecolors='k', zorder=3)
        
        # Create scatter plots for external nodes if they exist
        
        if M > 0:
            ext_rgba = np.tile(ext_off_rgba, (M, 1))
            ext_scat = ax.scatter(*zip(*[ext_pos[i] for i in range(M)]),
                                s=180, c=ext_rgba, edgecolors='k', zorder=3, marker='s')  # Square markers for external
            
        #     # Add labels to external nodes
        #     for i in range(M):
        #         x, y = ext_pos[i]
        #         ax.text(x, y, f'E{i}', ha="center", va="center", fontsize=8, zorder=4, color='white', weight='bold')
        
        # # Add labels to neuron nodes
        # for i in range(N):
        #     x, y = neuron_pos[i]
        #     ax.text(x, y, str(i), ha="center", va="center", fontsize=10, zorder=4)

        time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes,
                            fontsize=10, ha='left', va='top')

        # ----------------------------------------------------------------- animation
        decay = np.exp(-1.0 / fade_tau)            # per-step multiplicative fade

        def update(frame):
            nonlocal neuron_rgba
            if M > 0:
                nonlocal ext_rgba
            
            # Update neuron colors
            # 1. fade all neurons
            neuron_rgba[:, :3] = neuron_off_rgba[:3] + (neuron_rgba[:, :3] - neuron_off_rgba[:3]) * decay
            # 2. flash the spiking neurons in this frame
            recently_fired = spikes[frame].astype(bool)
            neuron_rgba[recently_fired] = neuron_on_rgba
            neuron_scat.set_facecolors(neuron_rgba)
            
            # Update external node colors if they exist
            if M > 0:
                # 1. fade all external nodes
                ext_rgba[:, :3] = ext_off_rgba[:3] + (ext_rgba[:, :3] - ext_off_rgba[:3]) * decay
                # 2. flash the active external signals in this frame
                ext_active = ext_signals[frame].astype(bool)
                ext_rgba[ext_active] = ext_on_rgba
                ext_scat.set_facecolors(ext_rgba)
                
                # 3. update clock / step counter
                time_text.set_text(f't = {frame * dt:.3f}')
                return neuron_scat, ext_scat, time_text
            else:
                # 3. update clock / step counter
                time_text.set_text(f't = {frame * dt:.3f}')
                return neuron_scat, time_text

        anim = animation.FuncAnimation(fig, update, frames=T,
                                    interval=1000 / fps, blit=True)
        if save_as:
            print("Rendering animation – this can take a moment …")
            Writer = animation.writers['ffmpeg'] if save_as.endswith('.mp4') else animation.PillowWriter
            anim.save(save_as, writer=Writer(fps=fps))
            print("Saved to", save_as)
        else:
            plt.show()

    @staticmethod
    def animate_response(
        config: dm.Config,
        y: np.ndarray,
        y_ref: np.ndarray,
        spikes: np.ndarray,
        show_legends: bool = True,
        save_as: str = None,
        fps: int = 30
    ):
        
        plant_type = config.plant.type
        n_neurons = config.spiking_network.n_neurons
        N = len(y)
        Ts = config.plant.sampling_time
        
        # Create the figure and subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Set up the axes
        plant_name = plant_type.replace('_', ' ').title()
        ax1.set_title(f'{plant_name} System Simulation ({n_neurons} Controllers)')
        ax1.set_ylabel('Position' if plant_type != 'first_order' else 'Output')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Spike Raster Plot')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Time (s)')
        ax2.set_yticks([])
        ax2.set_ylim(0, 1)
        
        # Set x-axis limits
        time_max = (N - 1) * Ts
        ax1.set_xlim(0, time_max)
        ax2.set_xlim(0, time_max)
        
        # Set y-axis limits for position plot
        y_min = min(np.min(y), np.min(y_ref)) * 1.1
        y_max = max(np.max(y), np.max(y_ref)) * 1.1
        ax1.set_ylim(y_min, y_max)
        
        # Initialize empty line objects for position plot
        line_ref, = ax1.plot([], [], label='Reference Signal', linestyle='--', color='red', linewidth=2)
        line_pos, = ax1.plot([], [], label='Position', color='black', linewidth=2)
        
        if show_legends:
            ax1.legend()
        
        # Set up spike raster plot
        colors = plt.cm.tab10(np.linspace(0, 1, n_neurons))
        spike_lines = []
        
        # Initialize empty eventplot collections for each neuron
        for i in range(n_neurons):
            label = f'C{i+1}'
            # Create empty eventplot
            collections = ax2.eventplot([[]], lineoffsets=0.1 + i*0.8/n_neurons, 
                                      linelengths=0.8/n_neurons, colors=[colors[i]], 
                                      linewidths=2, label=label)
            spike_lines.append(collections)
        
        if show_legends:
            ax2.legend()
        
        # Time text
        time_text = ax1.text(0.02, 0.96, '', transform=ax1.transAxes,
                            fontsize=10, ha='left', va='top')
        
        # Animation function
        def animate(frame):
            # Current time index
            current_idx = frame
            if current_idx >= N:
                current_idx = N - 1
            
            # Update time arrays
            time_current = np.arange(current_idx + 1) * Ts
            
            # Update position lines
            line_ref.set_data(time_current, y_ref[:current_idx + 1])
            line_pos.set_data(time_current, y[:current_idx + 1])
            
            # Update spike raster plot
            for i in range(n_neurons):
                # Get spike times up to current frame for this neuron
                spike_indices = np.where(spikes[:current_idx + 1, i] > 0)[0]
                spike_times = spike_indices * Ts
                
                # Remove old collections and add new ones
                for collection in spike_lines[i]:
                    collection.remove()
                
                if len(spike_times) > 0:
                    collections = ax2.eventplot([spike_times], lineoffsets=0.1 + i*0.8/n_neurons, 
                                              linelengths=0.8/n_neurons, colors=[colors[i]], 
                                              linewidths=2)
                    spike_lines[i] = collections
                else:
                    spike_lines[i] = []
            
            # Update time text
            time_text.set_text(f't = {frame * Ts:.3f}s')
            
            # Return all artists that were modified
            artists = [line_ref, line_pos, time_text]
            for spike_collection_list in spike_lines:
                if spike_collection_list:  # Only add if not empty
                    artists.extend(spike_collection_list)
            
            return artists
        
        # Create animation
        # Use interval to control speed (milliseconds between frames)
        interval = max(1, int(1000 / fps))
        
        anim = animation.FuncAnimation(fig, animate, frames=N, 
                                     interval=interval, blit=False, repeat=True)
        
        # Save or show animation
        if save_as:
            print("Rendering animation – this can take a moment …")
            if save_as.endswith('.mp4'):
                Writer = animation.FFMpegWriter
            elif save_as.endswith('.gif'):
                Writer = animation.PillowWriter
            else:
                Writer = animation.FFMpegWriter
            
            writer = Writer(fps=fps)
            anim.save(save_as, writer=writer)
            print(f"Animation saved to {save_as}")
        else:
            plt.tight_layout()
            plt.show()
        
        return anim
    
    @staticmethod
    def plot_epistemic_variance(
        config: dm.Config,
        controllers: list
        ):
        # Plot controller epistemic variance arrays over time for all controllers
        n_neurons = config.spiking_network.n_neurons
        Ts = config.plant.sampling_time
        n_cols = min(n_neurons, 3)
        n_rows = (n_neurons + n_cols - 1) // n_cols
        colors = plt.cm.tab10(np.linspace(0, 1, n_neurons))
        show_legends = config.plotting.show_legends
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), 
                                sharex=True, sharey=True)
        # Ensure axes is always a 2D array for consistent indexing
        if n_neurons == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        for i in range(n_neurons):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Plot all three epistemic variance arrays
            epistemic_var = np.array(controllers[i].epistemic_variance)
            epistemic_var_spike = np.array(controllers[i].epistemic_variance_spike)
            epistemic_var_nospike = np.array(controllers[i].epistemic_variance_nospike)
            
            time_axis = np.arange(len(epistemic_var)) * Ts
            
            ax.plot(time_axis, epistemic_var, label='Epistemic Variance', 
                    color=colors[i], linewidth=2)
            ax.plot(time_axis, epistemic_var_spike, label='Epistemic Variance (Spike)', 
                    color=colors[i], linestyle='--', linewidth=2, alpha=0.8)
            ax.plot(time_axis, epistemic_var_nospike, label='Epistemic Variance (No Spike)', 
                    color=colors[i], linestyle=':', linewidth=2, alpha=0.8)
            
            
            if show_legends:
                ax.set_title(f'Controller {i+1} - Epistemic Variance Over Time')
                ax.set_ylabel('Epistemic Variance')
                ax.legend()
            ax.grid(True, alpha=0.3)
        # Hide unused subplots
        for i in range(n_neurons, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            ax.axis('off')
        plt.tight_layout()
        plt.show()
