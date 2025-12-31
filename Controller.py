import numpy as np
import DataModels as dm

class SpikingController:
    def __init__(self, 
                 controller_config: dm.Controller,
                 m_in_plant=1,
                 m_in_spikes=0, 
                 reference_tracking_cost_enable='none',
                 reference_tracking_cost=1,
                 rng=np.random.default_rng(),
                 save_predictions=False
                ):
        self.controller_config = controller_config
        self.m_in_plant = m_in_plant  # Number of inputs from the plant
        self.m_in_spikes = m_in_spikes  # Number of spiking inputs
        self.m_in = m_in_plant + m_in_spikes # Number of inputs (to the controller, from the plant!)
        self.p_out = 1 # Number of outputs (from the controller, to the plant)
        self.Lp = controller_config.Lp  # Length of past buffer
        self.Lf = controller_config.Lf  # Length of future buffer
        self.gamma = controller_config.gamma + 0*np.random.normal(0, 1e-3)**2  # Forgetting factor for the covariance matrices
        self.lambda_ridge = controller_config.lambda_ridge + 0*np.random.normal(0, 1e-4)**2  # Ridge regularization parameter
        self.reference_tracking_cost = reference_tracking_cost
        self.reference_tracking_cost_enable = reference_tracking_cost_enable

        self.mu = controller_config.mu + 0*np.random.normal(0, 1e-3)**2  # Spike cost parameter

        self.Q = self.set_cost_matrices()  # Cost matrices for reference tracking and input prediction

        self.L = self.Lp + self.Lf # Total length of the buffer
        self.d = self.Lp*(self.p_out+self.m_in) + self.Lf * self.p_out # Dimension of the hankel column (omitting the future inputs y to be predicted)
        
        # Initialize Sigma with random values
        self.rng = rng
        self.Sigma = self.lambda_ridge*(0*np.eye(self.d, self.d) + self.rng.random((self.d, self.d))) # Covariance matrix with random initialization
        self.Psi = 0*self.lambda_ridge * self.rng.random((self.m_in*self.Lf, self.d)) # Cross-correlation matrix 
        self.Sigma_inv = np.linalg.inv(self.Sigma)  # Inverse of Sigma
        
        self.u_buf = np.zeros((self.p_out, self.L)) # Control output buffer
        if self.controller_config.update_on_spike:
            self.u_buf[:, -1] = np.ones((self.p_out, 1))  # Initialize the last control output to 1 (spike)
        self.y_buf = np.zeros((self.m_in, self.L)) # Input buffer

        self.h_t = np.zeros((self.d, 1))

        self.save_predictions = save_predictions  # Flag to save predictions
        self.predictions = []  # To store predictions at each time-step if save_predictions is True
        self.epistemic_variance = []  # To store epistemic variance at each time-step
        self.epistemic_variance_spike = []  # To store epistemic variance when a spike occurs
        self.epistemic_variance_nospike = []  # To store epistemic variance when no spike occurs

        # self.gamma_trace = 0.95  # Decay factor for traces in buffers
        # # Make trace stochastic
        # self.gamma_trace += 1*self.rng.uniform(-0.02, 0.02)

        self.gamma_trace = controller_config.exponential_traces.decay if controller_config.exponential_traces.enable is not None else 0


    def step(self, k, y_t, y_ref=None):
        """
        Perform a single time-step update of the spiking controller.
        Parameters:
        -----------
        k : int
            Current time step.
        y_t : np.ndarray, shape (m_in, 1)
            Current plant output (input to the controller).
        y_ref : np.ndarray, shape (m_in*Lf, 1), optional
            Reference trajectory for the next Lf time steps.
            If None, defaults to zero vector.
        Returns:
        --------
        u_t : np.ndarray, shape (p_out, 1)
            Control output (spikes) for the current time step.
        Voltage : float
            Computed membrane potential (voltage) for the current time step.
        Threshold : float
            Computed firing threshold for the current time step.
        """
        
        # Ensure y_t has the correct shape
        if y_t.shape[0] != self.m_in:
            raise ValueError(f"y_t must have shape ({self.m_in}, 1), got {y_t.shape}")
        
        # Ensure y_ref has the correct shape
        if y_ref is not None and y_ref.shape[0] != self.m_in*self.Lf:
            raise ValueError(f"y_ref must have shape ({self.m_in*self.Lf}, 1), got {y_ref.shape}")

        if y_ref is None:
            y_ref = np.zeros((self.m_in*self.Lf, 1))
    
        # 1. Roll output buffer, insert y_t
        y_buf_last = self.y_buf[:, -1] # Store the last column of y_buf before rolling (for traces)
        self.y_buf = np.roll(self.y_buf, shift=-1, axis=1)
        self.y_buf[:, -1] = y_t.reshape(-1)
        
        # Adding traces
        if self.controller_config.exponential_traces.enable:
            if self.controller_config.exponential_traces.enable == 'spike' and self.m_in_spikes > 0:
                # Set entries corresponding to plant inputs to 0
                y_buf_last[:self.m_in_plant] = 0
            elif self.controller_config.exponential_traces.enable == 'plant' and self.m_in_plant > 0:
                # Set entries corresponding to spike inputs to 0
                y_buf_last[self.m_in_plant:] = 0

            self.y_buf[:, -1] += self.gamma_trace*y_buf_last # Add the decayed last output to the current output
            

        # If t < L (Initial phase), collect data and return a random spike
        if k < self.L:
            # u_t = np.random.uniform(0, 1, (self.p_out, 1))
            # Sample u_t from a Bernoulli distribution with p=0.2
            # if k == 0:
            #     u_t = np.ones((self.p_out, 1)) 
            # else:
            p = 0.2
            u_t = np.ones((self.p_out, 1)) * (self.rng.random((self.p_out, 1)) < p)
            self.u_buf = np.roll(self.u_buf, shift=-1, axis=1)
            self.u_buf[:, -1] = u_t.flatten()

            self.predictions.append(np.zeros((self.m_in*self.Lf,1)))

            return u_t, np.ones((1, 1)), self.mu

        # 2. Update Psi and Sigma
        
        # if ( not self.controller_config.exponential_traces.enable and np.sum(self.u_buf) > 0 ) or self.controller_config.update_on_spike == False:
        if np.sum(self.u_buf) > 0 or self.controller_config.update_on_spike == False:

            # y_f = [y_{t-Lf+1}, ..., y_t] = [y_f]
            # h_t = [u_{t-Lp-Lf}, ..., u_{t-1}, y_{t-Lp-Lf+1}, ..., y_{t-Lf}] = [u_p, u_f, y_p]
            y_f = self.y_buf[:, -self.Lf:].reshape(-1, 1)
            h_t = np.concatenate((
                self.u_buf.flatten(), 
                self.y_buf[:, :-self.Lf].flatten()
                )).reshape(-1, 1)

            if self.gamma == 1:
                self.Psi = self.Psi + y_f@h_t.T
            else:
                self.Psi = self.gamma*self.Psi + (1 - self.gamma) * y_f@h_t.T

            # Update the covariance matrix Sigma

            if self.gamma == 1:
                self.Sigma += h_t @ h_t.T
            else:
                self.Sigma = self.gamma * self.Sigma + (1 - self.gamma) * (h_t @ h_t.T + self.lambda_ridge * np.eye(self.d))
            self.Sigma_inv = np.linalg.inv(self.Sigma)

        # elif self.controller_config.exponential_traces.enable != None:
        #     y_f = self.y_buf[:, -self.Lf:].reshape(-1, 1)
        #     h_t = np.concatenate((
        #         self.u_buf.flatten(), 
        #         self.y_buf[:, :-self.Lf].flatten()
        #         )).reshape(-1, 1)
            
        #     gamma_eff = 1 + (self.gamma-1) * self.u_buf[:, -1]

        #     self.Psi = gamma_eff*self.Psi + (1-gamma_eff) * y_f@h_t.T
        #     self.Sigma = gamma_eff * self.Sigma + (1 - gamma_eff) * (h_t @ h_t.T + self.lambda_ridge * np.eye(self.d))
        #     self.Sigma_inv = np.linalg.inv(self.Sigma)


        # 3. Collect most recent Lp outputs and inputs
        u_p = self.u_buf[:, -self.Lp:]
        y_p = self.y_buf[:, -self.Lp:]

        # 4. Compute the control output
        # Testing with traces
        if self.controller_config.exponential_traces.enable:
            u_s = np.array([self.gamma_trace**i for i in range(self.Lf)]).reshape(-1, 1)  # Future spike input with trace
        else:
            u_s = np.vstack((1, np.zeros((self.Lf * self.p_out-1, 1))))  # Spike now
        
        h_s = np.vstack((
            u_p.reshape(-1, 1), 
            u_s,
            y_p.reshape(-1, 1)
            ))

        u_ns = np.zeros((self.Lf * self.p_out, 1))
        h_ns = np.vstack((
            u_p.reshape(-1, 1),
            u_ns,
            y_p.reshape(-1, 1)
            ))
        
        # Spiking condition:
        K = self.Psi @ self.Sigma_inv

        y_pred_s = K @ h_s # Predicted output with spike
        y_pred_ns = K @ h_ns # Predicted output without spike

        # print(y_pred_s.shape)

        #if self.save_predictions:
        
        # Epistemic uncertainty estimation:
        epistemic_var_spike = self.lambda_ridge*h_s.T @ self.Sigma_inv @ h_s
        self.epistemic_variance_spike.append(epistemic_var_spike.flatten()[0])

        epistemic_var_nospike = self.lambda_ridge*h_ns.T @ self.Sigma_inv @ h_ns
        self.epistemic_variance_nospike.append(epistemic_var_nospike.flatten()[0])

        # Compute the voltage:
        # Reference tracking term:
        V_ref = (y_pred_ns - y_ref).T @ self.Q @ (y_pred_ns - y_ref) - (y_pred_s - y_ref).T @ self.Q @ (y_pred_s - y_ref)
        
        # Add voltage based on epistemic uncertainty (Choose lower uncertainty action)
        V_epistemic = (epistemic_var_spike - epistemic_var_nospike)
        V_epistemic *= self.controller_config.variance_minimizing_cost
            
        # Voltage = (y_pred_ns - y_ref).T @ self.Q @ (y_pred_ns - y_ref) - (y_pred_s - y_ref).T @ self.Q @ (y_pred_s - y_ref)
        Voltage = V_ref + V_epistemic
        # V_noise = np.random.normal(0, 3e-2, size=Voltage.shape)  # Add some noise to the voltage
        # Voltage += V_noise

        # # Attempt at adaptive threshold:
        # tr = np.trace(self.Sigma_inv)
        # tau_max = self.d / self.lambda_ridge
        # tau_min = 0.5 * tau_max
        # ratio = np.clip((tr - tau_min) / (tau_max - tau_min), 0.0, 1.0)

        # Threshold = self.mu * (1 - 2 * ratio)

        Threshold = self.mu

        # Spike condition:
        # spike = (Voltage >= Threshold)
        
        # Additional condition to prevent multiple spikes in a row (refractory period)
        spike = (Voltage >= Threshold) and self.u_buf[:, -1].sum() < self.p_out

        if spike:
            self.predictions.append(y_pred_s)
            epistemic_var = epistemic_var_spike

            # print(epistemic_var_spike, epistemic_var_nospike, epistemic_var_spike - epistemic_var_nospike, "SPIKE")
            # epistemic_var = self.lambda_ridge*h_s.T @ self.Sigma_inv @ h_s 
        else:
            self.predictions.append(y_pred_ns)
            epistemic_var = epistemic_var_nospike
            # print(epistemic_var_spike, epistemic_var_nospike, epistemic_var_spike - epistemic_var_nospike, self.u_buf[:, -1].sum())
            # epistemic_var = self.lambda_ridge*h_ns.T @ self.Sigma_inv @ h_ns
            
        self.epistemic_variance.append(epistemic_var.flatten()[0]) # Store the epistemic variance for the chosen action
    
        # Only spike if no spike in the last Lf steps (to keep consistent with the spike vs no-spike prediction)
        # spike = (Voltage >= Threshold) and self.u_buf[:, self.Lp:].sum() == 0 
        
        # print(self.u_buf, self.u_buf[:, self.Lp:])
        u_t = np.ones((self.p_out, 1)) * spike

        # 5. Update the control output buffer
        # Adding traces
        # Store the last column of u_buf and decay it
        if self.controller_config.exponential_traces.enable:
            u_buf_last = self.gamma_trace*self.u_buf[:, -1] #

        self.u_buf = np.roll(self.u_buf, shift=-1, axis=1) # Roll the buffer to the left
        # self.u_buf[:, -1] = u_t.flatten() + u_buf_last # Add the decayed last output to the current output
        self.u_buf[:, -1] = u_t.flatten() if not self.controller_config.exponential_traces.enable else u_t.flatten() + u_buf_last

        # Resetting the input traces when firing a spike
        if spike and self.controller_config.exponential_traces.reset_input_trace_on_spike and self.controller_config.exponential_traces.enable:
            if self.controller_config.exponential_traces.enable == 'spike' and self.m_in_spikes > 0:
                self.y_buf[self.m_in_plant:, -1] = 0  # Reset spike input traces to zero when a spike occurs
            elif self.controller_config.exponential_traces.enable == 'plant' and self.m_in_plant > 0:
                self.y_buf[:self.m_in_plant, -1] = 0
            elif self.controller_config.exponential_traces.enable == 'all':
                self.y_buf[:, -1] = 0
        


        return u_t, Voltage, Threshold

    def get_gain_matrices(self):
        Sigma_inv = np.linalg.inv(self.Sigma)
        K = self.Psi @ Sigma_inv
        return Sigma_inv, self.Psi, K
    
    def set_cost_matrices(self):
        discount_factor = self.controller_config.reference_tracking_cost.discount_factor
        if self.reference_tracking_cost_enable == 'none':
            Q = np.zeros((self.Lf * self.m_in, self.Lf * self.m_in))  # No reference tracking cost
        elif self.reference_tracking_cost_enable == 'plant':
            Q = self.reference_tracking_cost * np.diag([discount_factor**i * (i < self.Lf*self.m_in_plant) for i in range(self.Lf * self.m_in)])
        elif self.reference_tracking_cost_enable == 'spiking':
            Q = self.reference_tracking_cost * np.diag([1 * (i >= self.Lf*self.m_in_plant) for i in range(self.Lf * self.m_in)])
        elif self.reference_tracking_cost_enable == 'all':
            Q = self.reference_tracking_cost * np.diag([1 for i in range(self.Lf * self.m_in)])

        return Q

class NetworkBuilder:
    @staticmethod
    def create_connectivity(network_params: dm.SpikingNetwork):
        """
        Creates:
            - adjacency: The adjacency matrix for inter-neuron connectivity based on the specified type.
                Shape: (total_neurons, total_neurons)
                Element ij: 1 if neuron i connects to neuron j, 0 otherwise.
            - ext_in: Connectivity matrix for external inputs to neurons (from plant to neurons).
                Shape: (plant_output_dim, total_neurons)
            - ext_out: Connectivity matrix for external outputs from neurons (from neurons to plant).
                Shape: (plant_input_dim, total_neurons)
        """
        n_neurons = network_params.n_neurons
        connectivity = network_params.connectivity
        
        # Initialize matrices
        adjacency = np.zeros((n_neurons, n_neurons))
        ext_in = np.zeros((1, n_neurons))  # Plant output dimension is 1 (position)
        ext_out = np.zeros((1, n_neurons))  # Plant input dimension is 1 (force)

        if connectivity == 'full':
            adjacency.fill(1) # Fully connected network
            adjacency[np.arange(n_neurons), np.arange(n_neurons)] = 0 # No self-connections
            ext_in.fill(1)  # All neurons receive plant output
            ext_out[0, :n_neurons//2] = -1 # Half neurons apply negative force
            ext_out[0, n_neurons//2:] = 1  # Half neurons apply positive force
            # ext_out = np.array([(-1)**i for i in range(n_neurons)], dtype=float).reshape(1, -1)

        elif connectivity == 'none':
            ext_in.fill(1)  # All neurons receive plant output
            ext_out[0, :n_neurons//2] = -1 # Half neurons apply negative force
            ext_out[0, n_neurons//2:] = 1  # Half neurons apply positive force
            # ext_out = np.array([(-1)**i for i in range(n_neurons)], dtype=float).reshape(1, -1)

        elif connectivity == 'custom':
            custom_conn = network_params.custom_connectivity
            
            # Use custom adjacency matrix if provided
            if custom_conn.feedforward_connectivity is not None:
                adjacency = NetworkBuilder.feed_forward_adjacency(custom_conn.feedforward_connectivity)
            elif custom_conn.adjacency_matrix is not None:
                adjacency_array = np.array(custom_conn.adjacency_matrix)
                if adjacency_array.shape != (n_neurons, n_neurons):
                    raise ValueError(f"Custom adjacency matrix must be {n_neurons}x{n_neurons}, got {adjacency_array.shape}")
                adjacency = adjacency_array
            
            # Use custom external input connections if provided
            if custom_conn.ext_in is not None:
                ext_in_array = np.array(custom_conn.ext_in).reshape(1, -1)
                if ext_in_array.shape[1] != n_neurons:
                    raise ValueError(f"Custom ext_in must have {n_neurons} elements, got {ext_in_array.shape[1]}")
                ext_in = ext_in_array
            else:
                ext_in.fill(1)  # Default: all neurons receive plant output
            
            # Use custom external output connections if provided
            if custom_conn.ext_out is not None:
                ext_out_array = np.array(custom_conn.ext_out).reshape(1, -1)
                if ext_out_array.shape[1] != n_neurons:
                    raise ValueError(f"Custom ext_out must have {n_neurons} elements, got {ext_out_array.shape[1]}")
                ext_out = ext_out_array
            else:
                # Default: half neurons apply positive force, half apply negative force
                ext_out[0, :n_neurons//2] = 1
                ext_out[0, n_neurons//2:] = -1

        adjacency = adjacency.astype(int)  # Ensure adjacency is integer type

        return adjacency, ext_in, ext_out

    @staticmethod
    def create_network(config: dm.Config):
        """
        Create a spiking network based on the provided parameters.
        
        Parameters:
        -----------
        network_params : SpikingNetwork
            Configuration object containing network parameters
        
        Returns:
        --------
        list of SpikingController
            List of controllers representing the spiking network
        """
        network_params = config.spiking_network
        adjacency, ext_in, ext_out = NetworkBuilder.create_connectivity(network_params)

        print("Adjacency matrix:")
        print(adjacency)
        print("External input connections (ext_in):")
        print(ext_in)
        print("External output connections (ext_out):")
        print(ext_out)

        controllers = []

        # Handle reference_tracking_cost as either float or list
        reference_costs = network_params.controller.reference_tracking_cost.cost
        if isinstance(reference_costs, (int, float)):
            # Single value for all controllers
            reference_cost_list = [reference_costs] * network_params.n_neurons
        else:
            # List of values
            reference_cost_list = reference_costs
            if len(reference_cost_list) != network_params.n_neurons:
                raise ValueError(f"reference_tracking_cost list must have {network_params.n_neurons} elements, got {len(reference_cost_list)}")

        # Handle reference_tracking_cost_enable as either string or list of strings
        reference_tracking_cost_enable = network_params.controller.reference_tracking_cost.enable
        if isinstance(reference_tracking_cost_enable, str):
            reference_tracking_cost_enable = [reference_tracking_cost_enable] * network_params.n_neurons

        for i in range(network_params.n_neurons):
            m_in_plant = int(ext_in[:, i].sum())  # Number of inputs from the plant

            if network_params.controller.combine_spike_channels == 'all' and network_params.connectivity != 'none':
                m_in_spikes = 1  # All spike channels combined into one
            elif network_params.controller.combine_spike_channels == 'sign' and network_params.connectivity != 'none':
                m_in_spikes = 2  # Positive and negative spike channels combined
            else:
                m_in_spikes = int(adjacency[:, i].sum())  # Number of inputs from other neurons
            m_in = m_in_plant + m_in_spikes  # Total inputs for the controller
            controller = SpikingController(
                controller_config=network_params.controller,
                m_in_plant=m_in_plant,
                m_in_spikes=m_in_spikes,
                reference_tracking_cost_enable=reference_tracking_cost_enable[i],
                reference_tracking_cost=reference_cost_list[i],
                rng=np.random.default_rng(config.simulation.base_seed + i),
                save_predictions=True  # Save predictions for analysis
            )
            controllers.append(controller)

        return controllers, adjacency, ext_in, ext_out
    
    @staticmethod
    def feed_forward_adjacency(layer_neurons):
        """
        Return the adjacency matrix of a fully connected feed-forward network.

        Parameters
        ----------
        layer_neurons : list[int]
            Number of neurons in each successive layer, e.g. [1, 2, 1].

        Returns
        -------
        adj : np.ndarray, shape (N, N), dtype=int
            Adjacency matrix with entry (i, j) = 1 when neuron i connects to j.
            Neurons are indexed layer-by-layer starting from 0.
        """
        if not layer_neurons:
            raise ValueError("layer_neurons must contain at least one layer")
        if any(n <= 0 for n in layer_neurons):
            raise ValueError("each layer must have at least one neuron")

        total = sum(layer_neurons)
        adj = np.zeros((total, total), dtype=int)

        idx = 0                              # running index of first neuron in layer ℓ
        for n_curr, n_next in zip(layer_neurons[:-1], layer_neurons[1:]):
            curr = np.arange(idx, idx + n_curr)              # neuron indices in layer ℓ
            nex  = np.arange(idx + n_curr, idx + n_curr + n_next)  # indices in layer ℓ+1
            adj[np.repeat(curr, n_next), np.tile(nex, n_curr)] = 1
            idx += n_curr

        return adj