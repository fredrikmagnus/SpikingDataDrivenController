import numpy as np
from scipy.linalg import expm   # exact matrix exponential
import DataModels as dm

class DoubleIntegrator:
    """
    Double integrator system (non-oscillatory) in discrete time.
    
    ÿ = u (no spring constant, just mass and damping)
    
    State  x = [position; velocity]
    Input  u = force/acceleration
    Output y = position
    
    This system has no inherent oscillatory properties - it's purely
    integrative with optional damping.
    
    Parameters
    ----------
    damping : float
        Damping coefficient (0 = pure double integrator, >0 adds damping).
    Ts : float
        Sampling time interval.
    """
    def __init__(self, damping=0.1, Ts=0.01, x0=np.zeros((2,1))):
        self.damping, self.Ts = damping, Ts
        
        # continuous-time matrices (no spring term k=0)
        A_c = np.array([[0,        1],
                        [0,  -damping]])  # No spring restoring force
        B_c = np.array([[0],
                        [1]])  # Direct acceleration input
        C   = np.array([[1, 0]])
        D   = np.zeros((1,1))
        
        # exact discretisation using matrix exponential
        M = np.block([[A_c, B_c],
                      [np.zeros((1,3))]])
        M_d = expm(M*Ts)
        self.A = M_d[:2, :2]
        self.B = M_d[:2, 2:3]
        self.C = C
        self.D = D
        
        self.x = x0
        self.y = self.C @ self.x + self.D * 0
    
    def update_params(self):
        A_c = np.array([[0,        1],
                       [0,  -self.damping]])
        B_c = np.array([[0],
                       [1]])
        C   = np.array([[1, 0]])
        D   = np.zeros((1,1))
        
        M = np.block([[A_c, B_c],
                      [np.zeros((1,3))]])
        M_d = expm(M*self.Ts)
        self.A = M_d[:2, :2]
        self.B = M_d[:2, 2:3]
        self.C = C
        self.D = D
    
    def step(self, u):
        """
        Parameters
        ----------
        u : ndarray shape (1,1)
            Acceleration/force input.
            
        Returns
        -------
        y : ndarray shape (1,1)
            Position output after state update.
        """
        self.x = self.A @ self.x + self.B * u
        self.y = self.C @ self.x + self.D * u
        return self.y

class FirstOrderSystem:
    """
    Simple first-order system (completely non-oscillatory).
    
    ẏ + (1/τ)y = (1/τ)u
    
    State  x = [output]
    Input  u = reference input
    Output y = output
    
    This is a simple low-pass filter with no oscillatory behavior.
    
    Parameters
    ----------
    tau : float
        Time constant of the system.
    Ts : float
        Sampling time interval.
    """
    def __init__(self, tau=1.0, Ts=0.01, x0=np.zeros((1,1))):
        self.tau, self.Ts = tau, Ts
        
        # continuous-time: ẏ = -(1/τ)y + (1/τ)u
        A_c = np.array([[-1/tau]])
        B_c = np.array([[1/tau]])
        C   = np.array([[1]])
        D   = np.zeros((1,1))
        
        # exact discretisation
        self.A = np.array([[np.exp(-Ts/tau)]])
        self.B = np.array([[1 - np.exp(-Ts/tau)]])
        self.C = C
        self.D = D
        
        self.x = x0
        self.y = self.C @ self.x + self.D * 0
    
    def update_params(self):
        self.A = np.array([[np.exp(-self.Ts/self.tau)]])
        self.B = np.array([[1 - np.exp(-self.Ts/self.tau)]])
    
    def step(self, u):
        """
        Parameters
        ----------
        u : ndarray shape (1,1)
            Input signal.
            
        Returns
        -------
        y : ndarray shape (1,1)
            Output after state update.
        """
        self.x = self.A @ self.x + self.B * u
        self.y = self.C @ self.x + self.D * u
        return self.y

class SpringMassDamper:
    """
    1-DOF spring-mass-damper (position output, force input) in discrete time.

        m ẍ + c ẋ + k x = u

    State  x = [position; velocity]
    Input  u = force
    Output y = position

    Parameters
    ----------
    m : float
        Mass of the system.
    k : float
        Spring constant.
    c : float
        Damping coefficient.
    Ts : float
        Sampling time interval.
    """
    def __init__(self, m=1.0, k=1.0, c=0.1, Ts=0.01, x0=np.zeros((2,1))):
        self.m, self.k, self.c, self.Ts = m, k, c, Ts

        # 1. continuous-time matrices
        A_c = np.array([[0,        1],
                        [-k/m,  -c/m]])
        B_c = np.array([[0],
                        [1/m]])
        C   = np.array([[1, 0]])
        D   = np.zeros((1,1))

        # 2. exact discretisation  (using matrix exponential)
        M = np.block([[A_c, B_c],
                      [np.zeros((1,3))]])      # augment for integral
        M_d = expm(M*Ts)                       # expm([[A, B],[0,0]])
        self.A = M_d[:2, :2]                   # e^{A_c Ts}
        self.B = M_d[:2,  2:3]                 # ∫_0^{Ts} e^{A τ} B dτ
        self.C = C
        self.D = D

        self.x = x0               # initial state
        self.y = self.C @ self.x + self.D * 0  # initial output

    def update_params(self):
        A_c = np.array([[0,        1],
                     [-self.k/self.m,  -self.c/self.m]])
        B_c = np.array([[0],
                     [1/self.m]])
        C   = np.array([[1, 0]])
        D   = np.zeros((1,1))

        # exact discretisation  (using matrix exponential)
        M = np.block([[A_c, B_c],
                      [np.zeros((1,3))]])      # augment for integral
        M_d = expm(M*self.Ts)                       # expm([[A, B],[0,0]])
        self.A = M_d[:2, :2]                   # e^{A_c Ts}
        self.B = M_d[:2,  2:3]                 # ∫_0^{Ts} e^{A τ} B dτ
        self.C = C
        self.D = D
        

    # 3. one simulation step -------------------------------------------------
    def step(self, u):
        """
        Parameters
        ----------
        u : ndarray shape (1,1)
            Force applied during the current interval [kTs,(k+1)Ts).

        Returns
        -------
        y : ndarray shape (1,1)
            Output *after* the state update, i.e. y_{k+1}.
        """
        self.x = self.A @ self.x + self.B * 10*u
        self.y = self.C @ self.x
        return self.y


class NonlinearPendulum:
    """
    Nonlinear pendulum system with force input and sine/cosine angle observations.
    
    The dynamics are:
        θ̈ = -(g/L) * sin(θ) - (c/mL²) * θ̇ + (1/mL²) * u
    
    Where:
    - θ is the angle from vertical (positive counterclockwise)
    - g is gravitational acceleration
    - L is pendulum length
    - m is pendulum mass
    - c is damping coefficient
    - u is applied torque/force
    
    State  x = [angle; angular_velocity]
    Input  u = applied torque
    Output y = [sin(θ); cos(θ); θ̇] (3-dimensional output)
    
    The sine and cosine outputs allow neurons to properly observe the angle,
    and reference signals (angles) should be mapped to [sin(ref), cos(ref)]
    for the controller.
    
    Parameters
    ----------
    g : float
        Gravitational acceleration (default: 9.81 m/s²).
    L : float
        Pendulum length (default: 1.0 m).
    m : float
        Pendulum mass (default: 1.0 kg).
    c : float
        Damping coefficient (default: 0.1).
    Ts : float
        Sampling time interval.
    x0 : ndarray, optional
        Initial state [angle, angular_velocity]. Default is [0, 0].
    """
    
    def __init__(self, g=9.81, L=1.0, m=1.0, c=0.1, Ts=0.01, x0=None):
        self.g, self.L, self.m, self.c, self.Ts = g, L, m, c, Ts
        
        if x0 is None:
            x0 = np.zeros((2, 1))
        self.x = x0.copy()
        
        # Initial output
        self.y = self._compute_output()
    
    def _compute_output(self):
        """Compute the output [sin(θ), cos(θ), θ̇] from current state."""
        theta = self.x[0, 0]
        theta_dot = self.x[1, 0]
        return np.array([[np.sin(theta)],
                        [np.cos(theta)],
                        [theta_dot]])
    
    def _wrap_angle(self, angle):
        """Wrap angle to [-π, π] range."""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def _dynamics(self, x, u):
        """
        Compute the continuous-time dynamics: ẋ = f(x, u)
        
        Parameters
        ----------
        x : ndarray shape (2,)
            Current state [angle, angular_velocity].
        u : float
            Applied torque.
            
        Returns
        -------
        x_dot : ndarray shape (2,)
            State derivative [angular_velocity, angular_acceleration].
        """
        theta, theta_dot = x[0], x[1]
        
        # Nonlinear pendulum dynamics
        theta_ddot = (-(self.g / self.L) * np.sin(theta) 
                     - (self.c / (self.m * self.L**2)) * theta_dot 
                     + (1 / (self.m * self.L**2)) * u)
        
        return np.array([theta_dot, theta_ddot])
    
    def step(self, u):
        """
        Perform one simulation step using 4th-order Runge-Kutta integration.
        
        Parameters
        ----------
        u : ndarray shape (1,1) or float
            Applied torque during this time step.
            
        Returns
        -------
        y : ndarray shape (3,1)
            Output [sin(θ), cos(θ), θ̇] after state update.
        """
        # Convert input to scalar if needed
        if hasattr(u, 'shape') and u.shape == (1, 1):
            u = u[0, 0]
        elif hasattr(u, '__len__'):
            u = u[0]
        
        # Current state as 1D array for RK4
        x_current = self.x.flatten()
        
        # 4th-order Runge-Kutta integration
        dt = self.Ts
        k1 = self._dynamics(x_current, u)
        k2 = self._dynamics(x_current + 0.5 * dt * k1, u)
        k3 = self._dynamics(x_current + 0.5 * dt * k2, u)
        k4 = self._dynamics(x_current + dt * k3, u)
        
        # Update state
        x_new = x_current + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Wrap angle to [-π, π]
        x_new[0] = self._wrap_angle(x_new[0])
        
        # Update internal state
        self.x = x_new.reshape((2, 1))
        
        # Compute and return output
        self.y = self._compute_output()
        return self.y
    
    def update_params(self):
        """Update parameters (for compatibility with other systems)."""
        # Parameters are directly stored as instance variables
        pass
    
    @staticmethod
    def map_reference_to_observations(angle_ref):
        """
        Map a reference angle to sine/cosine observations for the controller.
        
        Parameters
        ----------
        angle_ref : float or ndarray
            Reference angle(s) in radians.
            
        Returns
        -------
        ref_obs : ndarray shape (2,1) or (2,N)
            Reference observations [sin(angle_ref), cos(angle_ref)].
        """
        if np.isscalar(angle_ref):
            return np.array([[np.sin(angle_ref)],
                           [np.cos(angle_ref)]])
        else:
            return np.array([np.sin(angle_ref),
                           np.cos(angle_ref)])

class ExternalSignal:
    def __init__(self, config: dm.Config):
        self.config = config
        self.Ts = config.plant.sampling_time
        self.signal_type = config.plant.external_signal.type
        self.sim_time = config.simulation.simulation_time

        self.step_signal_buf = None  # Buffer for step signal to avoid recomputing
        self.pattern_signal_buf = None  # Buffer for pattern signal to avoid recomputing

        self.y = 0 # temporary fix

        self.t = 0.0  # Current time in simulation

    def step(self, u=None):
        self.t += self.Ts
        if self.signal_type == 'pattern':
            return self.pattern_signal(self.t)
        elif self.signal_type == 'sine':
            return self.sine_signal(self.t)
        elif self.signal_type == 'step':
            return self.step_signal(self.t)

    def pattern_signal(self, t):
        """
        Generate a pattern signal based on the configured type.

        Parameters
        ----------
        t : float
            Current time in the simulation.

        Returns
        -------
        y : ndarray shape (1,1)
            The output of the pattern signal.
        """
        if self.pattern_signal_buf is None:
            pattern = np.array(self.config.plant.external_signal.pattern.pattern)
            period = len(pattern)
            N = int(self.sim_time / self.Ts)
            self.pattern_signal_buf = np.zeros((N, 1))  # Initialize buffer
            for i in range(N):
                self.pattern_signal_buf[i] = pattern[i % period]
        
        return self.pattern_signal_buf[int(t / self.Ts)].reshape((1, 1))

    def sine_signal(self, t):
        """
        Generate a sine wave signal based on the configured parameters.

        Parameters
        ----------
        t : float
            Current time in the simulation.

        Returns
        -------
        y : ndarray shape (1,1)
            The output of the sine wave signal.
        """
        amplitude = self.config.plant.external_signal.sine.amplitude
        frequency = self.config.plant.external_signal.sine.frequency
        phase = self.config.plant.external_signal.sine.phase
        offset = self.config.plant.external_signal.sine.offset

        return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset
    
    def step_signal(self, t):

        amplitude = self.config.plant.external_signal.step.amplitude
        ref_sign_flips = self.config.plant.external_signal.step.sign_flips
        N = int(self.sim_time / self.Ts)

        # Create and store the signal in a buffer to avoid recomputing
        if self.step_signal_buf is None:  
            self.step_signal_buf = np.ones((N, 1)) * amplitude  # Start with a constant reference signal
            flip_indices = [N // (ref_sign_flips + 1) * i for i in range(1, ref_sign_flips + 1)]
            for idx in flip_indices:
                self.step_signal_buf[idx:] *= -1

        return self.step_signal_buf[int(t / self.Ts)].reshape((1, 1))


def create_plant(config: dm.Config):
    plant_params = config.plant
    if plant_params.type == "spring_mass":
        plant = SpringMassDamper(
            m=plant_params.spring_mass.m,
            k=plant_params.spring_mass.k,
            c=plant_params.spring_mass.c,
            Ts=plant_params.sampling_time,
            x0=np.array([[plant_params.spring_mass.pos_init], 
                          [plant_params.spring_mass.vel_init]])
        )
    elif plant_params.type == "double_integrator":
        plant = DoubleIntegrator(
            damping=plant_params.double_integrator.damping,
            Ts=plant_params.sampling_time,
            x0=np.array([[plant_params.double_integrator.pos_init], 
                          [plant_params.double_integrator.vel_init]])
        )
    elif plant_params.type == "first_order":
        plant = FirstOrderSystem(
            tau=plant_params.first_order.tau,
            Ts=plant_params.sampling_time,
            x0=np.array([[plant_params.first_order.y_init]])
        )
    elif plant_params.type == "external_signal":
        plant = ExternalSignal(
            config=config
        )
    else:
        raise ValueError(f"Unknown plant type: {plant_params.type}")
    
    return plant