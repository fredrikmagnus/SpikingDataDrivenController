"""
Simple data models and configuration loader for spiking controller simulations.
"""

from pydantic import BaseModel, Field
from typing import Optional, Union, List, Literal
import ruamel.yaml


class SpikeForce(BaseModel):
    variability: bool = Field(False, description="Enable/disable spike force variability")
    mean: float = Field(1.0, description="Mean of Gaussian distribution for spike forces")
    std: float = Field(0.3, description="Standard deviation of Gaussian distribution")

class ReferenceTrackingCost(BaseModel):
    enable: Union[Literal['none', 'plant', 'spiking', 'all'], List[Literal['none', 'plant', 'spiking', 'all']]] = Field("plant", description="Enable reference tracking cost for specific inputs")
    cost: Union[float, List[float]] = Field(1, description="Cost for reference tracking")
    discount_factor: float = Field(1.0, description="Discount factor for future reference tracking costs (set to 1 to disable)")

class ExponentialTraces(BaseModel):
    enable: Optional[Literal['spike', 'plant', 'all']] = Field(None, description="Enable/disable exponential traces for output buffer and selected inputs (none, spike, plant or all)")
    decay: float = Field(0.95, description="Decay factor for exponential traces")
    reset_input_trace_on_spike: bool = Field(True, description="Reset input buffer trace when a spike occurs")

class Controller(BaseModel):
    gamma: float = Field(0.995, description="Forgetting factor for covariance matrices")
    Lp: int = Field(5, description="Length of observation horizon")
    Lf: int = Field(4, description="Length of prediction horizon")
    mu: float = Field(1e-3, description="Spike cost parameter")
    lambda_ridge: float = Field(1e-4, description="Ridge regression regularization parameter")
    update_on_spike: bool = Field(True, description="Update controller on spike")

    combine_spike_channels: Optional[Literal['all', 'sign']] = Field(None, description="Combine 'all' spike channels or by 'sign' (set to null for individual channels)")
    exponential_traces: ExponentialTraces = ExponentialTraces()
    spike_force: SpikeForce = SpikeForce()
    reference_tracking_cost: ReferenceTrackingCost = ReferenceTrackingCost()
    variance_minimizing_cost: float = Field(1.0, description="Cost for variance minimization")

class CustomConnectivity(BaseModel):
    adjacency_matrix: Optional[List[List[float]]] = Field(None, description="Custom adjacency matrix for connectivity")
    feedforward_connectivity: Optional[List[int]] = Field(None, description="Feedforward connectivity for each neuron (controller -> plant)")
    ext_in: Optional[List[float]] = Field(None, description="External input connections for each neuron (plant -> controller)")
    ext_out: Optional[List[float]] = Field(None, description="External output connections for each neuron (controller -> plant)")

class SpikingNetwork(BaseModel):
    n_neurons: int = Field(4, description="Number of neurons in the spiking network")
    connectivity: Literal['full', 'sign-alternating', 'sparse', 'custom', 'none'] = Field("full", description="Connectivity type")
    controller: Controller = Controller()
    custom_connectivity: CustomConnectivity = CustomConnectivity()


class SpringMass(BaseModel):
    m: float = Field(1.0, description="Mass")
    k: float = Field(1.0, description="Spring constant")
    c: float = Field(0.3, description="Damping coefficient")
    pos_init: float = Field(0.0, description="Initial position")
    vel_init: float = Field(-0.1, description="Initial velocity")


class DoubleIntegrator(BaseModel):
    damping: float = Field(0.3, description="Damping coefficient")
    pos_init: float = Field(0.0, description="Initial position")
    vel_init: float = Field(-0.1, description="Initial velocity")


class FirstOrder(BaseModel):
    tau: float = Field(1.0, description="Time constant")
    y_init: float = Field(0.0, description="Initial output")

class StepSignal(BaseModel):
    enable: bool = Field(True, description="Enable step reference signal")
    amplitude: float = Field(1.0, description="Amplitude of the step signal")
    sign_flips: int = Field(4, description="Number of flips in the step signal")

class SineSignal(BaseModel):
    enable: bool = Field(False, description="Enable sine reference signal")
    amplitude: float = Field(1.0, description="Amplitude of the sine signal")
    frequency: float = Field(0.1, description="Frequency of the sine signal in Hz")
    phase: float = Field(0.0, description="Phase shift of the sine signal in radians")
    offset: float = Field(0.0, description="Offset of the sine signal")

class PatternSignal(BaseModel):
    pattern: List[float] = Field([0, 1, 0, 1], description="Pattern for the external signal (to be repeated)")

class ExternalSignal(BaseModel):
    type: Literal['pattern', 'sine', 'step'] = Field("pattern", description="Type of external signal")
    pattern: PatternSignal = PatternSignal()
    sine: SineSignal = SineSignal()
    step: StepSignal = StepSignal()

class Plant(BaseModel):
    type: Literal['spring_mass', 'double_integrator', 'first_order', 'external_signal'] = Field("spring_mass", description="Plant type")
    spring_mass: SpringMass = SpringMass()
    double_integrator: DoubleIntegrator = DoubleIntegrator()
    first_order: FirstOrder = FirstOrder()
    external_signal: ExternalSignal = ExternalSignal()
    measurement_noise_amplitude: float = Field(0, description="Measurement noise amplitude")
    sampling_time: float = Field(0.05, description="Sampling time for the plant model")

class PlotMatrices(BaseModel):
    indices: Optional[List[int]] = Field(None, description="Indices of neurons to plot matrices for, or 'null' for all neurons")
    plot_covariance_matrices: bool = Field(False, description="Plot covariance matrices")
    plot_connectivity_matrices: bool = Field(False, description="Plot connectivity matrices")
    plot_gain_matrices: bool = Field(False, description="Plot gain matrices")

class Plotting(BaseModel):
    show_legends: bool = Field(True, description="Show legends in plots")
    plot_response: bool = Field(True, description="Plot system response")
    compare_predictions: bool = Field(True, description="Compare predictions")
    plot_voltage_threshold: bool = Field(True, description="Plot voltage and threshold for each controller")
    plot_variance: bool = Field(False, description="Plot epistemic variance for each controller")

    highlight_range: Optional[List[int]] = Field(None, description="Highlight specific time range in plots")

    plot_matrices: PlotMatrices = PlotMatrices()
    
    animate_response: bool = Field(True, description="Animate the system response")
    animate_network: bool = Field(True, description="Animate the spiking network")
    save_animations: bool = Field(False, description="Save animations to files")

class ReferenceSignal(BaseModel):
    signal_type: Literal['step', 'sine'] = Field("step", description="Type of reference signal")
    step: StepSignal = StepSignal()
    sine: SineSignal = SineSignal()


class Simulation(BaseModel):
    simulation_time: float = Field(500, description="Total simulation time in seconds")
    base_seed: int = Field(1, description="Base random seed for controllers")
    reference_signal: ReferenceSignal = ReferenceSignal()


class Config(BaseModel):
    simulation: Simulation = Simulation()
    spiking_network: SpikingNetwork = SpikingNetwork()
    plant: Plant = Plant()
    plotting: Plotting = Plotting()

def read_data_from_yaml(full_file_path, data_class):
    """Read data from YAML file and create instance of specified data class."""
    with open(full_file_path, 'r') as stream:
        yaml = ruamel.yaml.YAML(typ='safe', pure=True)
        yaml_str = yaml.load(stream)
    return data_class(**yaml_str)
