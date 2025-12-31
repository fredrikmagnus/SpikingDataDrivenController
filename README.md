# Spiking Data-Driven Controller
This repository contains to the best of my knowledge the first implementation of a Spiking Data-Driven Controller for control of Linear Time-Invariant (LTI) systems. The controller is based on the Data-Driven Design framework, where control actions or multi-step-ahead predictions are computed directly based on measured input-output data, without requiring any explicit model of the system to be controlled. This implementation also avoids the need for storing large input-output data matrices by writing the predictor in terms of covariance matrices which are updated online as the controller interacts with the system.

The neuron is modelled as a data-driven predictor which predicts the future outputs of the system based on a short history of past inputs and outputs. The future outputs are predicted in two ways: (i) assuming that the neuron does not spike, and (ii) assuming that the neuron spikes. Quadratic costs are assigned to both predictions with respect to a reference signal, and a spike cost is added to the spiking prediction. The neuron spikes if the prediction with a spike yields a lower total cost than the non-spiking prediction. 

The control signal from the neuron is an impulse delivered to the system when the neuron spikes. The strength of the impulse is fixed, and hence two neurons with opposite spike signs are needed to deliver both positive and negative control actions. It is observed that greater control performance is achieved when the neurons receive the impulses from other neurons in addition to the system output, i.e., when the neurons are connected in a network. Performance is also observed to improve with more than two neurons in the network.

## Results
To illustrate the performance of the Spiking Data-Driven Controller, we consider control of a simple spring-mass-damper system using a fully connected network of spiking neurons. As illustrated below, each neuron receives the system output (the mass position) as well as the spikes from all other neurons in the network as inputs. The input to the system is the sum of the spikes from all neurons in the network.

![Illustration](./Figures/SMD_sys.png)


Using a network of 8 spiking neurons, the system is able to accurately track a square wave reference signal, as shown below. Note that at the start of the simulation, the neurons have no prior knowledge of the system dynamics, but learn to control the system within a few seconds of interaction.
| ![Response](./Figures/8NeuronControl.png) | ![Response_zoom](./Figures/8NeuronControl_zoom.png) |
|:---:|:---:|


## Usage
To run the simulation, simply execute the `run.py` script. The simulation parameters are specified in a configuration file in YAML format. Configuration files for 2, 4, and 8 neurons are provided in the repository, and can be adapted as desired.

