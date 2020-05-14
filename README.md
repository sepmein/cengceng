# Infectious Disease Models
Python implementation of common infectious disease transmission models.

A Gallery of infectious disease models for the quick usage in the infectious disease control and prevention.

## quick start
This project use `pytorch` as the backend mathmatical framework, a pre-installed GPU version of pytorch is recommended.
```bash
pip install infectious_disease_model
```

## How to use

## Models:

### Compartment Models
These types of models seperate each individual into **states**.
- mosquito resistance prediction

#### Traditional Infectious Disease Models
- sir
- seir
- logistic

#### Modified Infectious Disease Models
- seir-caq

#### Malaria Transmission Model

##### States
At each point people can be in one of six states:
- *S* - susceptible
- *T* - treated clinical disease
- *D* - untreated clinical disease
- *A* - asymptomatic infection(which maybe detected by microscopy)
- *U* - sub-patent infection
- *P* - protected by a period of prophylaxis from prior treatment

## Development
**Skip this part if you are not intended to contribute to the project**
**But it will be helpful, if you understand the interior mechanism of how this projcet works**
### Compartmental Model's Implementations
This project use `pytorch` as a major mathmatical framework to calculus and differentiation. The parameters are all basically a torch variable.
Because models including `sir`, `seir` and other compartmental models can't directedly calculate the original function, the project use `torchdiffeq` to calculate the ode function.

### RNN Models
Recurrent Neural Network(RNN) is a class of recurrent neural network (RNN) is a class of artificial neural network where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior. Unlike feedforward neural networks, RNNs can use their internal state (memory) to process sequences of inputs. This makes them applicable to tasks such as unsegmented, connected handwriting recognition or speech recognition.

### Model Training

## References
All references are stored in the references folder and was stored as **.bib** type.
