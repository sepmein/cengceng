# Infectious Disease Models
Python implementation of common infectious disease transmission models.

A Gallery of infectious disease models for the quick usage in the infectious disease control and prevention.

> Cengceng is a name of a cat my son loves.

## quick start
### 1. Installation
This project use `pytorch` as the backend mathmatical framework, a pre-installed GPU version of pytorch is recommended.
```bash
pip install cengceng
```

### 2. Usage

1. Project predefined model
```python
from cengceng.compartmental import Sir

model = Sir()

# load data from csv file
model.load_data('./covid-19-cn.csv')
model.fit()
model.save('output/result.csv')
```
That's it.

2. Customize your own model

```python
from cengceng.models import Model

# Inherit from basic model class
class My_Own_Model(Model):
		def __init__(self):
				super().__init__()
```

## Models:

### Compartmental Models
These types of models seperate each individual into **states**. And the speed of one state to another was predefined. In `cengceng`, we define the state change function in the `forward()` function of model class. Models included:
- SIR
- SEIR
- mosquito resistance prediction

For instance, in the simplest **SIR** model:
```python
class SIR(Model):
		# defined in the forward function
    def forward(self, s0, i0, r0):
        si = s0
        ii = i0
        ri = r0
        dsdt = -self.beta * si * ii
        didt = self.beta * si * ii - self.gama * ii
        drdt = self.gama * ii
        return torch.cat((dsdt, didt, drdt))
```

### Definitive Models
Models with definitive function

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

### Create Your Own Model

1. Compartmental Model
Inherite from Compartmental Class, use`add_trainable_parameter` function to add trainable parameters.

```pytorch
class My_Compartmental_Model(Compartmental):
		def __init__(self, para1=1e-5):
				super().__init__()
				self.add_trainable_parameter('para1', para1)
```

### Compartmental Model's Implementations
This project use `pytorch` as a major mathmatical framework to calculus and differentiation. The parameters are all basically a torch variable.
Because models including `sir`, `seir` and other compartmental models can't directedly calculate the original function, the project use `torchdiffeq` to calculate the ode function.



### RNN Models
Recurrent Neural Network(RNN) is a class of recurrent neural network (RNN) is a class of artificial neural network where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior. Unlike feedforward neural networks, RNNs can use their internal state (memory) to process sequences of inputs. This makes them applicable to tasks such as unsegmented, connected handwriting recognition or speech recognition.

### Model Training

## References
All references are stored in the references folder and was stored as **.bib** type.
