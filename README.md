# Class Label Noise Simulation

Utility methods for simulating class noise, as used in [The Effects of Class Label Noise on Highly-Imbalanced Big Data](https://ieeexplore.ieee.org/document/9643276) by Johnson and Khoshgoftaar.

---

## Installation

Poetry

```
poetry add git+https://github.com/johnsonj561/Class-Noise-Simulation
```

PIP

```
pip install git+https://github.com/johnsonj561/Class-Noise-Simulation
```

---

## Noise Injection with Class Probability

This method adds noise to the positive and negative class by randomly flipping labels in each class with some probability.

Usage

```python
from class_noise_simulation import add_noise_by_prob
import numpy as np

clean_y = np.concatenate((np.ones(pos_count), np.zeros(neg_count)))

noisy_y = add_noise_by_prob(
  clean_y,
  neg_noise_prob=0.1
  pos_noise_prob=0.5
  neg_label=0,
  pos_label=1
)
```

`neg_noise_prob` defines the percentage of negative instances that are flipped to positive.

`pos_noise_prob` defines the percentage of positive instances that are flipped to negative.

---

## Noise Injection with Lambda and Psi

This method adds noise to the positive and negative class by randomly flipping labels in each class based on the paper: [Knowledge Discovery from imbalanced noisy data](https://www.sciencedirect.com/science/article/abs/pii/S0169023X09001141) by Jason Van Hulse and Taghi Khoshgoftaar.

Usage

```python
from class_noise_simulation import add_noise_by_prob
import numpy as np

clean_y = np.concatenate((np.ones(pos_count), np.zeros(neg_count)))

noisy_y, pos_noise_count, neg_noise_count = add_noise_van_hulse(
  y,
  Lambda=0.2,
  Psi=0.1,
  neg_label=0,
  pos_label=1,
)
```

`Lambda` defines the noise level.

`Psi` defines the percentage of noise that belongs to the positive class.

`pos_noise_count` is the number of Pos -> Neg corruptions.

`neg_noise_count` is the number of Neg -> Pos corruptions.
