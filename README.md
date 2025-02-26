# RAVEN: Reducing Attributes Via Evaluating Nearness [🐦‍⬛](https://en.wikipedia.org/wiki/Backronym)
![image](https://github.com/user-attachments/assets/e4b39edf-bc2d-480e-9b17-c2cc004f4fea)

An ultra-fast tool to reduce the attributes (features) of that insanely large dataset in a way that doesn't affect dataset quality. It does this by identifying clusters of linearly related (and therefore redundant) features, and only preserving the feature most 'near' to all other features.

## Dependencies

Make sure you have Pandas, NumPy and NetworkX installed. You can install these packages using `pip`

```
pip install pandas numpy networkx
```

## Usage

To use Raven, you can simply download the raw of [`raven.py`](raven.py) and import it as

```py
from raven import raven
```

Once you have it imported, you can identify redundant features. Here's an example usage:

```py
really_huge_dataset = pd.read_csv('./really_huge_dataset.csv')

redundant_features = raven(really_huge_dataset)

smaller_dataset = really_huge_dataset.drop(columns=redundant_features)
```
