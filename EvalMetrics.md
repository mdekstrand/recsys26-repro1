---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Evaluation Metrics

This notebook provides evaluation metrics for the RecSys 2026 tutorial on reproducibility.

+++

## Setup

We'll start by loading libraries:

```{code-cell} ipython3
from pathlib import Path
import re
```

```{code-cell} ipython3
import pandas as pd
import plotnine as pn
```

```{code-cell} ipython3
from lenskit.data import ItemListCollection, Dataset
from lenskit.metrics import MeasurementCollector, RBP, NDCG, ListLength, MeanPopRank, ExposureGini
```

And we'll load the recommendations & test data:

```{code-cell} ipython3
TRAIN_DATA = Path('data/ml-10m.train')
TEST_FILE = Path('data/ml-10m.test.parquet')
REC_DIR = Path('runs/ml-10m')
```

```{code-cell} ipython3
train_data = Dataset.load(TRAIN_DATA)
test_data = ItemListCollection.load_parquet(TEST_FILE)
```

```{code-cell} ipython3
recs = {}
for rf in REC_DIR.glob('*.recs.parquet'):
    recs[re.sub(r'\.recs\.parquet$', '', rf.name)] = ItemListCollection.load_parquet(rf)
```

## Computing Metrics

We'll compute several metrics of the recommendations.

```{code-cell} ipython3
mc = MeasurementCollector()
mc.add_metric(RBP)
mc.add_metric(NDCG)
mc.add_metric(ListLength)
mc.add_metric(MeanPopRank(train_data))
mc.add_metric(ExposureGini(items=train_data))
```

```{code-cell} ipython3
metrics = {
    name: mc.measure_run(rl, test_data)
    for (name, rl) in recs.items()
}
```

```{code-cell} ipython3
all_summaries = pd.DataFrame.from_dict({
    name: metrics[name].summary_metrics
    for name in metrics
}, orient='index')
all_summaries.index.name='model'
all_summaries
```

```{code-cell} ipython3
all_user_metrics = pd.concat({
    name: metrics[name].list_metrics
    for name in metrics
}, names=['model'])
all_user_metrics
```

## Visualizing Metrics

Now we'll visualize the metrics.

### Effectiveness

```{code-cell} ipython3
eff_metrics = pd.melt(all_user_metrics.reset_index(), id_vars=['model', 'user_id'], value_vars=['RBP', 'NDCG'], var_name='metric')
(
    pn.ggplot(eff_metrics)
    + pn.aes(x='model', y='value')
    + pn.stat_summary(geom='bar')
    + pn.coord_flip()
    + pn.facet_grid(cols='metric', scales='free')
)
```

### Exposure Spread (Gini)

We'll measure the extent of exposure distribution with the Gini coefficient.

```{code-cell} ipython3
(
    pn.ggplot(all_summaries.reset_index())
    + pn.aes(x='model', y='ExposureGini')
    + pn.geom_col()
    + pn.coord_flip()
)
```

```{code-cell} ipython3

```
