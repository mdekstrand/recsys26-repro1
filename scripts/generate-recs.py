#!/usr/bin/env python3
"""
Generate recommendations for test data.

Usage:
    generate-recs [-v] [-n N] PIPELINE DATASET

Options:
    -v, --verbose   Enable verbose logging output
    -n N, --num-recs=N
                    Number of recommendations to generate [default: 10]
    PIPELINE        Name of the pipeline to use.
    DATASET         Name of the dataset to use.
"""

from pathlib import Path

from docopt import docopt
from lenskit import Dataset, Pipeline, batch
from lenskit.data import ItemListCollection
from lenskit.logging import LoggingConfig, get_logger

_log = get_logger("generate-recs")

DATA_DIR = Path("data")
PIPELINE_DIR = Path("pipelines")
OUTPUT_DIR = Path("runs")


def main():
    args = docopt(__doc__)
    lc = LoggingConfig()
    if args["--verbose"]:
        lc.set_verbose()
    lc.apply()

    N = int(args["--num-recs"])
    pipe_name = args["PIPELINE"]
    pipe_file = PIPELINE_DIR / f"{pipe_name}.toml"
    _log.info("loading pipeline %s", pipe_file)
    pipe = Pipeline.load_config(pipe_file)

    ds_name = args["DATASET"]
    train_path = DATA_DIR / f"{ds_name}.train"
    test_path = DATA_DIR / f"{ds_name}.test.parquet"
    _log.info("loading dataset %s", train_path)
    train_data = Dataset.load(train_path)
    _log.info("loading test data")
    test_data = ItemListCollection.load_parquet(test_path)

    _log.info("training pipeline")
    pipe.train(train_data)

    _log.info("generating recommendations")
    recs = batch.recommend(pipe, test_data, n=N)
    out_file = OUTPUT_DIR / f"{ds_name}-{pipe_name}.recs.parquet"
    _log.info("saving recommendations to %s", out_file)
    recs.save_parquet(out_file)


if __name__ == "__main__":
    main()
