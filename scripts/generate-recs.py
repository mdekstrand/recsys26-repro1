#!/usr/bin/env python3
"""
Generate recommendations for test data.

Usage:
    generate-recs [-v] [-n N] [-o FILE] PIPELINE DATASET

Options:
    -v, --verbose   Enable verbose logging output
    -n N, --num-recs=N
                    Number of recommendations to generate [default: 10]
    -o FILE, --rec-out=FILE
                    Save recommendations to FILE.
    PIPELINE        Name of the pipeline or path to pipeline file to use.
    DATASET         Name of the dataset to use.
"""

from pathlib import Path

from docopt import docopt
from lenskit import Dataset, Pipeline, batch, configure
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
    # apply LensKit configuration
    configure()

    N = int(args["--num-recs"])
    pipe_name = args["PIPELINE"]
    if Path(pipe_name).exists():
        pipe_file = Path(pipe_name)
        pipe_name = pipe_file.name
    else:
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
    if out_path := args["--rec-out"]:
        out_file = Path(out_path)
        out_dir = out_file.parent
    else:
        out_dir = OUTPUT_DIR / ds_name
        out_file = out_dir / f"{pipe_name}.recs.parquet"

    out_dir.mkdir(exist_ok=True, parents=True)
    _log.info("saving recommendations to %s", out_file)
    recs.save_parquet(out_file)


if __name__ == "__main__":
    main()
