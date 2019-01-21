#!/usr/bin/env python3
import itertools
import argparse
import sys

from pathlib import Path
from collections import namedtuple
from dataclasses import replace
from config import Config

parser = argparse.ArgumentParser(
    description="""Generates config files for multiple configurations.
    It requires the source directory containing the jsons of the
    base configurations from which the new configurations have to be
    generated.
    """
)
parser.add_argument(
    "--src",
    type=Path,
    default=Path("conf"),
    help="Source directory where base configurations are found",
)
parser.add_argument(
    "--dest",
    type=Path,
    default=Path("gen_confs"),
    help="Destination directory where the files will be created",
)

Combination = namedtuple("Combination", ["T", "m_p", "dataset_name"])


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_filename(conf_dir: Path, dataset: str):
    return (conf_dir / dataset).with_suffix(".json")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.dest.exists():
        eprint(
            f"{args.dest} already exists, please delete it or change destination"
        )
        exit(1)

    args.dest.mkdir()

    if not args.src.exists():
        eprint(f"{args.src} does not exist, change source")
        exit(1)

    Ts = [3, 5, 10, 15, 25]
    m_p = [16, 32, 64, 128, 256]
    datasets = list(
        map(
            lambda f_path: f_path.with_suffix("").name, args.src.glob("*.json")
        )
    )

    base_confs = {
        d: Config.from_file(get_filename(args.src, d)) for d in datasets
    }
    combinations = itertools.product(Ts, m_p, datasets)
    for T, m_p, dataset in combinations:
        new_c_name = f"{dataset}_T{T}_m-p{m_p}"
        replace(
            base_confs[dataset],
            T=T,
            m=m_p,
            p=m_p,
            log_dir=str(Path(base_confs[dataset].log_dir) / new_c_name),
        ).to_file(get_filename(args.dest, new_c_name))
