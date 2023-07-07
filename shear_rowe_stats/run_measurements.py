#copied directly from https://github.com/beckermr/misc/blob/68b42ea226913b97236ce5145378c6cf6fe05b8b/bin/run-simple-des-y3-sim
import logging
import sys
import numpy as np, pandas as pd

import click
import yaml

from prepping import download
from measuring import MeasurementRunner, ImageRunner

@click.group()
def cli():
    """Measure shapes of stars and PSFs."""
    pass


@cli.command()
@click.option('--web-path', type=str, required=True, help='the web path to the exposure image file')
@click.option('--output-path', type=str, required=True)
@click.option('--exp-filename', type=str, required=True, help = 'name of the exposure file')
def prep(web_path, output_path, exp_filename):
    download(web_path, output_path, exp_filename)



@cli.command()
@click.option('--start', type=str, required=True, help='start index')
@click.option('--end', type=str, required=True, help='end index')
@click.option('--path-to-explist', type=str, required=True, help='list of exposure names')
@click.option('--seed', type=int, required=True, help='the base RNG seed')
def run(start, end, path_to_explist, seed):

    #Get filenames and remove r{procid}_immasked.fits
    df = pd.read_csv(path_to_explist).drop_duplicates().sort_values('EXPNUM')
    explist = np.char.ljust(df.FILENAME.values.astype(str), 15)
    magzp   = df.MAG_ZERO.values.astype(str)
    runner  = ImageRunner(start, end, seed, explist, magzp)
    runner.go()


if __name__ == '__main__':
    cli()
