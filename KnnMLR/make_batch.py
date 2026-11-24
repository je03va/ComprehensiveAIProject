"""Simulates the environment: writes a batch of new transcoding jobs for the agent to pick up.

  python make_batch.py            # 5 normal jobs + 1 noisy job + 1 unseen-codec job
  python make_batch.py --finished # same, plus the real utime of the normal jobs (feedback)
"""
import argparse

import pandas as pd

from common import load_data


def make_batch(finished=False):
    """Return a batch of 7 jobs; with finished=True the normal jobs carry their measured utime."""
    X, y = load_data()
    batch = X.sample(5, random_state=None).astype(object)
    utime = list(y.loc[batch.index])

    noisy = batch.iloc[0].copy()
    noisy['duration'], noisy['o_width'] = 'abc', 9999          # not a number, and far too wide
    unseen = batch.iloc[1].copy()
    unseen['codec'] = 'av1'                                     # codec never seen in training

    batch = pd.concat([batch, noisy.to_frame().T, unseen.to_frame().T], ignore_index=True)
    if finished:
        batch['utime'] = utime + [None, None]                   # only the normal jobs have finished
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--finished', action='store_true', help='include measured utime so the agent learns')
    args = parser.parse_args()

    batch = make_batch(args.finished)
    batch.to_csv('transcode_batch.csv', index=False)
    print(f'Wrote transcode_batch.csv with {len(batch)} jobs' + (' (with measured utime)' if args.finished else ''))
