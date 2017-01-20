#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd


def get_data():
    """Loads data frame from file.

    Returns:
        DataFrame.
    """
    return pd.read_csv("data.csv")


def get_unignored_data():
    """Loads data frame from file and drops ignored data.

    Returns:
        DataFrame.
    """
    df = get_data()
    return df[df.ignore == False]


def filter_by_gender(df, male):
    """Filters the data by gender.

    Args:
        df: DataFrame.
        male: True if male, False otherwise.

    Returns:
        DataFrame.
    """
    gender = 1 if male else 0
    return df[df.male == gender]


if __name__ == "__main__":
    total = get_data().count().id
    unignored = get_unignored_data().count().id
    ignored = total - unignored

    print("found {} entries, ignored {}".format(total, ignored))
