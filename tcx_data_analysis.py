import xml.etree.ElementTree as ET
import re
import numpy as np
import pandas as pd
from scipy.signal import boxcar
from scipy.signal import convolve
from bs4 import BeautifulSoup


def strip_tag(t):
    return re.search(
        r'v2}(.*?)\'',
        str(t)
    ).group(1)


def tcx_to_dataframe(tcx_path):
    with open(tcx_path) as f:
        xml = f.read()
    soup = BeautifulSoup(xml, features="lxml")
    trackpoints = soup.find_all('trackpoint')
    time = np.fromiter(((i.find('time').text) for i in trackpoints), dtype='datetime64[s]')
    power = np.fromiter(((i.find('watts').text) for i in trackpoints), dtype=float)
    cadence = np.fromiter(((i.find('cadence').text) for i in trackpoints), dtype=float)
    speed = np.fromiter(((i.find('speed').text) for i in trackpoints), dtype=float)
    hr = np.fromiter(((i.find('heartratebpm').text) for i in trackpoints), dtype=float)
    distance = np.fromiter(((i.find('distancemeters').text) for i in trackpoints), dtype=float)
    return pd.DataFrame(
        {'time': time, 'power': power, 'cadence': cadence,
         'speed': speed, 'hr': hr, 'distance': distance}
    )


# Data analysis methods
def group_by_minute(df):
    df2 = df.drop(['distance'], axis=1)
    df2['time'] = df['time'].values.astype('datetime64[m]')
    return df2.groupby('time').mean()


def get_power_curve(power, min_n=1, max_n=60):
    n_minute_power = []
    for n in range(min_n, max_n + 1):
        n_minute_power.append(
            np.max(convolve(power, boxcar(n), 'valid') / n)
        )
    return np.array(n_minute_power)
