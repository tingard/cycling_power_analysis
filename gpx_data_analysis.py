import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


def value_check(res):
    if res is not None:
        return res.text
    return np.nan


def gpx_to_dataframe(fname):
    # would like speed but would need to calculate from dx/dt
    with open(fname) as f:
        s = f.read()
    soup = BeautifulSoup(s, features='lxml')
    trackpoints = soup.find_all('trkpt')
    time = np.zeros(len(trackpoints), dtype='datetime64[s]')
    power = np.zeros(len(trackpoints))
    elevation = np.zeros(len(trackpoints))
    cadence = np.zeros(len(trackpoints))
    hr = np.zeros(len(trackpoints))
    for i, t in enumerate(trackpoints):
        time[i] = value_check(t.find('time'))
        power[i] = value_check(t.find('power'))
        elevation[i] = value_check(t.find('ele'))
        hr[i] = value_check(t.find('gpxtpx:hr'))
        cadence[i] = value_check(t.find('gpxtpx:cad'))
    return pd.DataFrame(
        list(zip(time, power, cadence, hr)),
        columns=('time', 'power', 'cadence', 'hr'),
    )


if __name__ == '__main__':
    df = gpx_to_dataframe('Something_like_a_fitness_test.gpx')
    print(df.head())
