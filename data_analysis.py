import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from scipy.signal import boxcar, convolve
import requests
from multiprocessing import Pool
from io import BytesIO
from PIL import Image
import json
from scipy.interpolate import splprep, splev


# -------------- SECTION: File handling and data loading --------------
def load_file(file_path, **kwargs):
    file_type = file_path.split('.')[-1]
    with open(file_path) as f:
        data = f.read()
    soup = BeautifulSoup(data, 'xml')
    trackpoint_key = ('trkpt' if file_type == 'gpx' else 'Trackpoint')
    trackpoints = soup.find_all(trackpoint_key)
    extracted = pd.DataFrame([parse_point(t) for t in trackpoints])
    return type_check(extracted).set_index('Time')

# (attr name, gpx name) pairs
TAGS = (
    ('HeartRateBpm', 'hr'),
    ('DistanceMeters', 'distance'),
    ('Speed', 'speed'),
    ('Cadence', 'cad'),
    ('Watts', 'power'),
    ('Time', 'time'),
    ('Elevation', 'ele'),
    ('Temperature', 'atemp'),
    ('Latitude', 'lat'),
    ('Longitude', 'lon'),
)
ALLOWED_TAGS = {j for i in TAGS for j in i}
GPX_MAP = {i[1]: i[0] for i in TAGS}


def parse_point(point):
    return {
        **{GPX_MAP.get(k, k): v for k, v in point.attrs.items() if k in ALLOWED_TAGS},
        **{
            GPX_MAP.get(i.name, i.name): i.string
            for i in point.descendants
            if i.name in ALLOWED_TAGS
        },
    }


def parse_gpx_trackpoint(point):
    if type(point) == str:
        point_ = BeautifulSoup(point, 'xml')
    else:
        point_ = point
    return {
        'Latitude': point_.attrs.get('lat', np.nan),
        'Longitude': point_.attrs.get('lon', np.nan),
        **{
            name: point_.find(key).text
            for name, key in TAGS
            if point_.find(key) is not None
        }
    }


def parse_tcx_trackpoint(point):
    if type(point) == str:
        point_ = BeautifulSoup(point, 'xml')
    else:
        point_ = point
    return {
        t[0]: point_.find(t[0]).text
        for t in TAGS
        if point_.find(t[0]) is not None
    }


FLOAT_KEY_LIST = ('DistanceMeters', 'Speed', 'Cadence', 'Watts', 'Temperature',
                  'Elevation', 'HeartRateBpm', 'Latitude', 'Longitude')


def type_check(df):
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%dT%H:%M:%SZ')
    for k in FLOAT_KEY_LIST:
        try:
            df[k] = df[k].apply(float)
        except KeyError:
            pass
    return df


# -------------- SECTION: Data cleaning and manipulation --------------
def interpolate_to_second(df):
    # make an array of all times between the start and end point
    all_times = np.arange(
        df.index.values[0],
        df.index.values[-1] + 1,
        dtype='datetime64[s]',
    )
    # reindex and perform linear interpolation
    return df.reindex(all_times).interpolate('linear')


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # Radius of earth in kilometers is 6371
    m = 6371000 * c
    return m


def get_distance(df, from_speed=False):
    if from_speed:
        dt = df.index.to_series().diff()\
            .astype('timedelta64[s]').values.astype('f8')
        dist_metres = (df['Speed'] * dt).fillna(0.0).cumsum()
        return dist_metres / 1000

    return np.concatenate((
        [0],
        np.cumsum(
            haversine_distance(
                df['Longitude'].values[:-1],
                df['Latitude'].values[:-1],
                df['Longitude'].values[1:],
                df['Latitude'].values[1:],
            )
        )
    ))


def get_speed(df):
    if 'DistanceMeters' not in df.columns:
        df['DistanceMeters'] = get_distance(df)
    seconds_from_start = (
        (df.index - df.index[0]) / np.timedelta64(1, 's')
    ).values
    speed = np.gradient(
        df['DistanceMeters'] / 1000,
        seconds_from_start / 3600
    )
    return pd.Series(speed, index=df.index)


def get_acceleration(df):
    if 'Speed' not in df.columns:
        df['Speed'] = get_speed(df)
    seconds_from_start = (
        (df.index - df.index[0]) / np.timedelta64(1, 's')
    ).values
    acc = np.gradient(
        df['Speed'] / 1000,
        seconds_from_start / 3600
    )
    return pd.Series(acc, index=df.index)


def get_gradient(df):
    grad_df = pd.DataFrame({
        'x': df['DistanceMeters'],
        'dx': df['DistanceMeters'].diff(),
        'y': df['Elevation'].rolling(100).mean(),
        'dy': df['Elevation'].rolling(100).mean().diff(),
    })
    grad_df = grad_df.dropna().query('dx > 0.01')
    gradient = np.rad2deg(np.arctan2(grad_df['dy'], grad_df['dx']))
    return gradient.reindex(df.index).interpolate('linear')


def get_ns_power(df, n=3):
    if 'Watts' not in df.columns:
        print('No "Watts" column in DataFrame')
        return np.zeros(df.index.values.shape)
    return convolve(df['Watts'], boxcar(n), 'same') / n


def encode_geojson(df, n=200):
    coords = df[['Longitude', 'Latitude']].values
    mask = np.concatenate((
        np.ones(1, dtype=bool),
        np.logical_and.reduce(coords[1:] != coords[:-1], axis=1)
    ))
    scaling = {
        'mean': coords[mask].mean(axis=0),
        'std': coords[mask].std(axis=0)
    }
    tck, u = splprep(
        ((coords[mask] - scaling['mean']) / scaling['std']).T,
        s=0
    )
    new_coords_scaled = np.array(splev(np.linspace(0, 1, n), tck)).T
    new_coords = np.round(
        new_coords_scaled * scaling['std'] + scaling['mean'],
        4
    )
    return json.dumps(dict(
        type='LineString',
        coordinates=new_coords.tolist()
    )).replace(' ', '')


def get_mapbox_image(token, **kwargs):
    params = dict(token=token, res_x=512, res_y=512)
    params.update(kwargs)
    url = (
        'https://api.mapbox.com/styles/v1/mapbox/streets-v11/static/'
        'geojson({overlay})/auto/'
        '{res_x}x{res_y}?access_token={token}'
    ).format(**params)
    response = requests.get(url)
    return Image.open(BytesIO(response.content))
