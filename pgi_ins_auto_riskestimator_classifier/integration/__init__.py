import config
import requests


def get_number_of_samples(n):
    resp = requests.get(config.application['BE_ROOT'] + '/auto/riskestimator/observations/top-{:d}/metric'.format(n))
    if resp.status_code != 200:
        # This means something went wrong.
        raise RuntimeError('GET ' + config.application['APPLICATION_ROOT'] + '/auto/riskestimator/observations/top-{:d}/ {}'.format(n, resp.status_code))
    return resp.json()


def get_number_of_doors_map():
    resp = requests.get(config.application['BE_ROOT'] + '/auto/riskestimator/features/number_of_doors')
    if resp.status_code != 200:
        # This means something went wrong.
        raise RuntimeError('GET ' + config.application['APPLICATION_ROOT'] + '/auto/riskestimator/features/number_of_doors {}'.format(resp.status_code))
    return resp.json()


def get_number_of_cylinders_map():
    resp = requests.get(config.application['BE_ROOT'] + '/auto/riskestimator/features/number_of_cylinders')
    if resp.status_code != 200:
        # This means something went wrong.
        raise RuntimeError('GET ' + config.application['APPLICATION_ROOT'] + '/auto/riskestimator/features/number_of_cylinders {}'.format(resp.status_code))
    return resp.json()
