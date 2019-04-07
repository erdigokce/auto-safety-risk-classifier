import requests
import config


def get_number_of_samples(n):
    resp = requests.get(config.application['BE_ROOT'] + '/auto/riskestimator/observations/top-{:d}/metric'.format(n))
    if resp.status_code != 200:
        # This means something went wrong.
        raise RuntimeError(
            'GET ' + config.application['APPLICATION_ROOT'] + '/auto/riskestimator/observations/top-{:d}/ {}'.format(n, resp.status_code))
    return resp.json()
