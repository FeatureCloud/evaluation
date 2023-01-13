import warnings
from time import sleep
import FeatureCloud.api.cli.test.commands as fc
import pandas as pd
import os
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

warnings.simplefilter(action='ignore', category=FutureWarning)


def run_workflow(app_images: [str], data_dir: str, generic_dir: str, results_path: str,
                 controller_host: str = 'http://localhost:8000', channel: str = 'local', query_interval: int = 1):
    print(f'Run workflow {app_images}')
    for app_image in app_images:
        test_id = run_app(app_image, results_path, controller_host, channel, query_interval, data_dir, generic_dir)
        sleep(5)
        instances = check_finished(test_id, controller_host)
        results_path = f'{results_path}/{app_image.split("/")[1]}'
        sleep(5)
        move_results(results_path, instances, test_id, data_dir)
        print('')
        sleep(1)
    print(f'Workflow run {app_images} finished.')


def run_app(app_image: str, input_dir: str, controller_host: str, channel: str, query_interval: int, data_dir: str,
            generic_dir: str):
    print(f'Run app {app_image}')
    client_dirs = []
    for it in os.scandir(input_dir):
        if it.is_dir():
            str(it.path)
            if "client_" in str(it.path).split("/")[-1]:
                client_dirs.append(str(it.path).replace(data_dir, '')[1:])
    client_dirs = ",".join(client_dirs)
    test_id = fc.start(controller_host=controller_host, client_dirs=client_dirs,
                       generic_dir=generic_dir, app_image=app_image, channel=channel, query_interval=query_interval,
                       download_results="./")

    return test_id


retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    backoff_factor=1
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)


def get_tests(url: str):
    response = http.get(url=f'{url}/app/tests/')

    if response.status_code == 200:
        return True, response.json()
    else:
        return False, response.json()


def get_test(url: str, test_id: str or int):
    response = http.get(url=f'{url}/app/test/{test_id}/')

    if response.status_code == 200:
        return True, response.json()
    else:
        return False, response.json()


def check_finished(test_id: str or int, controller_host: str):
    finished = False
    df = None
    while not finished:
        df = fc.info(test_id=test_id, format='dataframe', controller_host=controller_host)
        if df is not None:
            status = df.loc[test_id, 'status']
            if status == 'finished' or status == 'error' or status == 'stopped':
                finished = True
                print(f'Test {test_id} {status}')
        else:
            print('Test not started yet...')

        sleep(5)

    return df.loc[test_id, 'instances']


def move_results(results_path: str, instances: pd.DataFrame, test_id: str or int, data_dir: str):
    print('Move results...')
    os.makedirs(results_path, exist_ok=True)
    for instance in instances:
        client_dir = results_path + f'/client_{instance["id"] + 1}'
        filename = f'results_test_{test_id}_client_{instance["id"]}_{instance["name"]}'
        os.makedirs(client_dir, exist_ok=True)
        filepath = data_dir + "/tests/" + filename + ".zip"
        while not os.path.exists(filepath):
            sleep(5)
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(client_dir)
        sleep(5)
        while not os.path.exists(client_dir):
            sleep(5)
        os.remove(data_dir + "/tests/" + filename + ".zip")

# all images need to be pulled first and exist locally
app_images = ['featurecloud.ai/fc_logistic_regression']
# data directory of the controller
data_dir = '/Users/spaethju/GolandProjects/fc-controller-go/data'
# contains the config file for all clients
generic_dir = f'tests/fc_paper/scalability/generic/'

# this example runs the analysis on the data in data_dir/dataset/n_clients
for clients_n in [1, 5, 10, 15, 20, 25, 30]:
    for n_iterations in [1, 5, 10]:
        for dataset in ['scalability']:
            print(f'Run dataset {dataset} with {n_iterations} iterations and {clients_n} clients')
            input_path = f'{data_dir}/tests/fc_paper/{dataset}/{clients_n}_clients/{n_iterations}_iterations'
            run_workflow(app_images, data_dir, generic_dir, input_path, 'http://localhost:8002')
            sleep(1)
            print(f'Finished run dataset {dataset}')
        sleep(5)
print("Workflow Completed.")
