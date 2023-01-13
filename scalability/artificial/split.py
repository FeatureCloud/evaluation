from sklearn.datasets import make_classification
import pandas as pd
from pathlib import Path
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

random_state = 42


for clients_n in [1, 5, 10, 15, 20, 25, 30]:
    X, y = make_classification(n_samples=clients_n * 1000, n_features=100, n_informative=95, n_redundant=5,
                               n_repeated=0, n_classes=2, shuffle=True, random_state=random_state)
    y = pd.Series(y)
    X = pd.DataFrame(X)
    scaler = StandardScaler()
    scaler.fit(X)
    data = pd.DataFrame(scaler.transform(X))
    data['y'] = y
    clients = np.array_split(data, clients_n)
    for n_client in range(clients_n):
        train, test = train_test_split(clients[n_client], test_size=0.2, shuffle=True, random_state=random_state)
        for n in [1, 5, 10]:
            Path(f'{clients_n}_clients/{n}_iterations/client_{n_client + 1}').mkdir(parents=True, exist_ok=True)
            train.to_csv(f'{clients_n}_clients/{n}_iterations/client_{n_client + 1}/train.csv', index=False)
            test.to_csv(f'{clients_n}_clients/{n}_iterations/client_{n_client + 1}/test.csv', index=False)
            shutil.copyfile(f'config{n}.yml', f'{clients_n}_clients/{n}_iterations/client_{n_client + 1}/config.yml')
