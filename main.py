import sys
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_dataset(x, y, window_size=1, periods_ahead=2):
    periods_ahead -= 1
    xs = []
    ys = []
    for i in range(len(x) - window_size - periods_ahead):
        v = x.iloc[i: i + window_size, 0].to_numpy()
        v = np.append(v, x.iloc[i + window_size, 1])#d_semana
        v = np.append(v, x.iloc[i + window_size, 2])#d_mes
        v = np.append(v, x.iloc[i + window_size, 3])#d_ano
        v = np.append(v, x.iloc[i + window_size, 4])#ano
        v = np.append(v, x.iloc[i + window_size, 5])#hora_d
        xs.append(v)
        ys.append(y.iloc[i + window_size + periods_ahead])
    return np.array(xs), np.array(ys)

if __name__ == '__main__':
    args = sys.argv[1:]
    path = args[0]
    min_samples_split = int(args[1])
    min_samples_leaf = int(args[2])
    bootstrap = bool(args[3])
    max_features = args[4]
    try:
        max_features = float(max_features)
    except:
        pass

    n_estimators = int(args[5])
    criterion = args[6]
    n_jobs = int(args[7])

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.resample('10T').mean().interpolate()
    df = df[5::6]
    df = df[['T (degC)']]
    df.reset_index(inplace=True)
    df.rename(columns={"Date Time": "date", "T (degC)": "y"}, inplace=True)

    df = df.query("date >= '2014-01-01 00:00:00'")
    df['d_semana'] = df['date'].dt.strftime("%u").astype(int)
    df['d_mes'] = df['date'].dt.strftime("%d").astype(int)
    df['d_ano'] = df['date'].dt.strftime("%j").astype(int)
    df['ano'] = df['date'].dt.strftime("%y").astype(int)
    df['hora_d'] = df['date'].dt.strftime("%H").astype(int)
    df.drop(columns=['date'], inplace=True)

    window_size = 365*24 # Um ano
    pred_size = 30*24  # Um mês
    train_size = len(df) - pred_size
    train = df[:train_size]
    test = df[train_size - window_size:]

    periods_ahead = 5
    x_train, y_train = create_dataset(train, train['y'], window_size, periods_ahead)
    x_test, y_test = create_dataset(test, test['y'], window_size, periods_ahead)

    model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split, max_features=max_features, bootstrap=bootstrap, n_jobs=n_jobs)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    plt.figure(figsize=(20, 6))
    plt.plot(range(len(preds)), preds, label="Predito")
    plt.plot(range(len(y_test)), y_test, label="Real")
    plt.legend()
    plt.savefig("prediction.png")
    mlflow.log_artifact("prediction.png")
    plt.clf()

    rmse = mean_squared_error(y_test, preds) ** (1/2)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f'RMSE={rmse}')
    print(f'MAE={mae}')
    print(f'R2={r2}')

    plt.scatter(y_test, preds)
    plt.xlabel('Real')
    plt.ylabel('Predito')
    plt.savefig("scatter.png")
    mlflow.log_artifact("scatter.png")

