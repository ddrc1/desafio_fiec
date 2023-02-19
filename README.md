# desafio_fiec

### How to reproduce using MLflow:
mlflow run . -e main -P min_samples_split=4 min_samples_leaf=1 bootstrap=False max_features=0.25 n_estimators=200 criterion='friedman_mse' n_jobs=-1 </br>
It will run a Random Forest algoritm</br>
If you desire to use different parameters, feel free to change the values
