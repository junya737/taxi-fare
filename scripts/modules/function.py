import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
import optuna
import lightgbm as lgb


def standardize(train_data, test_data):
    """訓練データとテストデータを標準化，テストデータは訓練データを用いて標準化される

    Args:
        train_data (pd_dataframe): 訓練データ
        test_data (pd_dataframe): テストデータ
    Returns
        pd_dataframe: 標準化後のデータ
    """

    # 不偏分散で標準化 std()が不偏分散平方根であるため
    train_data_sc = (train_data - train_data.mean()) / train_data.std(ddof=1)
    test_data_sc = (test_data - train_data.mean()) / train_data.std(ddof=1)

    return train_data_sc, test_data_sc


def add_one_hot(df, column_name):
    """ Add one hot vector

    Args:
        df (pd dataframe):

    Returns:
        _type_: df
    """
    if (column_name in df.columns):
        one_hot = pd.get_dummies(df[column_name], prefix=column_name)
        df = pd.concat([df, one_hot], axis=1)
        df.pop(column_name)

    return df


def nan_to_mean(df):
    df = df.fillna(df.mean())
    return df


def save_kaggle_prediction(Id, prediction, Id_col_name="Id",
                           target_col_name="target"):
    # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
    my_solution = pd.DataFrame(prediction, Id, columns=[target_col_name])
    dt_now = datetime.datetime.now()
    # my_tree_one.csvとして書き出し
    my_solution.to_csv("../data/result/prediction_" +
                       dt_now.strftime('%Y%m%d_%H%M%S') + ".csv",
                       index_label=[Id_col_name])


def get_score_StratifiedKFold_cv(clf, x, y, n_splits, scoring, shuffle=True):
    """層化抽出KfoldCVのスコアを取得

    Args:
        clf (class): model
        x (np_array): fitting
        y (np_array): answer
        n_splits (int): num of split
        scoring (str): the wey of scoring
        shuffle (bool): Defaults to True.

    Returns:
        float: score
    """
    kf = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=42)
    result = cross_validate(clf, x, y, cv=kf, scoring=scoring)

    return np.mean(result['test_score'])


def cross_val_score_lgbm_earlystopping(clf, x, y, cv, stopping_rounds=50,
                                       scoring="accuracy",
                                       eval_metric="logloss"):
    """Get cross validation score using LightGBM with early stopping

    Args:
        clf (class): model
        x (np_array): features
        y (np_array): labels
        cv (class): cross val
        stopping_rounds (int, optional): _description_. Defaults to 50.
        scoring (str, optional): score metric. Defaults to "accuracy".
        eval_metric (str, optional): metric for early stopping.
            Defaults to "logloss".

    Returns:
        float: score
    """

    # クロスバリデーションのデータ分割
    scores = []
    for _, (train_idx, val_idx) in enumerate(cv.split(x, y)):
        x_train = x[train_idx]
        x_val = x[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        verbose_eval = 0  # この数字を1にすると学習時のスコア推移がコマンドライン表示される
        clf.fit(x_train, y_train,
                # early_stoppingの評価指標(学習用の'metric'パラメータにも同じ指標が自動入力される)
                eval_metric=eval_metric,
                eval_set=[(x_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds,
                                              verbose=False),
                           # early_stopping用コールバック関数
                           lgb.log_evaluation(verbose_eval)]
                # コマンドライン出力用コールバック関数
                )

        y_pred = clf.predict(x_val)
        score = accuracy_score(y_true=y_val, y_pred=y_pred)
        scores.append(score)

    return np.mean(scores)


def lgbm_bayesian_opt(x, y, cv, params, n_trials=50, config="study"):
    """ Optimaize hypara of lgbm using bayesian optimization

    Args:
        x (_type_): _description_
        y (_type_): _description_
        cv (_type_): _description_
        params (_type_): _description_
        n_trials (int, optional): _description_. Defaults to 50.
        config (str, optional): if print, print logs. Defaults to "study".

    Returns:
        class: study instance
    """

    # 最適化する関数を用意　
    def objective(trial):
        # trial型 次に試すパラメータをsuggestするメソッドを持つ suggest_category, intなどもある
        lgbm_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'num_leaves': trial.suggest_int(
                "num_leaves", params["num_leaves"][0],
                params["num_leaves"][1]),
            "max_depth": trial.suggest_int(
                "max_depth", params["max_depth"][0],
                params["max_depth"][1]),
            'learning_rate': trial.suggest_float(
                "learning_rate", params["learning_rate"][0],
                params["learning_rate"][1]),
            'random_state': 42,
            "class_weight": "balanced"
        }
        clf = lgb.LGBMClassifier(**lgbm_params)
        score = cross_val_score_lgbm_earlystopping(clf, x, y, cv)

        return score

    # ログを非表示
    optuna.logging.disable_default_handler()
    # 次のトライアルのパラメータ選択方法
    sampler = optuna.samplers.TPESampler(seed=0)
    # study型　
    study = optuna.create_study(sampler=sampler, direction="maximize")
    # 最適化
    study.optimize(objective, n_trials=n_trials)

    # print 設定なら結果を表示
    if (config == "print"):
        # 探索後の最良値
        print("trial:", study.best_trial.number+1)
        print(study.best_value)
        print(study.best_params)
        # 探索の履歴
        for trial in study.get_trials():
            print(trial.number+1, ":", trial.value, trial.params)

    # そうでないならstudyを返す
    else:
        return study


def cv_lgbm_bayesian_opt(x, y, cv_cv, cv_opt, params, n_trials):
    """Cross validation and bayesian opt

    Args:
        x (_type_): _description_
        y (_type_): _description_
        cv_cv (_type_): cv for cv
        cv_opt (_type_): cv for opt
        params (dict): hypara support
        n_trials (int): num of trials in opt

    Returns:
        list: list of study
    """
    study_list = []
    for _, (train_idx, _) in enumerate(cv_cv.split(x, y)):
        x_train = x[train_idx]
        y_train = y[train_idx]
        # bayesian opt
        study_list.append(lgbm_bayesian_opt(x_train, y_train, cv_opt,
                                            params, n_trials))

    return study_list


def get_now_time():
    """return now time as string

    Returns:
        _type_: _description_
    """
    dt_now = datetime.datetime.now()
    return dt_now.strftime('%Y%m%d_%H%M%S')


def lgbm_fit_earlystopping(clf, x_train, y_train, x_val, y_val,
                           eval_metric, stopping_rounds):
    """fitting lgbm with earlystopping

    Args:
        clf (class): _description_
        x_train (_type_): _description_
        y_train (_type_): _description_
        x_val (_type_): val data for early stopping
        y_val (_type_): _description_
        eval_metric (str): _description_
        stopping_rounds (int): _description_

    Returns:
        class: model after fitting
    """
    clf.fit(x_train, y_train,
            eval_metric=eval_metric, eval_set=[(x_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds,
                                          verbose=False),
                       lgb.log_evaluation(0)])
    return clf


def set_lgbm_params(params):
    """lgbmにハイパラをセットする

    Args:
        params (list): 
    Returns:
        class: _description_
    """

    lgbm_params = {
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'num_leaves': params["num_leaves"],
        "max_depth": params["max_depth"],
        'learning_rate': params["learning_rate"],
        'n_estimators': 10000,
        'random_state': 42,
    }

    return lgb.LGBMRegressor(**lgbm_params)


def cv_lgbm_clf(x, y, cv_cv, params_list, eval_metric, stopping_rounds):
    """lgbmのclfリストを取得

    Args:
        x (_type_): _description_
        y (_type_): _description_
        cv_cv (_type_): _description_
        params_list (_type_): _description_
        eval_metric (_type_): _description_
        stopping_rounds (_type_): _description_

    Returns:
        list: _description_
    """

    clf_list = []
    for i, (train_idx, test_idx) in enumerate(cv_cv.split(x, y)):
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        params = params_list[i]
        clf = set_lgbm_params(params)
        clf = lgbm_fit_earlystopping(
            clf, x_train, y_train, x_test, y_test, eval_metric,
            stopping_rounds)

        # ハイパラの確認
        for key in ["num_leaves", "max_depth", "learning_rate"]:
            assert clf.get_params()[key] == params[key], "ハイパラの値が一致していません"
        clf_list.append(clf)

    return clf_list


def drop_many_nan_column(df, threshold):
    """欠損値がある割合以上の列を削除する
    """
    return df.dropna(thresh=len(df)*threshold, axis=1)
