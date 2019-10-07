import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import optuna


class High_and_Low_Classifier:
    
    def __init__(self,train,use_col):
        self.use_col = use_col
        self.X = train.loc[:,self.use_col] 
        self.y = train.loc[:,'high_price_flag']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                        test_size=0.3, random_state=0)
        self.lgb_train = lgb.Dataset(self.X_train, self.y_train)
        self.lgb_test = lgb.Dataset(self.X_test,self.y_test)


    def train(self):
        def objective(trial):

            learning_rate = trial.suggest_uniform('learning_rate', 0, 1.0)
            num_leaves = trial.suggest_int('num_leaves', 10, 2**8)
            max_depth = trial.suggest_int('max_depth', 3, 8)

            lgbm_params = {
                'task': 'train',
                # "metrics": 'xentropy',
                'boosting_type': 'gbdt',
                'objective': 'binary',
                "learning_rate": learning_rate,
                "num_leaves": num_leaves,
                "max_depth": max_depth,
                "n_jobs": 1,
                'verbose': -1,
                "seed": 0
            }

            # cv_results = lgb.cv(lgbm_params, self.lgb_train, nfold=5, stratified=False)
            # print(cv_results['xentropy-mean'])
            # print(len(cv_results['xentropy-mean']))
            # score = np.array(cv_results['xentropy-mean']).mean()
            mdl = lgb.LGBMClassifier(**lgbm_params)
            stratifiedkfold = StratifiedKFold(n_splits=3)
            scores = cross_val_score(mdl,self.X_train,self.y_train,cv=stratifiedkfold,scoring='neg_log_loss')
            score = np.mean(scores)

            return score

        study = optuna.create_study()
        study.optimize(objective,n_trials=10)

        mdl = lgb.LGBMClassifier(**study.best_params)
        mdl.fit(self.X_train,self.y_train)

        pred_train = mdl.predict_proba(self.X_train)[:,1]
        pred_train = [1 if i>0.5 else 0 for i in pred_train]

        pred_test = mdl.predict_proba(self.X_test)[:,1]
        pred_test = [1 if i>0.5 else 0 for i in pred_test]

        train_accuracy = accuracy_score(self.y_train,pred_train)
        test_accuracy = accuracy_score(self.y_test,pred_test)

        print('---------------------------------')
        print('train score: ',train_accuracy)
        print('test score: ',test_accuracy)
        print('---------------------------------')

        # self.lgb_train = lgb.Dataset(self.X, self.y)
        # mdl = lgb.train(study.best_params, self.lgb_train)
        mdl = lgb.LGBMClassifier(**study.best_params)
        mdl.fit(self.X,self.y)

        self.trained_mdl = mdl

    def pred(self,test):
        self.test = test.loc[:,self.use_col]
        mdl = self.trained_mdl
        self.test_pred = mdl.predict_proba(self.test)[:,1]
        print(self.test_pred)

    def labeling(self):
        labels = [1 if i>0.5 else 0 for i in self.test_pred]
        return self.test_pred,labels

if __name__ == "__main__":
    train = pd.read_csv('processed_data/train_v10.csv')
    test = pd.read_csv('processed_data/test_v10.csv')

    use_col = train.columns
    un_use_col = ['id','y','log_y','high_price_flag','location', 'access', 'layout', 'age', 'direction', 'area','floor', 'bath_toilet', 'kitchen',
                 'broadcast_com', 'facilities','parking', 'enviroment', 'structure', 'contract_period',
                 'walk_time','23ku',
                #  'area_num_countall','floor_countall','room_num_countall','facilities_countall','age_countall','area_num_countall',
                ]
    use_col = [c for c in use_col if c not in un_use_col]

    print(use_col)

    clf = High_and_Low_Classifier(train,use_col)
    clf.train()
    clf.pred(test)

    prob, label = clf.labeling()

    print(sum(label))
