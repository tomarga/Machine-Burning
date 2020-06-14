def choose_features():
    import pandas as pd

    
    print("1. accuracy ")
    print("2. roc_auc_ovo ")
    print("3. f1_micro ")
    scoring = input("Koje značajke želim s obzirom na scoring parametar u RFECV-u? ")
    print("")

    if scoring == "1":
        
        print("1. univarijantna metoda + rfecv ")
        print("2. samo rfecv ")
        print("3. univarijantna metoda + izbacivanje koreliranih značajki + rfecv ")
        print("4. izbacivanje koreliranih značajki + rfecv ")
        option = input("Koje značajke želim s obzirom na korištene feature selection tehnike? ")
        print("")

        if option == "1":
            # UNI + RFECV
            # load features from train and test dataset given using rfecv
            selected_features_train = pd.read_csv("https://raw.githubusercontent.com/tomarga/Machine-Burning/master/dataset/selected%20features/ACC/uni/features_train_UNI_and_RFECV_rfecv_rf_accuracy_cv5_nestim200.csv")
            selected_features_test = pd.read_csv("https://raw.githubusercontent.com/tomarga/Machine-Burning/master/dataset/selected%20features/ACC/uni/features_test_UNI_and_RFECV_rfecv_rf_accuracy_cv5_nestim200.csv")

            # load table with feature names and their scores, sorted
            feature_importances = pd.read_csv("https://raw.githubusercontent.com/tomarga/Machine-Burning/master/dataset/selected%20features/ACC/uni/UNI_and_RFECV_rfecv_rf_accuracy_cv5_nestim200_feature_selected_with_names_and_importance_scores_sorted.csv")
            return [feature_importances, selected_features_train, selected_features_test]

        elif option == "2":
            # RFECV
            # load features from train and test dataset given using rfecv
            selected_features_train = pd.read_csv("https://raw.githubusercontent.com/tomarga/Machine-Burning/master/dataset/selected%20features/ACC/samo%20rfecv/features_train_SAMORFECV_rfecv_rf_accuracy_cv5_nestim200.csv")
            selected_features_test = pd.read_csv("https://raw.githubusercontent.com/tomarga/Machine-Burning/master/dataset/selected%20features/ACC/samo%20rfecv/features_test_SAMORFECV_rfecv_rf_accuracy_cv5_nestim200.csv")

            # load table with feature names and their scores, sorted
            feature_importances = pd.read_csv("https://raw.githubusercontent.com/tomarga/Machine-Burning/master/dataset/selected%20features/ACC/samo%20rfecv/SAMORFECV_rfecv_rf_accuracy_cv5_nestim200_feature_selected_with_names_and_importance_scores_sorted.csv")
            return [feature_importances, selected_features_train, selected_features_test]

        elif option == "3":
            # UNI + COR + RFECV
            # load features from train and test dataset given using rfecv
            selected_features_train = pd.read_csv("https://raw.githubusercontent.com/tomarga/Machine-Burning/master/dataset/selected%20features/ACC/uni%2Bcor/features_train_uni_and_cor_rfecv_rf_accuracy_cv5_nestim200.csv")
            selected_features_test = pd.read_csv("https://raw.githubusercontent.com/tomarga/Machine-Burning/master/dataset/selected%20features/ACC/uni%2Bcor/features_test_uni_and_cor_rfecv_rf_accuracy_cv5_nestim200.csv")

            # load table with feature names and their scores, sorted
            feature_importances = pd.read_csv("https://raw.githubusercontent.com/tomarga/Machine-Burning/master/dataset/selected%20features/ACC/uni%2Bcor/uni_and_cor_rfecv_rf_accuracy_cv5_nestim200_feature_selected_with_names_and_importance_scores_sorted.csv")
            return [feature_importances, selected_features_train, selected_features_test]

        elif option == "4":
            # COR + RFECV
            # load features from train and test dataset given using rfecv
            selected_features_train = pd.read_csv("https://raw.githubusercontent.com/tomarga/Machine-Burning/master/dataset/selected%20features/ACC/cor/features_train_cor_rfecv_rf_accuracy_cv5_nestim200.csv")
            selected_features_test = pd.read_csv("https://raw.githubusercontent.com/tomarga/Machine-Burning/master/dataset/selected%20features/ACC/cor/features_test_cor_rfecv_rf_accuracy_cv5_nestim200.csv")

            # load table with feature names and their scores, sorted
            feature_importances = pd.read_csv("https://raw.githubusercontent.com/tomarga/Machine-Burning/master/dataset/selected%20features/ACC/cor/cor_rfecv_rf_accuracy_cv5_nestim200_feature_selected_with_names_and_importance_scores_sorted.csv")
            return [feature_importances, selected_features_train, selected_features_test]

        else:
            print('Ne postoji opcija ' + option +'!')
            raise ValueError("Niste upisali valjani broj! Upišite ili 1 ili 2 ili 3 ili 4! ")
            return
    elif scoring == "2":
        raise ValueError("Podaci još nisu pripremljeni")
        return
    elif scoring == "3":
        raise ValueError("Podaci još nisu pripremljeni")
        return
    else:
        print('Ne postoji opcija ' + scoring +'!')
        raise ValueError("Niste upisali valjani broj! Upišite ili 1 ili 2 ili 3! ")
        return


def choose_scoring():
    print("1. accuracy")
    print("2. neg_log_loss")
    print("3. f1_micro")
    print("4. roc_auc_ovo")
    scoring = input("Unesi vrstu točnosti: ")

    if scoring == "1":
        return "accuracy"
    elif scoring == "2":
        return "neg_log_loss"
    elif scoring == "3":
        return "f1_micro"
    elif scoring == "4":
        return "roc_auc_ovo"
    else:
        raise ValueError('Odaberi jednu od ponuđenih opcija!')
        return

# randomly search for hyperparameters
def RandomizedSearchCV_with_SMOTE(name, data, labels, random_grid, sampling_strategy, k_neighbors, cv=5, scoring="accuracy", n_iter=20, random_state=42):
    import xgboost as xgb
    from xgboost import XGBClassifier
    from sklearn.model_selection import RandomizedSearchCV
    import pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE

    # we use the RandomizedSearchCV to find the best parameters for our XGB model

    # getting ready for saving later
    filename = 'RandomizedSearchCV_' + name + '.sav'

    model_XGB = XGBClassifier(random_state=random_state)

    model = Pipeline([
            ('oversample', SMOTE(random_state=random_state, n_jobs=-1, sampling_strategy=sampling_strategy, k_neighbors=k_neighbors)),
            ('classification', model_XGB)
        ])
    
    # RandomizedSearchCV
    rand_XGB = RandomizedSearchCV(model_XGB, 
                                  param_distributions = random_grid, 
                                  cv=StratifiedKFold(n_splits=cv), 
                                  scoring=scoring, 
                                  n_iter=20, 
                                  random_state=random_state, 
                                  return_train_score=False, 
                                  verbose=True,
                                  n_jobs=-1)
    # fit
    rand_XGB.fit(data, labels)

    try:
        # save 
        pickle.dump(rand_XGB, open(filename, 'wb'))
        print('RandomizedSearchCV je spremljen u: ' + filename )
    except:
      print("RandomizedSearchCV nije spremljen!")

    # show results
    rand_XGB_results_df = pd.DataFrame(rand_XGB.cv_results_)[['mean_test_score', 'std_test_score', 'params', 'rank_test_score']]
    rand_XGB_results_df

    # plot of randomized search results
    rand_XGB_mean_scores = rand_XGB.cv_results_['mean_test_score']
    plt.plot(list(range(1, 21)), rand_XGB_mean_scores)
    plt.xlabel('k-ti Model Randomized Search CV treniranja (XGB)')
    plt.ylabel('Točnost unakrsne validacije')

    return [rand_XGB, rand_XGB_results_df]


#grid search in space of given parameters
def GridSearchCV_for_SMOTE(name, param_grid, data, labels, cv=5, random_state=42):
    import xgboost as xgb
    from xgboost import XGBClassifier
    import pickle
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE

    # we use the GridSearchCV to find even better parameters for our XGB model

    m_XGB = XGBClassifier(random_state=random_state)

    model = Pipeline([
        ('oversample', SMOTE(random_state=random_state, n_jobs=-1)),
        ('classification', m_XGB )
    ])

    # for saving later
    filename = 'GridSearchCV_for_SMOTE_' + name + '.sav'

    # grid search
    grid_search = GridSearchCV(m_XGB, param_grid=param_grid, cv=StratifiedKFold(n_splits=cv), verbose=True, n_jobs=-1)
    grid_search.fit(data, labels)

    try:
        # save 
        pickle.dump(grid_search, open(filename, 'wb'))
        print("GridSearchCV je spremljen u: " + filename )
    except:
        print("GridSearchCV nije spremljen!")


    # show results
    grid_search_results_df = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params', 'rank_test_score']]
    grid_search_results_df

    return [grid_search, grid_search_results_df]


#grid search in space of given parameters
def GridSearchCV_with_SMOTE(name, param_grid, sampling_strategy, k_neighbors, data, labels, cv=5, random_state=42):
    import xgboost as xgb
    from xgboost import XGBClassifier
    import pickle
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE

    # we use the GridSearchCV to find even better parameters for our XGB model

    m_XGB = XGBClassifier(random_state=random_state)

    model = Pipeline([
        ('oversample', SMOTE(random_state=random_state, n_jobs=-1)),
        ('classification', m_XGB )
    ])

    # for saving later
    filename = 'GridSearchCV_with_SMOTE_' + name + '.sav'

    # grid search
    grid_search = GridSearchCV(m_XGB, param_grid=param_grid, cv=StratifiedKFold(n_splits=cv), verbose=True, n_jobs=-1)
    grid_search.fit(data, labels)

    try:
        # save 
        pickle.dump(grid_search, open(filename, 'wb'))
        print("GridSearchCV je spremljen u: " + filename )
    except:
        print("GridSearchCV nije spremljen!")


    # show results
    grid_search_results_df = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params', 'rank_test_score']]
    grid_search_results_df

    return [grid_search, grid_search_results_df]



# make model with specific parameters
def XGBClassifier_with_SMOTE(name, grid_or_random_search, sampling_strategy, k_neighbors, X_train, y_train, X_test, y_test, early_stopping_rounds=20, eval_metric=["merror", "mlogloss"], random_state=42):
    import xgboost as xgb
    from xgboost import XGBClassifier
    import pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    from collections import Counter

    # smote resample
    X_train, y_train = SMOTE(sampling_strategy=sampling_strategy, 
                             random_state=random_state, 
                             k_neighbors=k_neighbors).fit_resample(X_train, y_train)

    print(sorted(Counter(y_train).items()))

    # we use the RandomizedSearchCV to find the best parameters for our XGB model
    eval_set = [(X_train, y_train), (X_test, y_test)]

    # for saving later
    filename = 'XGBClassifier_' + name + '.sav'

    # making a model of best parameters
    param_tuning_xgb = XGBClassifier(reg_lambda        = grid_or_random_search.best_params_['reg_lambda'],
                                     reg_alpha         = grid_or_random_search.best_params_['reg_alpha'],
                                     n_estimators      = grid_or_random_search.best_params_['n_estimators'],
                                     min_samples_split = grid_or_random_search.best_params_['min_samples_split'],
                                     min_samples_leaf  = grid_or_random_search.best_params_['min_samples_leaf'],
                                     max_features      = grid_or_random_search.best_params_['max_features'],
                                     max_depth         = grid_or_random_search.best_params_['max_depth'],
                                     learning_rate     = grid_or_random_search.best_params_['learning_rate'],
                                     gamma             = grid_or_random_search.best_params_['gamma'],
                                     bootstrap         = grid_or_random_search.best_params_['bootstrap'])

    # fit
    param_tuning_xgb.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric, eval_set=eval_set, verbose=True)

    try:
        # save 
        pickle.dump(param_tuning_xgb, open(filename, 'wb'))
        print("XGBClassifier je spremljen u: " + filename )
    except:
        print("XGBClassifier nije spremljen!")

    # show results
    param_tuning_xgb_results_df = model_results(param_tuning_xgb, X_test, y_test)
    param_tuning_xgb_results_df

    return [param_tuning_xgb, param_tuning_xgb_results_df]



def pca_plot(data, labels):
    # packages
    import numpy as np
    import pandas as pd
    import seaborn as sn
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.mplot3d import Axes3D

    import sklearn as sk
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
        
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)


    #Standardize the Data
    #PCA is effected by scale so you need to scale the features in your data before applying PCA. 
    #Use StandardScaler to help you standardize the dataset’s features onto unit scale (mean = 0 and variance = 1) 
    #which is a requirement for the optimal performance of many machine learning algorithms. 
    #If you want to see the negative effect not scaling your data can have, scikit-learn has a section 
    #on the effects of not standardizing your data.
    #

    # TO 3D
    scaler_3 = StandardScaler()
    X_3 = data.copy()
    X_scaled_3 = scaler_3.fit_transform(X_3)

    pca_3 = PCA(n_components=3) 
    X_pca_3 = pca_3.fit_transform(X_scaled_3) 


    fig = plt.figure(1, figsize=(50,50))
    
    ax = plt.axes(projection='3d')
    ax.scatter(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2], c = labels, cmap=plt.cm.nipy_spectral, linewidth=0.5)

