# print model results
def model_results(model, test_data, test_lebel):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from tabulate import tabulate
    from sklearn import metrics
    
    table = pd.DataFrame(columns=["metrika", "uspješnost"])
    print('Uspješnost modela:')

    proba           = model.predict_proba(test_data)
    predicted_label = np.asarray(model.predict(test_data))
    
    table.loc[0]  = ["logloss"] + [metrics.log_loss(test_lebel, proba)]
    table.loc[1]  = ["accuracy_test"] + [metrics.accuracy_score(test_lebel, predicted_label)]
    table.loc[2]  = ["F1_test"] + [metrics.f1_score(test_lebel, predicted_label,average='weighted')]
    table.loc[3]  = ["precision_test"] + [metrics.precision_score(test_lebel, predicted_label, average='weighted')]
    table.loc[4]  = ["auc_test_ovr"] + [metrics.roc_auc_score(test_lebel, proba, multi_class="ovr",average='weighted')]  
    table.loc[5]  = ["auc_test_ovo"] + [metrics.roc_auc_score(test_lebel, proba, multi_class="ovo",average='weighted')]  
    table.loc[6]  = ["r2_test"] + [metrics.r2_score(test_lebel.astype(int), predicted_label.astype(int))]
    
    return table

# plot feature importance
def plot_feature_importance(model, feature_columns):
    import matplotlib.pyplot as plt
    import numpy as np

    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    y_pos  = np.arange(feature_importance.shape[0]) + .5
    fig, ax = plt.subplots()
    f = fig
    fig.set_size_inches(18.5, 10.5, forward=True)
    ax.barh(y_pos, 
            feature_importance, 
            align='center', 
            color='green', 
            ecolor='black', 
            height=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_columns)
    ax.invert_yaxis()
    ax.set_xlabel('Relativna važnost značajki')
    ax.set_title('Važnost značajki')
    plt.show()

# plot fancy confusion matrix
def make_and_plot_confusion_matrix(test_lebel, best_preds):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import matplotlib.cm as cm
    from sklearn.metrics import confusion_matrix

    malware_dict = { 1 : 'Ramnit', 2 : 'Lollipop', 3 : 'Kelihos_ver3', 4 : 'Vundo', 5 : 'Simba', 
                 6 : 'Tracur', 7 : 'Kelihos_ver1', 8 : 'Obfuscator.ACY', 9 : 'Gatak'}

    names = list(malware_dict.values())
    cm = confusion_matrix(test_lebel, best_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    norm_conf = []
    for i in cm:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure(figsize=(10, 10))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Blues, interpolation='nearest')

    width = len(cm)
    height = len(cm[0])

    for x in range(width):
        for y in range(height):
            ax.annotate(str(format(round(cm[x][y], 2))), xy=(y, x), horizontalalignment='center', verticalalignment='center')
    plt.title('Konfuzijska matrica')
    cb = fig.colorbar(res)
    plt.xticks(range(width), names)
    plt.yticks(range(height), names)
    plt.grid(False)

# plot learning curves
def plot_learning_curve( model, X_train, y_train, X_test, y_test, cv, seed ):

    import warnings
    warnings.filterwarnings("ignore")

    # load libraries
    import numpy as np
    from numpy import loadtxt
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split as tts
    from sklearn.metrics import accuracy_score, make_scorer, log_loss

    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve, StratifiedKFold
    

    #plt.style.use('ggplot')
    malware_dict = { 1 : 'Ramnit', 2 : 'Lollipop', 3 : 'Kelihos_ver3', 4 : 'Vundo', 5 : 'Simba', 
                     6 : 'Tracur', 7 : 'Kelihos_ver1', 8 : 'Obfuscator.ACY', 9 : 'Gatak'}

    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(model,
                                               X_train, y_train, cv=StratifiedKFold(n_splits=cv), 
                                               scoring="accuracy",
                                               #scoring=make_scorer(log_loss, needs_proba=True, labels=list(malware_dict.keys())), 
                                               n_jobs=-1,
                                               random_state=seed)

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.subplots(1, figsize=(12,12))
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Uspješnost treniranja")
    plt.plot(train_sizes, test_mean, color="#111111", label="Uspješnost unakrsne validacije")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title("Krivulja učenja")
    plt.xlabel("Veličina skupa za treniranje"), plt.ylabel("Točnost"), plt.legend(loc="best")
    plt.tight_layout(); plt.show()    

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['merror'])
    x_axis = range(0, epochs)

    # plot log loss
    fig, ax = plt.subplots(figsize=(12,12))
    ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.show()

    # plot classification error
    fig, ax = plt.subplots(figsize=(12,12))
    ax.plot(x_axis, results['validation_0']['merror'], label='Train')
    ax.plot(x_axis, results['validation_1']['merror'], label='Test')
    ax.legend()
    plt.ylabel('Pogreška klasifikacije')
    plt.title('XGBoost pogreška klasifikacije')
    plt.show()

# make submission file
def make_submisson_file( md5_hash, predictions ):
    import pandas as pd
    name = input("Ime za submisson file? ")

    result = pd.concat([md5_hash, pd.DataFrame(predictions)], axis=1, sort=False)
    result.columns = ['Id','Prediction1','Prediction2','Prediction3','Prediction4','Prediction5','Prediction6','Prediction7','Prediction8', 'Prediction9']
    filename = '../dataset/submissioni/submisson_' + name + '.csv'
    result.to_csv(filename, index=False)

    return result

# distribution of malware-s over classes
def draw_malware_distribution_over_classes(classes):
    import matplotlib.pyplot as plt

    features_class_quantity = { }
    features_class_precentage = []

    malware_dict = { 1 : 'Ramnit', 2 : 'Lollipop', 3 : 'Kelihos_ver3', 4 : 'Vundo', 5 : 'Simba', 
                 6 : 'Tracur', 7 : 'Kelihos_ver1', 8 : 'Obfuscator.ACY', 9 : 'Gatak'}

    for i in range(1,10):
        features_class_quantity[i] = sum( classes == i)  
        features_class_precentage.append(sum(classes == i)/len(classes) * 100)

    quantity = list(features_class_quantity.values())
    print("Broj malwarea po klasama:")
    print(features_class_quantity.values())
    print("Postotci malware-aa po klasama:")
    print(features_class_precentage)

    fig, ax = plt.subplots(figsize=(15,7))
    ax.bar(list(malware_dict.values()), quantity, color = ['Salmon', 'lightblue', 'lightgreen', 'yellow', 'pink', 'cyan', 'plum', 'peachpuff', 'khaki'])
    plt.xticks(rotation='vertical')
    plt.xlabel('Klase malware-a', fontweight='bold')
    plt.ylabel('Količina', fontweight='bold')

    plt.show()

# show best (default 3) parameters and return them in a list
def report(results, n_top=3):
    import pandas as pd
    import numpy as np

    params = []
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model ranga: {0}".format(i))
            print("Mean validation score: {0:.5f} (std: {1:.5f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parametri: {0}".format(results['params'][candidate]))
            print("")
            params.append(results['params'][candidate])
    return params

# make or load basic model
def XGBClassifier_load_or_make(X_train, y_train, X_test, y_test, n_estimators=1000, early_stopping_rounds=20, eval_metric=["merror", "mlogloss"]):
    import xgboost as xgb
    from xgboost import XGBClassifier
    import pickle
    import numpy as np

    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    load_or_make = input("Load or make XGBClassifier? ")

    if load_or_make == "make":
        # for saving
        name = input("Ime za XGBClassifier? ")
        filename = 'data/model_xgb_' + name + '.sav'

        basic_model_xgb = xgb.XGBClassifier(n_estimators=n_estimators, n_jobs=-1)
        basic_model_xgb.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric, eval_set=eval_set, verbose=True)
        
        # save
        pickle.dump(basic_model_xgb, open(filename, 'wb'))
        print('XGBClassifier je spremljen u: ' + filename )
        return basic_model_xgb
        
    elif load_or_make == "load":
        # pick model
        print('Izaberi XGBClassifier:')
        print('1. basic_model_xgb')
        print('2. basic_one_gram')
        #print('3. basic_metadata_bytes.sav')
        print('4. basic_entropy')
        print('5. basic_image')
        print('6. basic_string_len')
        print('7. basic_metadata_asm')
        print('8. basic_symbols')
        print('9. basic_opcode')
        print('10. basic_reg')
        print('11. basic_section')
        print('12. basic_dd')
        print('13. basic_api')
        print('14. basic_keywords')
        print('15. all_features')

        option = input()
        if option == '1':
            filename = '../dataset/basic_model/basic_model_xgb.sav'
        elif option == '2':
            filename = '../dataset/basic_model/model_xgb_one_gram.sav'
        #elif option == '3':
        #    filename = '../../dataset/basic_model/model_xgb_one_gram.sav'
        elif option == '4':
            filename = '../dataset/basic_model/model_xgb_entropija_80_20.sav'
        elif option == '5':
            filename = '../dataset/basic_model/model_xgb_image_80_20.sav'
        elif option == '6':
            filename = '../dataset/basic_model/model_xgb_string_length_features_80_20.sav'
        elif option == '7':
            filename = '../dataset/basic_model/model_xgb_metadata_asm_features.sav'
        elif option == '8':
            filename = '../dataset/basic_model/model_xgb_symbols_features.sav'
        elif option == '9':
            filename = '../dataset/basic_model/model_xgb_opcode_features.sav'
        elif option == '10':
            filename = '../dataset/basic_model/model_xgb_reg_features.sav'
        elif option == '11':
            filename = '../dataset/basic_model/model_xgb_section_features_80_20.sav'
        elif option == '12':
            filename = '../dataset/basic_model/model_xgb_dd_features.sav'
        elif option == '13':
            filename = '../dataset/basic_model/model_xgb_apis.sav'
        elif option == '14':
            filename = '../dataset/basic_model/model_xgb_key_features.sav'
        elif option == '15':
            filename = '../dataset/basic_model/model_xgb_svi_featuri.sav'
        else:
            print('Ne postoji zatražena opcija!')
            raise ValueError('Ne postoji zatražena opcija - pri učitavanju XGBClassifiera!')
            return 
        
        basic_model_xgb = pickle.load(open(filename, 'rb'))
        basic_model_xgb.get_params()

        return basic_model_xgb

    else:
        print("Krivi unos! Upiši 'load' ili 'make'!")
        raise ValueError('Nije upisano load ili make - pri učitavanju/izradi XGBClassifiera!')
        return
  
#newer xgboost version
def XGBClassifier_newer_load_or_make(X_train, y_train, X_test, y_test, n_estimators=1000, early_stopping_rounds=20, eval_metric=["merror", "mlogloss"]):
    import xgboost as xgb
    from xgboost import XGBClassifier
    import pickle
    import numpy as np

    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    load_or_make = input("Load or make XGBClassifier? ")

    if load_or_make == "make":
        # for saving
        name = input("Ime za XGBClassifier? ")
        filename = '../dataset/basic_model/model_xgb_' + name + '.sav'

        basic_model_xgb = xgb.XGBClassifier(n_estimators=n_estimators, n_jobs=-1)
        basic_model_xgb.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric, eval_set=eval_set, verbose=True)
        
        # save
        print('XGBClassifier je spremljen u: ' + filename )
        basic_model_xgb.save_model(filename)
        return basic_model_xgb
        
    elif load_or_make == "load":
        # pick model
        print('Izaberi XGBClassifier:')
        print('1. basic_model_xgb')
        print('2. basic_one_gram')
        #print('3. basic_metadata_bytes.sav')
        print('4. basic_entropy')
        print('5. basic_image')
        print('6. basic_string_len')
        print('7. basic_metadata_asm')
        print('8. basic_symbols')
        print('9. basic_opcode')
        print('10. basic_reg')
        print('11. basic_section')
        print('12. basic_dd')
        print('13. basic_api')
        print('14. basic_keywords')
        print('15. all_features')

        option = input()
        if option == '1':
            filename = '../dataset/basic_model/basic_model_xgb.sav'
        elif option == '2':
            filename = '../dataset/basic_model/model_xgb_one_gram.sav'
        #elif option == '3':
        #    filename = '../../dataset/basic_model/model_xgb_one_gram.sav'
        elif option == '4':
            filename = '../dataset/basic_model/model_xgb_entropija_80_20.sav'
        elif option == '5':
            filename = '../dataset/basic_model/model_xgb_image_80_20.sav'
        elif option == '6':
            filename = '../dataset/basic_model/model_xgb_string_length_features_80_20.sav'
        elif option == '7':
            filename = '../dataset/basic_model/model_xgb_metadata_asm_features.sav'
        elif option == '8':
            filename = '../dataset/basic_model/model_xgb_symbols_features.sav'
        elif option == '9':
            filename = '../dataset/basic_model/model_xgb_opcode_features.sav'
        elif option == '10':
            filename = '../dataset/basic_model/model_xgb_reg_features.sav'
        elif option == '11':
            filename = '../dataset/basic_model/model_xgb_section_features_80_20.sav'
        elif option == '12':
            filename = '../dataset/basic_model/model_xgb_dd_features.sav'
        elif option == '13':
            filename = '../dataset/basic_model/model_xgb_apis.sav'
        elif option == '14':
            filename = '../dataset/basic_model/model_xgb_key_features.sav'
        elif option == '15':
            filename = '../dataset/basic_model/model_xgb_svi_featuri.sav'
        else:
            print('Ne postoji zatražena opcija!')
            raise ValueError('Ne postoji zatražena opcija - pri učitavanju XGBClassifiera!')
            return 
        
        basic_model_xgb = pickle.load(open(filename, 'rb'))
        basic_model_xgb.get_params()

        return basic_model_xgb

    else:
        print("Krivi unos! Upiši 'load' ili 'make'!")
        raise ValueError('Nije upisano load ili make - pri učitavanju/izradi XGBClassifiera!')
        return


# randomly search for hyperparameters
def RandomizedSearchCV_load_or_make(model, data, labels, random_grid, cv="5", scoring="accuracy", n_iter=20, random_state=47):
    import xgboost as xgb
    from xgboost import XGBClassifier
    from sklearn.model_selection import RandomizedSearchCV
    import pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

    # we use the RandomizedSearchCV to find the best parameters for our XGB model

    load_or_make = input("Load or make RandomizedSearchCV?")
    if load_or_make == "load":
        
        # pick RandomizedSearchCV to load
        print('Izaberi RandomizedSearchCV:')
        print('1. RandomizedSearchCV10_basic_all_features_neg_log_loss ')
        
        option = input()
        if option == '1':
            filename = '.../dataset/basic_modeli/RandomizedSearchCV10_basic_all_features_neg_log_loss.sav'
        else:
            print('Ne postoji zatražena opcija!')
            raise ValueError('Ne postoji zatražena opcija - pri učitavanju RandomizedSearchCV!')
            return
        
        rand_XGB = pickle.load(open(filename, 'rb'))
        rand_XGB.get_params()
        
        # show results
        rand_XGB_results_df = pd.DataFrame(rand_XGB.cv_results_)[['mean_test_score', 'std_test_score', 'params','rank_test_score']]
        rand_XGB_results_df

        # plot of randomized search results
        rand_XGB_mean_scores = rand_XGB.cv_results_['mean_test_score']
        plt.plot(list(range(1, 21)), rand_XGB_mean_scores)
        plt.xlabel('k-ti Model Randomized Search CV treniranja (XGB)')
        plt.ylabel('Točnost unakrsne validacije')

        return [rand_XGB, rand_XGB_results_df]
        
    elif load_or_make == "make":
        # getting ready for saving later
        name = input('Unesi ime novo-pokrenutoga RandomizedSearchCV? ')
        filename = 'data/RandomizedSearchCV_' + name + '.sav'

        model_XGB = model
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

        # save 
        pickle.dump(rand_XGB, open(filename, 'wb'))
        print('RandomizedSearchCV je spremljen u: ' + filename )

        # show results
        rand_XGB_results_df = pd.DataFrame(rand_XGB.cv_results_)[['mean_test_score', 'std_test_score', 'params', 'rank_test_score']]
        rand_XGB_results_df

        # plot of randomized search results
        rand_XGB_mean_scores = rand_XGB.cv_results_['mean_test_score']
        plt.plot(list(range(1, 21)), rand_XGB_mean_scores)
        plt.xlabel('k-ti Model Randomized Search CV treniranja (XGB)')
        plt.ylabel('Točnost unakrsne validacije')

        return [rand_XGB, rand_XGB_results_df]

    else:
        print("Krivi unos! Upiši 'load' ili 'make'!")
        raise ValueError('Nije upisano load ili make - pri učitavanju/izradi RandomizedSearchCV!')
        return

# make model with specific parameters
def XGBClassifier_with_params(grid_or_random_search, X_train, y_train, X_test, y_test, early_stopping_rounds=20, eval_metric=["merror", "mlogloss"]):
    import xgboost as xgb
    from xgboost import XGBClassifier
    import pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # we use the RandomizedSearchCV to find the best parameters for our XGB model
    eval_set = [(X_train, y_train), (X_test, y_test)]

    load_or_make = input("Load or make XGBClassifier?")

    if load_or_make == "load":
        print("Još nema opcija.")
        raise ValueError('Još uvijek nema opcija - pri učitavanju XGBClassifier_with_params!')
        return

    elif load_or_make == "make":
        name = input('Unesi ime novo-pokrenutoga XGBClassifiera? ')
        filename = 'data/XGBClassifier_' + name + '.sav'

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
        param_tuning_xgb.fit(X_train=X_train, y_train=y_train, early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric, eval_set=eval_set, verbose=True)

        # save 
        pickle.dump(param_tuning_xgb, open(filename, 'wb'))
        print("XGBClassifier je spremljen u: " + filename )

        # show results
        param_tuning_xgb_results_df = model_results(param_tuning_xgb, X_test, y_test)
        param_tuning_xgb_results_df

        return [param_tuning_xgb, param_tuning_xgb_results_df]
        
    else:
        print("Krivi unos! Upiši 'load' ili 'make'!")
        raise ValueError('Nije upisano load ili make - pri učitavanju/izradi XGBClassifier_with_params!')
        return

#grid search in space of given parameters
def GridSearchCV_load_or_make(model, param_grid, data, labels, cv=5):
    import xgboost as xgb
    from xgboost import XGBClassifier
    import pickle
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # we use the GridSearchCV to find even better parameters for our XGB model
    load_or_make = input("Load or make GridSearchCV?")

    if load_or_make == "load":
        print("Još nema opcija.")
        raise ValueError('Nije upisano load ili make - pri učitavanju GridSearchCV_load_or_make!')
        return
        
    elif load_or_make == "make":
        m_XGB = XGBClassifier()

        name = input("Ime za novi GridSearchCV? ")
        filename = 'data/GridSearchCV_' + name + '.sav'

        # grid search
        grid_search = GridSearchCV(m_XGB, param_grid=param_grid, cv=cv, verbose=2, n_jobs=-1)
        grid_search.fit(data, labels)

        # save 
        pickle.dump(grid_search, open(filename, 'wb'))
        print("GridSearchCV je spremljen u: " + filename )

        # show results
        grid_search_results_df = pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params', 'rank_test_score']]
        grid_search_results_df


        # plot of randomized search results
        grid_search_mean_scores = grid_search.cv_results_['mean_test_score']
        plt.plot(list(range(1, 21)), grid_search_mean_scores)
        plt.xlabel('k-ti Model Randomized Search CV treniranja (XGB)')
        plt.ylabel('Točnost unakrsne validacije')

        return [grid_search, grid_search_results_df]
        
    else:
        print("Krivi unos! Upiši 'load' ili 'make'!")
        raise ValueError('Nije upisano load ili make - pri učitavanju GridSearchCV_load_or_make!')
        return

# distribution of malware-s over classes
def draw_malware_distribution_over_classes(classes):
    import matplotlib.pyplot as plt

    features_class_quantity = { }
    features_class_precentage = []

    malware_dict = { 1 : 'Ramnit', 2 : 'Lollipop', 3 : 'Kelihos_ver3', 4 : 'Vundo', 5 : 'Simba', 
                 6 : 'Tracur', 7 : 'Kelihos_ver1', 8 : 'Obfuscator.ACY', 9 : 'Gatak'}

    for i in range(1,10):
        features_class_quantity[i] = sum( classes == i)  
        features_class_precentage.append(sum(classes == i)/len(classes) * 100)

    quantity = list(features_class_quantity.values())
    print("Broj malwarea po klasama:")
    print(features_class_quantity.values())
    print("Postotci malware-aa po klasama:")
    print(features_class_precentage)

    fig, ax = plt.subplots(figsize=(15,7))
    ax.bar(list(malware_dict.values()), quantity, color = ['Salmon', 'lightblue', 'lightgreen', 'yellow', 'pink', 'cyan', 'plum', 'peachpuff', 'khaki'])
    plt.xticks(rotation='vertical')
    plt.xlabel('Klase malware-a', fontweight='bold')
    plt.ylabel('Količina', fontweight='bold')

    plt.show()

def compare_malware_distribution_over_classes(classes):
    import matplotlib.pyplot as plt
    import pandas as pd

    features_class_quantity = { }
    features_class_precentage = []

    malware_dict = { 1 : 'Ramnit', 2 : 'Lollipop', 3 : 'Kelihos_ver3', 4 : 'Vundo', 5 : 'Simba', 
                 6 : 'Tracur', 7 : 'Kelihos_ver1', 8 : 'Obfuscator.ACY', 9 : 'Gatak'}
    features_train = pd.read_csv("https://raw.githubusercontent.com/mateastanisic/lfs/master/merged_all_features.csv")

    
    classes_begining_and_end_in_features = { }
    last = 0

    for i in range(1,10):
        classes_begining_and_end_in_features[i] = [last, last + len(features_train.loc[features_train.Class == i]) - 1]
        last = last + len(features_train.loc[features_train.Class == i])

    features_class_quantity_original = { }

    for i in range(1,10):
        features_class_quantity_original[i] = classes_begining_and_end_in_features[i][1] - classes_begining_and_end_in_features[i][0]  
        features_class_quantity[i] = sum( classes == i)  
        
    classes_original = list(malware_dict.values())
    quantity_original = list(features_class_quantity_original.values())
    quantity_test = list(features_class_quantity.values())
    
    listaa = [ classes_original, quantity_original, quantity_test ]
    df = pd.DataFrame() 
    df['Classes'] = classes_original
    df['Original'] = quantity_original
    df['Test'] = quantity_test
    df.set_index("Classes", inplace = True) 
    
    
    df.plot( kind="bar", figsize=( 15,5 ), color=['lightskyblue', 'darkblue'], title = "Distribucija malwarea po klasama",
            rot=1 ).title.set_size( 15 )
    plt.ylabel( "Malware quantity", fontsize = 10 )
    plt.show()

