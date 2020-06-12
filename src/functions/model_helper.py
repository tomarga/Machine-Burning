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


def make_submisson_file( md5_hash, predictions ):
    import pandas as pd

    result = pd.concat([md5_hash, pd.DataFrame(predictions)], axis=1, sort=False)
    result.columns = ['Id','Prediction1','Prediction2','Prediction3','Prediction4','Prediction5','Prediction6','Prediction7','Prediction8', 'Prediction9']
    result.to_csv('submisson.csv', index=False)

    return result

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
    print("Postotci malwarea po klasama:")
    print(features_class_precentage)

    fig, ax = plt.subplots(figsize=(15,7))
    ax.bar(list(malware_dict.values()), quantity, color = ['Salmon', 'lightblue', 'lightgreen', 'yellow', 'pink', 'cyan', 'plum', 'peachpuff', 'khaki'])
    plt.xticks(rotation='vertical')
    plt.xlabel('Klase malware-a', fontweight='bold')
    plt.ylabel('Količina', fontweight='bold')

    plt.show()