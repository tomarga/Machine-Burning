
def rfecv_save(rfecv_rf, rfecv_name, feature_train, feature_train_original, feature_test):
    # packages
    import numpy as np
    import pandas as pd
    import seaborn as sn
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.mplot3d import Axes3D

    import sklearn as sk
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import RFECV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
        
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)


    # write parameters of rfecv in file
    rfecv_parameters = rfecv_rf.get_params()
    filename = 'rfecv_parameters_' + rfecv_name + '.txt'

    with open(filename, 'w') as f:
        print(rfecv_parameters, file=f)

    # save selected features wiith their importances into csv file 
    index = 0
    selected_features = []
    selected_features_importance = []
    for i in list(feature_train.columns.values[:][rfecv_rf.get_support()]): 
        # append selected feature name
        selected_features.append(i)
        # append selected feature importance
        selected_features_importance.append(rfecv_rf.estimator_.feature_importances_.ravel()[index])
        # increase index
        index = index + 1
    
    temp_df = pd.concat([pd.DataFrame(selected_features), pd.DataFrame(selected_features_importance)], ignore_index=True, sort =False, axis=1)
    temp_df.columns = ["feature_name","feature_importance"]

    filename = rfecv_name + '_feature_selected_with_names_and_importance_scores.csv'
    temp_df.to_csv(filename, index=False)

    # save sorted
    filename = rfecv_name + '_feature_selected_with_names_and_importance_scores_sorted.csv'
    temp2_df = temp_df.sort_values(by=['feature_importance'], ascending=False)
    temp2_df.to_csv(filename, index=False)


    # save table of filtered features
    selected_features_rfc_train = feature_train[list( feature_train.columns.values[:][rfecv_rf.get_support()] )]
    filename = 'features_train_' + rfecv_name + '.csv'
    selected_features_rfc_train.to_csv(filename, index=False)

    selected_features_rfc_test = feature_test[list( feature_test.columns.values[:][rfecv_rf.get_support()] )]
    filename = 'features_test_' + rfecv_name + '.csv'
    selected_features_rfc_test.to_csv(filename, index=False)

    return [temp2_df, selected_features_rfc_train, selected_features_rfc_test]


def rfecv_hist(rfecv_rf):
    # packages
    import numpy as np
    import pandas as pd
    import seaborn as sn
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.mplot3d import Axes3D

    import sklearn as sk
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import RFECV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
        
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)


    # draw histogram of features importances
    pd.Series(rfecv_rf.estimator_.feature_importances_.ravel()).hist(figsize=(15,10))
    plt.ylabel('Number of features')
    plt.xlabel('Importances')


def rfecv_plot(rfecv_rf):
    # packages
    import numpy as np
    import pandas as pd
    import seaborn as sn
    
    import matplotlib.cm as cm
    from mpl_toolkits.mplot3d import Axes3D

    import sklearn as sk
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import RFECV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
        
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    import matplotlib.pyplot as plt
    
    # plot recursive elimination of features 
    plt.figure(figsize=(15,5))

    plt.plot(range(1, len(rfecv_rf.grid_scores_) + 1), rfecv_rf.grid_scores_, '-o',color='gray')
    plt.xlabel('Broj odabranih značajki')
    plt.ylabel('Točnost na unakrsnoj validaciji')
    plt.title('Rekurzivna eliminacija značajki s unakrsnom validacijom')
    plt.vlines(rfecv_rf.n_features_, 
               np.min(rfecv_rf.grid_scores_), 
               rfecv_rf.grid_scores_[rfecv_rf.n_features_-1], 
               color='red', linestyle='--')
    plt.show()



def features_type_quantity(feature_names, feature_train_original):

    # plot graph of feature distribution over groups of features
    features_type_quantity = { 'one_gram' : 0, 'metadata_bytes' : 0, 'entropy' : 0,
                              'image' : 0, 'string_length' : 0, 'metadata_asm' : 0,
                              'symbols' : 0, 'opcode' : 0, 'reg' : 0, 'section' : 0,
                              'dd' : 0, 'api' : 0, 'key' : 0 }
    
    features_type_quantity_original = { 'one_gram' : 256, 'metadata_bytes' : 2, 'entropy' : 202,
                              'image' : 52, 'string_length' : 116, 'metadata_asm' : 2,
                              'symbols' : 7, 'opcode' : 93, 'reg' : 26, 'section' : 24,
                              'dd' : 24, 'api' : 794, 'key' : 95 }
    
    for i in list(feature_names):
        
        column_index = feature_train_original.columns.get_loc( i )

        if ( column_index < 258 ) : features_type_quantity['one_gram'] += 1
        elif ( column_index < 260 ) : features_type_quantity['metadata_bytes'] += 1
        elif ( column_index < 462 ) : features_type_quantity['entropy'] += 1
        elif ( column_index < 514 ) : features_type_quantity['image'] += 1
        elif ( column_index < 630 ) : features_type_quantity['string_length'] += 1
        elif ( column_index < 632 ) : features_type_quantity['metadata_asm'] += 1
        elif ( column_index < 639 ) : features_type_quantity['symbols'] += 1
        elif ( column_index < 732 ) : features_type_quantity['opcode'] += 1
        elif ( column_index < 758 ) : features_type_quantity['reg'] += 1
        elif ( column_index < 782 ) : features_type_quantity['section'] += 1
        elif ( column_index < 806 ) : features_type_quantity['dd'] += 1
        elif ( column_index < 1600 ) : features_type_quantity['api'] += 1
        elif ( column_index < 1695 ) : features_type_quantity['key'] += 1
            
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(15,7))
    ax.bar(features_type_quantity_original.keys(), features_type_quantity_original.values(), color = ['whitesmoke'])
    ax.bar(features_type_quantity.keys(), features_type_quantity.values(), color = ['Salmon', 'lightblue', 'lightgreen', 'yellow', 'pink', 'cyan', 'plum', 'peachpuff', 'khaki', 'wheat', 'darkseagreen','thistle','wheat'])
    plt.xticks(rotation='vertical')
    plt.xlabel('Kategorija značajki', fontweight='bold')
    plt.ylabel('Broj odabranih značajki', fontweight='bold')

    plt.show()   



def pca_plot(train, test, malware_classes, rfecv_name):
    # packages
    import numpy as np
    import pandas as pd
    import seaborn as sn
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.mplot3d import Axes3D

    import sklearn as sk
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import RFECV
    from sklearn.ensemble import RandomForestClassifier
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

    scaler = StandardScaler()
    X = train.copy()
    X_scaled = scaler.fit_transform(X)

    pca = PCA() 
    X_pca = pca.fit_transform(X_scaled) 

    print("Dimenzije originalnih podataka: %s" % str(X_scaled.shape))
    print("Dimenzije projiciranih podataka: %s" % str(X_pca.shape))
    
    # save features
    selected_features_pca_train = pd.DataFrame(X_pca, columns=['PC'+str(i) for i in range(1,X_pca.shape[1]+1)])
    filename = 'selected_features_pca_train_' + rfecv_name + '.csv'
    selected_features_pca_train.to_csv(filename, index=False)

    # test data
    scalert = StandardScaler()
    newdata = test.copy()
    newdata_scaled = scalert.fit_transform(newdata)

    # transform new data using already fitted pca(don't re-fit the pca)
    newdata_transformed = pca.transform(newdata_scaled)

    # save features
    selected_features_pca_test = pd.DataFrame(X_pca, columns=['PC'+str(i) for i in range(1,X_pca.shape[1]+1)])
    filename = 'selected_features_pca_test_' + rfecv_name + '.csv'
    selected_features_pca_test.to_csv(filename, index=False)



    # TO 3D
    scaler_3 = StandardScaler()
    X_3 = train.copy()
    X_scaled_3 = scaler_3.fit_transform(X_3)

    pca_3 = PCA(n_components=3) 
    X_pca_3 = pca_3.fit_transform(X_scaled_3) 


    fig = plt.figure(1, figsize=(50,50))
    
    ax = plt.axes(projection='3d')
    ax.scatter(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2], c = malware_classes.values.ravel(), cmap=plt.cm.nipy_spectral, linewidth=0.5)

    return [selected_features_pca_train, selected_features_pca_test]
