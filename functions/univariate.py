# funkcija za univarijantni odabir značajki

# ulazne varijable:
#                     features_category - dataframe značajki kategorije
#                     k_best - broj najboljih značajki koje funkcija vraća

def univariate( features_category, k_best ):

    # potrebni paketi
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import SelectKBest, f_classif
    
    
    features = pd.read_csv("https://raw.githubusercontent.com/mateastanisic/lfs/master/merged_all_features.csv")
    
    malware_train, malware_test, malware_classes_train, malware_classes_test = train_test_split( features_category.dropna(), features['Class'], test_size=0.4, random_state=47)
    selector = SelectKBest(f_classif, k=k_best)
    selector.fit(malware_train, malware_classes_train)

    scores = -np.log10(replaceZeroes(selector.pvalues_))
    indices = np.argsort(scores)[::-1]
    noises = indices[k_best:]

    return features_category.columns.values[indices[:k_best]]




# funkcija koja uz univarijantni odabir značajki crta i graf

# ulazne varijable:
#                     features_category - dataframe značajki kategorije
#                     k_best - broj najboljih značajki koje funkcija vraća


def univariate_plot( features_category, k_best ):

    # potrebni paketi
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import SelectKBest, f_classif
    
    
    features = pd.read_csv("https://raw.githubusercontent.com/mateastanisic/lfs/master/merged_all_features.csv")
    
    malware_train, malware_test, malware_classes_train, malware_classes_test = train_test_split( features_category.dropna(), features['Class'], test_size=0.4, random_state=47)
    selector = SelectKBest(f_classif, k=k_best)
    selector.fit(malware_train, malware_classes_train)

    scores = -np.log10(replaceZeroes(selector.pvalues_))
    indices = np.argsort(scores)[::-1]
    noises = indices[k_best:]

    plt.figure(figsize=(100,30))
    plt.grid(False)
    plt.plot(range(malware_train.shape[1]), scores[indices], 'o', color='gray', markersize=20)

    plt.title('Univarijantni odabir znacajki za najvaznijih ' + (str)(k_best) + ' znacajki', fontsize=100)
    plt.ylabel('log ANOVA p-vrijednosti', fontsize = 75 )
    plt.xlim([-1, malware_train.shape[1]])
    plt.xticks(range(malware_train.shape[1]), features_category.columns.values[indices], rotation=90, fontsize=20)
    plt.yticks( fontsize=50 )
    plt.vlines(k_best-0.5, 0, np.max(scores), color='red', linestyle='--', linewidth=10)
    plt.grid(True,linestyle='--')
    plt.tight_layout()

    [ i.set_color("red") for i in plt.gca().get_xticklabels() if i.get_text() in [ features_category.columns.values[noise] for noise in noises] ]

    plt.show()
    return features_category.columns.values[indices[:k_best]]



# pomoćna funkcija 
def replaceZeroes(data):
    
    import numpy as np
    
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data