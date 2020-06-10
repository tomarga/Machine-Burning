# funkcija za prikaz koreliranosti (najboljih) značajki neke kategorije 

# ulazne varijable:
#                    features_category - dataframe značajki kategorije
#                    best_features_in_category - lista najboljih značajki

def draw_heatmap (features_category, best_features_in_category): 
    
    # potrebni paketi
    import pandas as pd
    import seaborn as sn
    import matplotlib.pyplot as plt
    
    best_features_in_category_dataframe = pd.DataFrame()
    
    for features in features_category.columns:
        for best_features in best_features_in_category:
            if (features == best_features):
                best_features_in_category_dataframe.insert(best_features_in_category_dataframe.shape[1], features, features_category[features])

    plt.subplots(figsize=(20,15))
    sn.heatmap(best_features_in_category_dataframe.corr(), cmap='Purples', annot=True)
    
    
    
def draw_heatmap_anNot (features_category, best_features_in_category): 
    
    # potrebni paketi
    import pandas as pd
    import seaborn as sn
    import matplotlib.pyplot as plt
    
    best_features_in_category_dataframe = pd.DataFrame()
    
    for features in features_category.columns:
        for best_features in best_features_in_category:
            if (features == best_features):
                best_features_in_category_dataframe.insert(best_features_in_category_dataframe.shape[1], features, features_category[features])

    plt.subplots(figsize=(20,15))
    sn.heatmap(best_features_in_category_dataframe.corr(), cmap='Purples')