# funkcija za crtanje histograma pri čemu se uzimaju u obzir kvantili pri odabiru binsa

# ulazne varijable:
#                    data - dataframe varijabla sa značajkama
#                    classes - varijabla koja ima informacije o početnom i krajnjem retku neke klase malwarea
#                    features - polje sa točno određenim stupcim koji predstavljaju značajke za koje ćemo nacrtati histogram
#                    sections - na koliko dijelova ćemo podijeliti histogram
#                    no_rows - broj redaka histograma
#                    no_cols - broj stupaca histograma
#                    fi_x, fig_y - dimenzije grafa histograma
#                    i_want_whole_range - True or False, depending if we want histrogram throughout whole range max - min or not

def draw_histograms_rows_quantile( data, classes, features, sections, no_rows, no_cols, fig_x, fig_y, i_want_whole_range ):
    
    # potrebni paketi
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # rječnik za pamćenje imena klasa malwarea
    malware_dict = { 1 : 'Ramnit', 2 : 'Lollipop', 3 : 'Kelihos_ver3', 4 : 'Vundo', 5 : 'Simba', 
                     6 : 'Tracur', 7 : 'Kelihos_ver1', 8 : 'Obfuscator.ACY', 9 : 'Gatak'}

    fig, axes = plt.subplots(no_rows, no_cols, figsize=(fig_x, fig_y))
    ax = axes.ravel() 

    for i in range(len(features)):
        col = features[i]
        
        noNaNdata = data.iloc[:,col].to_numpy()[~np.isnan(data.iloc[:,col].to_numpy())]
        q20 = np.quantile(noNaNdata, 0.25)
        q80 = np.quantile(noNaNdata, 0.75)
        if q20 == q80 or i_want_whole_range[i] :
            _, bins = np.histogram(noNaNdata, bins=sections)
        else:
            bins = [ q20 + j * ( (q80-q20)/sections ) for j in range(sections)]
        
        ax[i].hist(data.iloc[classes[1][0]:classes[1][1], col].to_numpy(), 
                   bins=bins, color='red', alpha=.5, label=malware_dict[1])

        ax[i].hist(data.iloc[classes[2][0]:classes[2][1], col].to_numpy(), 
                   bins=bins, color='blue', alpha=.5, label=malware_dict[2])

        ax[i].hist(data.iloc[classes[3][0]:classes[3][1], col].to_numpy(), 
                   bins=bins, color='green', alpha=.5, label=malware_dict[3])

        ax[i].hist(data.iloc[classes[4][0]:classes[4][1], col].to_numpy(), 
                   bins=bins, color='yellow', alpha=.5, label=malware_dict[4])

        ax[i].hist(data.iloc[classes[5][0]:classes[5][1], col].to_numpy(), 
                   bins=bins, color='orange', alpha=.5, label=malware_dict[5])

        ax[i].hist(data.iloc[classes[6][0]:classes[6][1], col].to_numpy(), 
                   bins=bins, color='pink', alpha=.5, label=malware_dict[6])

        ax[i].hist(data.iloc[classes[7][0]:classes[7][1], col].to_numpy(), 
                   bins=bins, color='grey', alpha=.5, label=malware_dict[7])

        ax[i].hist(data.iloc[classes[8][0]:classes[8][1], col].to_numpy(), 
                   bins=bins, color='magenta', alpha=.5, label=malware_dict[8])   

        ax[i].hist(data.iloc[classes[9][0]:classes[9][1], col].to_numpy(), 
                   bins=bins, color='purple', alpha=.5, label=malware_dict[9]) 
        
        ax[i].set_title(data.columns.values[col])
        ax[i].set_yticks(()) # remove ticks on y-axis
        ax[i].legend(loc='upper right')
        fig.tight_layout()