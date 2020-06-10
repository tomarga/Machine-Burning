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
#                    one - array of indices of the class we want to compare with other classes in same histogram

def draw_histograms_rows_quantile_one_vs_others( data, classes, features, sections, no_rows, no_cols, fig_x, fig_y, i_want_whole_range, one ):
    
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
        
        index = i
        
        for f in one:
            
            ax[index].hist(data.iloc[classes[f][0]:classes[f][1], col].to_numpy(), 
                           bins=bins, color='red', lw=3, alpha=.5, label=malware_dict[1])

            for j in range(1,10):
                if not j == f:
                    ax[index].hist(data.iloc[classes[j][0]:classes[j][1], col].to_numpy(), 
                               bins=bins, color='grey', alpha=.5, label=malware_dict[j])


            ax[index].set_title(data.columns.values[col])
            ax[index].set_yticks(()) # remove ticks on y-axis
            ax[index].legend(loc='upper right')
            fig.tight_layout()
            
            index += 1