def in_good_range(data, feature, classes):
    import numpy as np
    malware_dict = { 1 : 'Ramnit', 2 : 'Lollipop', 3 : 'Kelihos_ver3', 4 : 'Vundo', 5 : 'Simba', 
                 6 : 'Tracur', 7 : 'Kelihos_ver1', 8 : 'Obfuscator.ACY', 9 : 'Gatak'}
    
    in_range = {}
    outliers = {}
    prosjek = {}
    
    print("Prosjek koliko se malware-a nalazi unutar 25 % najmanjih i 25 % najvećih elemenata skupa podataka:")
    for i in range(1,10):
        one_class = data.iloc[classes[i][0]:classes[i][1], feature]
        q25 = np.quantile(one_class, 0.25)
        q75 = np.quantile(one_class, 0.75)
        brk_donji =  q25 - 3/2 * (q75-q25)
        brk_gornji = q75 + 3/2 * (q75-q25)
        one_class_in_range = [i for i in one_class if i >= brk_donji and i <= brk_gornji]
        in_range.update({ i : len(one_class_in_range) })
        outliers.update({ i : len(data.iloc[classes[i][0]:classes[i][1], feature]) - len(one_class_in_range) })
        prosjek.update({ i : len(one_class_in_range)/len(data.iloc[classes[i][0]:classes[i][1], feature])})
        print(malware_dict[i], " - ", prosjek[i]*100, "%")
        
    return [in_range, outliers, prosjek]