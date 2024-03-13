base_dpi={
        'Trametinib':{
                'P36507':(0.9,-1),
                'Q02750':(0.9,-1),
                },
        'Crizotinib':{
                'P08581':(0.9,-1),
                'Q9UM73':(0.9,-1),
                },
        'Lapatinib':{
                'P04626':(0.9,-1),
                'P00533':(0.9,-1),
                'Q15303':(0.9,-1),
                },
        'Gemcitabine':{
                'P23921':(0.9,-1), # RRM1
                'P04818':(0.9,-1), # TYMS
                },
        }

def base_dpi_data(name):
    return base_dpi[name]
