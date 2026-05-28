def Ca_Related_Channels_modifier(cell, params):
    # Calcium channels
    if params['gcalbar_ratio'] is not None:
        for sec in cell.apical:
            sec.gcalbar_cal = sec.gcalbar_cal * params['gcalbar_ratio']
        for sec in cell.soma:
            sec.gcalbar_cal = sec.gcalbar_cal * params['gcalbar_ratio']
        for sec in cell.basal:
            sec.gcalbar_cal = sec.gcalbar_cal * params['gcalbar_ratio']

    if params['gcanbar_ratio'] is not None:
        for sec in cell.apical:
            sec.gcanbar_can = sec.gcanbar_can * params['gcanbar_ratio']
        for sec in cell.soma:
            sec.gcanbar_can = sec.gcanbar_can * params['gcanbar_ratio']
        for sec in cell.basal:
            sec.gcanbar_can = sec.gcanbar_can * params['gcanbar_ratio']

    if params['gcatbar_ratio'] is not None:
        for sec in cell.apical:
            sec.gcatbar_cat = sec.gcatbar_cat * params['gcatbar_ratio']
        for sec in cell.soma:
            sec.gcatbar_cat = sec.gcatbar_cat * params['gcatbar_ratio']
        for sec in cell.basal:
            sec.gcatbar_cat = sec.gcatbar_cat * params['gcatbar_ratio']

    # Calcium-dependent K+ channels
    if params['gcakbar_ratio'] is not None:
        for sec in cell.apical:
            sec.gbar_cagk = sec.gbar_cagk * params['gcakbar_ratio']
        for sec in cell.soma:
            sec.gbar_cagk = sec.gbar_cagk * params['gcakbar_ratio']
        for sec in cell.basal:
            sec.gbar_cagk = sec.gbar_cagk * params['gcakbar_ratio']
    if params['gslowcakbar_ratio'] is not None:
        for sec in cell.apical:
            sec.gbar_kca = sec.gbar_kca * params['gslowcakbar_ratio']
        for sec in cell.soma:
            sec.gbar_kca = sec.gbar_kca * params['gslowcakbar_ratio']
        for sec in cell.basal:
            sec.gbar_kca = sec.gbar_kca * params['gslowcakbar_ratio']
