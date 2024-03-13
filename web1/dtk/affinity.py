# NOTE: These values were originally from databases/matching/opt_c50...
# but now all c50/ki evidence conversion should route through here.
C50_CONV = dict(for_hi=200, for_lo=630, hi_evid=0.9, lo_evid=0.5)
KI_CONV = dict(for_hi=40, for_lo=320, hi_evid=0.7, lo_evid=0.4)

def measure_to_ev(conv, measurement):
    if measurement <= conv['for_hi']:
        return conv['hi_evid']
    elif measurement <= conv['for_lo']:
        return conv['lo_evid']
    else:
        return 0

def c50_to_ev(c50):
    return measure_to_ev(C50_CONV, c50)

def ki_to_ev(ki):
    return measure_to_ev(KI_CONV, ki)

