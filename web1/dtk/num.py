# For serious number crunching, use numpy.
# For quick-and-dirty implementations of basic algorithms, look here.

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_non_zero_min(l):
    smallest = None
    for c in l:
        if c == 0.:
            continue
        if smallest is None:
            smallest = c
            continue
        smallest = min([smallest,c])
    return smallest

def is_single_value(l):
    '''are all elements in the vector the same?'''
    return all([x == l[0] for x in l[1:]])

def map_nan(v,replace_with):
    import math
    if math.isnan(v):
        return replace_with
    return v

def avg(l):
    # If the input vector is something like [1.00000000000001]*10000,
    # floating point rounding error will cause the sum divided by the
    # length to not equal the constant value in the vector.  This causes
    # all kinds of trouble, so check for it first.
    if is_single_value(l):
        return l[0]
    return sum(l)/float(len(l))

def median(l):
    tmp = sorted(l)
    ll = len(l)
    mid = ll//2
    if ll % 2:
        return tmp[mid]
    return avg(tmp[mid-1:mid+1])

def avg_sd(l):
    '''Return (avg,sd)'''
    av=avg(l)
    diff_sq=[pow(x-av,2) for x in l]
    sd=pow(sum(diff_sq)/len(diff_sq),0.5)
    return av,sd

def sigma(x):
    '''Map any real x into the range [0,1] in a smooth sigmoid curve.

    -2 maps to ~.12
    -1 maps to ~.27
    0 maps to .5
    1 maps to ~,73
    2 maps to ~.88
    '''
    import numpy as np 
    import warnings
    with warnings.catch_warnings():
        # exp regularly overflows here, which will (correctly) end up
        # with a result of 0.  We don't need to hear about it all the time.
        warnings.simplefilter('ignore')
        try:
            to_ret = 1.0/(1.0+np.exp(-x))
        except OverflowError:
            to_ret = 0.0
    return to_ret

def corr(first_or_both,second=None,method='unspecified'):
    '''return correlation coefficient between two vectors.

    - 'method' specifies which algorithm to use
    - data may be passed as [(x,y),...] as the first parameter,
      or [x,...] as the first and [y,...] as the second
    Note that if either vector is constant, this function returns
    NaN; if this is not acceptable, you can use safe_corr() below
    to get zero instead, or roll your own wrapper to handle the
    nan some other way.
    '''
    import scipy.stats
    if method == 'spearman':
        if second is not None:
            raw = scipy.stats.spearmanr(first_or_both,second)
        else:
            raw = scipy.stats.spearmanr(first_or_both)
    elif method == 'pearson':
        if second is not None:
            raw = scipy.stats.pearsonr(first_or_both,second)
        else:
            raw = scipy.stats.pearsonr(
                    [x[0] for x in first_or_both],
                    [x[1] for x in first_or_both],
                    )
    else:
        raise NotImplementedError("no '%s' method"%method)
    return raw[0]

def safe_corr(*args,**kwargs):
    return map_nan(corr(*args,**kwargs),0.0)

