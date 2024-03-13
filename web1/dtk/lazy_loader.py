
class LazyLoader(object):
    '''Base class for calculating properties as needed.

    Derived classes should:
    - set _kwargs to a list of valid keyword arguments to ctor
    - implement _XXX_loader() methods for each lazy-loaded property XXX
    - if you need non-keyword arguments and/or other special init processing,
      be sure to call super(YourClassName,self).__init__(**kwargs) from
      inside your __init__ method
    '''
    _kwargs=[]
    def __init__(self,**kwargs):
        for x in self._kwargs:
            if x in kwargs:
                setattr(self,x,kwargs.pop(x))
        if kwargs:
            raise TypeError("%s got unexpected keyword arguments '%s'"%(
                    self.__class__.__name__,
                    ' '.join(list(kwargs.keys())),
                    ))
    def __getattr__(self,attr):
        if attr.startswith('_') and attr.endswith('_loader'):
            # in py3, this goes into an infinite loop looking for:
            # _WHATEVER_loader
            # __WHATEVER_loader_loader
            # etc. So, don't do that.
            return None
        load_method_name = '_%s_loader' % attr
        try:
            func = getattr(self,load_method_name)
        except AttributeError:
            func = None
        if func is None:
            raise AttributeError("%s has no attribute '%s'"%(
                    self.__class__.__name__,
                    attr,
                    ))
        result = func()
        setattr(self,attr,result)
        return result

