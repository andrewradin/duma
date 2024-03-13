#-------------------------------------------------------------------------------
# Introspection tools
#-------------------------------------------------------------------------------
class SubclassRegistry(object):
    '''A base class for dynamic subclass lookups.

    The model is a class hierarchy like
      (thing1,thing2,...)->TypeOfThing->SubclassRegistry
    TypeOfThing will inherit all the methods below for dynamically finding
    thingx classes based on their names.
    '''
    @classmethod
    def get_subclasses(cls, sort_output=True):
        result = []
        import sys
        mod = sys.modules[cls.__module__]
        for k,v in mod.__dict__.items():
            try:
                if not issubclass(v,cls):
                    continue
                if v == cls:
                    continue
                # The following allows a TypeOfThing class to define an
                # intermediates list, which will allow intermediate layers
                # in a deeper hierarchy to be ignored.  Each class in an
                # intermediate layer must be listed.
                try:
                    if v in cls.intermediates:
                        continue
                except AttributeError:
                    pass
                result.append( (k,v) )
            except TypeError:
                continue
        if sort_output:
            result.sort(key=lambda x:x[0])
        return result
    @classmethod
    def get_all_names(cls):
        return [x[0] for x in cls.get_subclasses()]
    @classmethod
    def lookup(cls,name):
        for k,v in cls.get_subclasses():
            if k == name:
                return v
        raise KeyError("'%s' does not exist" % name)
    @classmethod
    def pretty_print_list(cls):
        print(cls.__name__,'list:')
        for name,atype in cls.get_subclasses():
            print('  ',name+' - '+atype.description().rstrip())
        print()
    @classmethod
    def description(self):
        return self.__doc__ or '(no description supplied)'

