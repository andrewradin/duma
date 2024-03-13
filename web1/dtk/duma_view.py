from browse.models import Workspace
import datetime
from django.utils.decorators import classonlymethod
from django.shortcuts import render
from django.http import HttpResponseRedirect

import logging
import six

################################################################################
# parameter parsing support
################################################################################
def from_string(some_type,some_string):
    if hasattr(some_type,'from_string'):
        return some_type.from_string(some_string)
    else:
        return some_type(some_string) # assume ctor works with strings

def to_string(some_instance):
    if hasattr(some_instance,'to_string'):
        return some_instance.to_string()
    else:
        return str(some_instance)

class boolean:
    trueish=['t','true','1']
    @classmethod
    def from_string(cls,string):
        return True if string.lower() in cls.trueish else False

def list_of(element_type,delim=','):
    class ListOf(list):
        @classmethod
        def from_string(cls,s):
            if not s:
                return cls([])
            return cls([from_string(element_type,x) for x in s.split(delim)])
        def to_string(self):
            return delim.join([to_string(x) for x in self])
    return ListOf

################################################################################
# Link creation
################################################################################
def qstr(_base_parms,**kwargs):
    '''Return a querystring for a URL.

    - String will be appropriately encoded
    - String is meant to apply modification to an existing set of query
      parameters, passed in as base_parms
    - kwargs specify replacement parameters and values; a value of None
      causes the parameter to be deleted (this is the difference between
      modify_dict and dict.update)
    - it will always return at least '?', so if used in an href without
      a path part, it will still blank out any existing querystring
    - it uses the above 'to_string' protocol for rendering values, allowing
      classes to serialize themselves differently from the default str()
      method, if desired
    '''
    opts = _base_parms.copy()
    from dtk.data import modify_dict
    # XXX rather than using modify_dict, maybe the convention should be
    # XXX to skip None values on output; this simplifies modifications
    # XXX to base_parms (although using pop() is a good idiom there)
    modify_dict(opts,kwargs)
    result = '?'
    if opts:
        # sort allows more consistent testing
        opts = {k:to_string(v) for k,v in sorted(six.iteritems(opts))}
        from django.utils.http import urlencode
        result += urlencode(opts,doseq=True)
    return result

################################################################################
# DumaView class
################################################################################
class DumaView(object):
    ##############################
    # default data members
    ##############################
    index_dropdown_stem='workflow'
    button_map = {}
    demo_buttons = []
    GET_parms = {}
    ##############################
    # view function factory (borrowed from Django View class)
    ##############################
    @classonlymethod
    def as_view(cls, **initkwargs):
        """
        Main entry point for a request-response process.
        """
        for key in initkwargs:
            if not hasattr(cls, key):
                raise TypeError("%s() received an invalid keyword %r. as_view "
                            "only accepts arguments that are already "
                            "attributes of the class." % (cls.__name__, key))
        def view(request, *args, **kwargs):
            self = cls(**initkwargs)
            self.request = request
            # convert to standard dict from QueryDict
            self.args = args
            self.kwargs = kwargs
            from dtk.timer import Timer
            self.timer = Timer()
            self.do_time_logging=True
            result = self.dispatch()
            elapsed = self.timer.get()
            if elapsed.total_seconds() > 3:
                self.timelog('page load time')
            return result
        view.view_class = cls
        view.view_initkwargs = initkwargs
        from functools import update_wrapper
        # take name and docstring from class
        update_wrapper(view, cls, updated=())
        # and possible attributes set by decorators
        # like csrf_exempt from dispatch
        update_wrapper(view, cls.dispatch, assigned=())
        return view
    ##############################
    # service routines
    ##############################
    def here_url(self,**kwargs):
        '''Return the url of this page, possibly with a modified querystring.
        '''
        return self.request.path+qstr(self.base_qparms,**kwargs)
    def url_builder_factory(self,show):
        '''Return a url_builder function that opens a collapse section.

        This is useful when a collapse section holds a dtk.table, and
        you want that section open when you sort a column.
        '''
        def url_builder(**kwargs):
            return self.here_url(show=show,**kwargs)
        return url_builder
    def username(self):
        return self.request.user.username
    @classmethod
    def user_is_demo(cls, user):
        return not (user.is_staff or user.groups.filter(name='consultant').exists())
    def is_demo(self):
        return self.user_is_demo(self.request.user)
    def in_group(self,group):
        return self.request.user.groups.filter(name=group).exists()
    def drugname(self):
        # assumes self.wsa is set
        return self.wsa.get_name(self.is_demo())
    def context_alias(self,**kwargs):
        '''Set values in context and as data members simultaneously.

        This is useful when the value is a structure, where the data member
        name functions as an alias to access the same structure in the context.
        It will not work as expected if the data member is subsequently on the
        left side of an assignment -- this will just break the aliasing.
        '''
        for name,value in six.iteritems(kwargs):
            setattr(self,name,value)
            self.context[name] = value
    def message(self,text):
        from django.contrib import messages
        messages.add_message(self.request, messages.INFO, text)
    def log(self,msg,*args):
        logger = logging.getLogger(self.__class__.__name__)
        logger.info(msg,*args)
    def timelog(self,msg,*args):
        if self.do_time_logging:
            self.log('%s '+msg,str(self.timer.get()),*args)
    def tsv_response(self,spec,rows,filename):
        '''Return an HttpResponse object for downloading a tsv file.

        The file is formatted on the fly, as in print_engine():
        rows - an array of objects corresponding to the tsv rows
        spec - an array of tsv column specs; each spec is a tuple with
           the column header string in the first element, and a lambda
           in the second element that takes an element from rows as a
           parameter and returns a string with that column's value
        filename - a default filename passed to the browser
        '''
        if self.is_demo():
            rows = [] # don't send any data in demo mode
        from django.http import HttpResponse
        header = [x[0] for x in spec]
        tsv_data = ''
        for row in [header]+[
                [s[1](x) for s in spec]
                for x in rows
                ]:
            tsv_data += '\t'.join(row)+'\n'
        response = HttpResponse(tsv_data,content_type='text/tsv')
        response['Content-Disposition'] \
                = f"attachment; filename={filename}"
        return response
    ##############################
    # common view arg handlers (can be overridden)
    ##############################
    def handle_ws_id_arg(self,ws_id):
        self.context_alias(ws=Workspace.objects.get(pk=ws_id))
        from runner.process_info import JobCrossChecker,JobInfo
        self.jcc=JobCrossChecker()
        self.context['job_cross_checker'] = self.jcc
    def handle_wsa_id_arg(self,wsa_id):
        from browse.models import WsAnnotation
        self.context_alias(wsa = WsAnnotation.objects.get(pk=wsa_id))
    def handle_elec_id_arg(self,elec_id):
        self.elec_id = int(elec_id)
    def handle_prot_id_arg(self,prot_id):
        from browse.models import Protein
        self.prot_id = prot_id
        p=Protein.get_canonical_of_uniprot(prot_id)
        self.context_alias(
                protein=p,
                protein_label=prot_id if p.uniprot == prot_id
                        else '%s (for %s)'%(p.uniprot,prot_id),
                )
    def handle_tissue_id_arg(self,tissue_id):
        from browse.models import Tissue
        self.context_alias(tissue=Tissue.objects.get(pk=tissue_id))
    def handle_scoreset_id_arg(self,scoreset_id):
        from browse.models import ScoreSet
        self.context_alias(scoreset=ScoreSet.objects.get(pk=scoreset_id))
    def handle_prescreen_id_arg(self,prescreen_id):
        from browse.models import Prescreen
        self.context_alias(prescreen=Prescreen.objects.get(pk=prescreen_id))
    def handle_job_id_arg(self,job_id):
        from runner.models import Process
        self.context_alias(job=Process.objects.get(pk=job_id))
    def handle_jobname_arg(self,jobname):
        self.context_alias(jobname=jobname)
    def handle_dtc_arg(self,dtc):
        self.context_alias(dtc=dtc)
    def handle_copy_job_arg(self,job_id):
        from runner.models import Process
        self.context_alias(copy_job = Process.objects.get(pk=job_id))
    def handle_ae_search_id_arg(self,search_id):
        from browse.models import AeSearch
        self.context_alias(search = AeSearch.objects.get(pk=search_id))
    def handle_kt_search_id_arg(self,search_id):
        from ktsearch.models import KtSearch
        self.context_alias(search = KtSearch.objects.get(pk=search_id))
    def handle_disease_vocab_arg(self,vocab_name):
        from dtk.vocab_match import DiseaseVocab
        vdefaults=self.ws.get_versioned_file_defaults()
        self.context_alias(
                disease_vocab=DiseaseVocab.get_instance(
                        vocab_name,
                        version_defaults=vdefaults,
                        )
                )
    ##############################
    # main dispatch flow
    ##############################
    def dispatch(self):
        # XXX maybe instead of all the plumbing for handing responses back
        # XXX through multiple layers of subroutines, we should have a special
        # XXX exception that any layer can throw which holds a response object,
        # XXX and then wrap all the logic below in a try block, and return any
        # XXX thrown response out of an except block
        # verify logged in
        if not self.request.user.is_authenticated:
            return HttpResponseRedirect(
                    '/account/login/?next='+self.request.get_full_path(),
                    )
        # do common setup
        self.initialize_context()
        self.handle_view_args()
        self.handle_GET_parms()
        response = self.custom_setup()
        if response:
            return response
        # handle POST
        if self.request.method == 'POST':
            response = self.handle_post()
            if response:
                return response
            # else there's a form error; continue to re-display
            # page for user correction
        self.initialize_remaining_forms()
        response = self.custom_context()
        if response:
            return response
        return render(self.request,self.template_name,self.context)
    ##############################
    # empty dispatch hooks for derived class overrides
    ##############################
    def custom_setup(self): return
    def custom_context(self): return
    ##############################
    # dispatch flow subroutines
    ##############################
    def initialize_context(self):
        # set up all context variables needed by base template
        from path_helper import PathHelper
        self.context = {
            'view':self,
            'function_root': self.index_dropdown_stem,
            'now':datetime.datetime.now(),
            'ph':PathHelper,
            }
    def handle_view_args(self):
        # assure all args are kwargs, and invoke a handler method for each
        assert not self.args,'DumaView only supports named arguments'
        for key in self.kwargs:
            handler = getattr(self,'handle_%s_arg'%key)
            handler(self.kwargs[key])
    def handle_GET_parms(self):
        # XXX maybe, we should have a consistent process that combines
        # XXX this approach with the view args approach; if the key exists
        # XXX in a hash, just add it to self; else, expect a handler function
        #
        # XXX and/or, maybe there should be a global list of common parms
        # XXX that get combined with the view-specific ones in GET_parms
        base_qparms = {}
        for k,v in six.iteritems(self.request.GET):
            # profile key is used by the profiling middleware, ignore it.
            if k == 'profile':
                continue
            if k not in self.GET_parms:
                raise TypeError(
                        "%s() received an invalid querystring parm %r."
                        " Only keys defined in GET_parms (%s) are accepted."
                        % (
                                self.__class__.__name__,
                                k,
                                ', '.join(list(self.GET_parms.keys())),
                                )
                        )
            parm_type,default = self.GET_parms[k]
            if parm_type:
                v = from_string(parm_type,v)
            setattr(self,k,v)
            base_qparms[k] = v
        # self.base_qparms holds an independent copy of all parameters actually
        # present in the GET hash, for the purpose of reconstructing a URL
        # - it's set up here so any special types (list_of, etc.) are already
        #   converted
        # - it's deep copied so that it's unaffected by changes to the data
        #   members
        import copy
        self.base_qparms = copy.deepcopy(base_qparms)
        # now, supply defaults for any parameters not specified in the URL
        for k in self.GET_parms:
            if not hasattr(self,k):
                parm_type,default = self.GET_parms[k]
                if parm_type and isinstance(default,str):
                    # If default is a string, and there's a cast that's
                    # applied to param strings, apply it to the default
                    # as well. This allows empty list_of() params to be
                    # built with the correct type.
                    default = from_string(parm_type,default)
                setattr(self,k,default)
    # Form handling in derived class:
    # - define button_map with a key for each postback button, with the
    #   corresponding value being a (possibly empty) list of the forms
    #   that button accesses.  Both the _btn and _form suffixes are omitted.
    # - for each form, a make_<form_key>_form(data) method must be provided
    #   to return the form
    # - for each button, a <button_key>_post_valid() method must be provided
    #   to implement the button's action
    # - 'shortcut' buttons provide a one-click way to specify some form
    #   field values.
    #   - To support shortcut buttons, add an 'overrides' keyword parameter
    #     to one of your post_valid methods (typically, pick the one with
    #     the closest function to that of the shortcut button).
    #     - The overrides parameter should have a default value that results
    #       in the original processing path for the posted data.
    #     - Other values of overrides should drive logic that implements the
    #       various shortcut cases.
    #   - Then, in custom_setup, call add_shortcut_handler for each shortcut
    #     button instead of writing separate post_valid methods for each one.
    #     This allows the number of buttons to be determined dynamically. You
    #     can associate different values for the overrides parameter with each
    #     call.
    #   - You may also need to pass data in the context to instruct the
    #     template how and where to render these buttons.
    #   - You may also need to modify the button map to associate form data
    #     with these buttons. See ktsearch ResolveView for an example.
    def add_shortcut_handler(self,btn_name,base_handler,overrides):
        setattr(
                self,
                btn_name+'_post_valid',
                lambda: base_handler(overrides=overrides),
                )
    def _get_form_keys(self,stems=None):
        if stems is None:
            import itertools
            stems = itertools.chain(*list(self.button_map.values()))
        return [x+'_form' for x in stems]
    def handle_post(self):
        valid_handler = None
        for k in self.button_map:
            if k+'_btn' in self.request.POST:
                if self.is_demo() and k not in self.demo_buttons:
                    self.message('Action disallowed in demo mode')
                    return HttpResponseRedirect('#')
                post_forms = self._get_form_keys(self.button_map[k])
                valid_handler = getattr(self,k+'_post_valid')
                break
        if not valid_handler:
            raise NotImplementedError(
                    'unexpected POST: %s. Supported buttons: %s' %(
                        repr(self.request.POST),
                        ', '.join(list(self.button_map.keys()))
                        )
                    )
        invalid = []
        for key in post_forms:
            form = self._instantiate_form(key,self.request.POST)
            if not form.is_valid():
                invalid.append((key, form.errors))
        if invalid:
            self.log('error in forms: %s',str(invalid))
        else:
            return valid_handler()
    def initialize_remaining_forms(self):
        # instantiate default versions of all forms that aren't
        # already set up; in GET handling, this will typically be
        # all forms; in POST handling it will be all forms not
        # involved in the POST (this will only happen in case of
        # an error in one of the POST forms)
        for key in self._get_form_keys():
            if key not in self.context:
                self._instantiate_form(key,None)
    def _instantiate_form(self,key,data):
        factory = getattr(self,'make_%s'%key)
        form = factory(data)
        self.context[key] = form
        setattr(self,key,form)
        return form

