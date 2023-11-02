from django import forms

class ElectionForm(forms.Form):
    due_date = forms.DateTimeField(label="Due Date (YYYY-MM-DD):")
    def __init__(self, ws, e_id, demo, flavor, *args, **kwargs):
        super(ElectionForm,self).__init__(*args, **kwargs)
        from browse.models import Election,ElectionFlavor,WsAnnotation,Vote
        if e_id:
            # get users and drugs currently active
            e = Election.objects.get(id=e_id)
            self.instance = e
            self.fields['due_date'].initial = e.due
            vote_qs = e.vote_set.filter(disabled=0)
            assert e.ws_id == ws.id
            active_users = vote_qs.values_list('reviewer',flat=True).distinct()
            active_drugs = vote_qs.values_list('drug',flat=True).distinct()
            flavor = e.flavor # ignore passed-in value if election exists
        self.flavor = ElectionFlavor(flavor)
        # populate reviewers list
        from django.contrib.auth.models import Group
        grp = Group.objects.get(name="reviewers")
        for u in grp.user_set.order_by('username'):
            if e_id:
                initial = u.username in active_users
            else:
                initial = True
            self.fields['u_'+str(u.id)] = forms.BooleanField(
                                                initial=initial,
                                                label=u.username,
                                                required=False,
                                                )
        # populate drugs list
        enum=WsAnnotation.indication_vals
        from django.db.models import Q
        # set staged depending on flavor
        staged = Q(indication=self.flavor.input)
        if self.flavor.filter_dups:
            allocated = Vote.objects.filter(
                    election__ws=ws,
                    election__flavor=self.flavor.flavor_string,
                    disabled=False,
                    ).values_list('drug_id',flat=True).distinct()
            staged = staged & ~Q(id__in=allocated)
        ws_qs = WsAnnotation.objects.filter(ws=ws)
        if e_id:
            # note 'previous' is different from 'active_drugs' above
            # because it includes disabled drugs
            previous=Q(
                id__in=e.vote_set.values_list('drug',flat=True).distinct()
                )
            qs = ws_qs.filter(staged|previous)
        else:
            qs = ws_qs.filter(staged)
        wsa_list=list(qs)
        wsa_list.sort(key=lambda x:x.agent.canonical)
        self.wsa_list = wsa_list
        for wsa in wsa_list:
            if e_id:
                initial = wsa.id in active_drugs
            else:
                initial = True
            self.fields['d_'+str(wsa.id)] = forms.BooleanField(
                                                initial=initial,
                                                label=wsa.get_name(demo),
                                                required=False,
                                                )
    def groups(self):
        # returned structure is:
        # [
        #  ["group label",[ (label,html), ... ]],
        #  ...
        #  [label,[ ("",html) ]],
        #  ...
        # ]
        groups = [
            ["Reviewers:",[]],
            ["Drugs:",[]],
            ]
        for name,field in self.fields.items():
            info = (field.label+':',self[name])
            if name.startswith("u_"):
                groups[0][1].append(info)
            elif name.startswith("d_"):
                groups[1][1].append(info)
            else:
                groups.append([info[0][:-1],[("",info[1])]])
        return groups

