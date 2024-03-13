def fmt_time(dt,
        #fmt = "%b %d, %Y, %I:%M %p",
        fmt = "%Y-%b-%d %H:%M",
        ):
    if dt is None:
        return ''
    from django.utils.timezone import is_aware,make_naive,get_current_timezone
    if is_aware(dt):
        # convert to local timezone
        dt = make_naive(dt,timezone=get_current_timezone())
    return dt.strftime(fmt)

def fmt_delta(end,start):
    if start and end:
        d=end-start
        return fmt_timedelta(d)
    return 'N/A'

def fmt_timedelta(td):
    secs = td.total_seconds()
    if secs <= 0:
        return 0
    return '%d:%02d' % divmod(secs,60)

def fmt_interval(delta,unit=None):
    days = delta.days
    if not unit:
        mag = abs(days)
        if mag < 30:
            unit=('week',7)
        else:
            unit=('month',365/12)
    count = round(days/unit[1])
    if count == 0:
        return f'this {unit[0]}'
    neg = count < 1
    mag = abs(count)
    fmt = f'{mag} {unit[0]}'
    if mag > 1:
        fmt += 's'
    if neg:
        return f'{fmt} ago'
    return f'in {fmt}'

def fmt_size(size):
    units='KMGTP'
    idx=0
    K=1024
    if size < K:
        return "%dB"%size
    size /= K
    while size > K*1.2:
        size /= K
        idx += 1
    return "%.1f%s" % (size,units[idx])

def fmt_english_list(l):
    if len(l) > 1:
        return ', '.join(l[:-1])+' and '+l[-1]
    return l[0]

def limit(s,maxlen=50,ellipses='...',end=False):
    if len(s) > maxlen:
        if end:
            return ellipses+s[-maxlen-len(ellipses):]
        else:
            return s[:maxlen-len(ellipses)]+ellipses
    return s

def diffstr(v1,v2,minmatch=0,keep=None,repl='...',autojunk=False):
    limit = None if keep is None else 2*keep+len(repl)
    import difflib
    s = difflib.SequenceMatcher(autojunk=autojunk)
    s.set_seqs(v1,v2)
    m = [x for x in s.get_matching_blocks()]
    if minmatch:
        m = [x for x in m if x.size==0 or x.size>=minmatch]
    diff = []
    if m and (m[0].a or m[0].b):
        # there's not a common block at the beginning
        common = ''
        deleted = v1[0:m[0].a]
        added = v2[0:m[0].b]
        diff.append( (common,deleted,added) )
    for i in range(0,len(m)-1):
        common = v1[m[i].a:m[i].a+m[i].size]
        if limit and len(common) > limit:
            common = common[0:keep] + repl + common[-keep:]
        deleted = v1[m[i].a+m[i].size:m[i+1].a]
        added = v2[m[i].b+m[i].size:m[i+1].b]
        diff.append( (common,deleted,added) )
    return diff

def diff_fmt(v1,v2,del_fmt,add_fmt,**kwargs):
    l = diffstr(v1,v2,**kwargs)
    return ''.join(
            p+del_fmt(r)+add_fmt(a)
            for p,r,a in l
            )

def uni_combine(s,combining_char):
    '''Return a string where each char in s is preceeded by combining_char.

    This can be used with combining_char values from the unicode
    Combining Diacritical Marks block.
    '''
    # XXX Note that although the unicode standard is that the combining
    # XXX character should follow the main character, this isn't configured
    # XXX correctly in all fonts. In particular, the default Ubuntu terminal
    # XXX window font Monospace Regular wants it in the opposite order. It's
    # XXX fixed in the very similar DejaVu Sans Mono Book font, but in both
    # XXX fonts the gnome terminal has issues rendering a combining char
    # XXX in the first column of the screen.
    # XXX
    # XXX This is deprecated for being overly tempermental, but is left
    # XXX in place to capture research thus far.
    return ''.join(x+combining_char for x in s)

def uni_underline(s): return uni_combine(s,chr(0x332))
def uni_strikethru(s): return uni_combine(s,chr(0x336))

def split_multi_lines(rows):
    result = []
    for row in rows:
        if any('\n' in col for col in row):
            stack = [x.split('\n') for x in row]
            out_count = max(len(x) for x in stack)
            for col in stack:
                missing = out_count - len(col)
                if missing:
                    col += ['']*missing
            for i in range(out_count):
                result.append([x[i] for x in stack])
        else:
            result.append(row)
    return result

def wrap(rows,colnum,width):
    for row in rows:
        s = row[colnum]
        done = ''
        while len(s) > width:
            try:
                split_point = s[:width].rindex(' ')
                done += (s[:split_point] + '\n')
            except ValueError:
                split_point = width
                done += (s[:split_point] + '\n-')
            s = s[split_point:]
        row[colnum] = done + s

def ljustify(rows,colnum):
    max_width = max(len(row[colnum]) for row in rows)
    for row in rows:
        missing = max_width - len(row[colnum])
        if missing:
            row[colnum] += (' '*missing)

def print_table(table,div=u'|',header_delim=None):
    widths = [
            max([len(row[i]) for row in table])
            for i in range(len(table[0]))
            ]
    for row in table:
        row = [
                ' '*(width-len(val))+val
                for val,width in zip(row,widths)
                ]
        print(div.join(row))
        if header_delim:
            print(div.join([header_delim*x for x in widths]))
            header_delim = None

def print_engine(columns,rows):
    # columns is a list of pairs [(label,lambda),...], where:
    #   - label is a column header
    #   - lambda is a function that takes a row and returns a string value
    # rows is a list of items to be passed to the lambda functions
    rows = [[x[0] for x in columns]]+[
            [x[1](row) for x in columns]
            for row in rows
            ]
    print_table(rows)

def indent(lead,lines):
    return [lead+x for x in lines]

def dict_replace(repls, text):
    """Replaces each instance of 'key' with 'value' in text, for all keys & values in repls."""
    import re
    to_find = sorted([re.escape(k) for k in repls.keys()])
    regex = re.compile('|'.join(to_find))
    repl_func = lambda x: repls[x.group(0)]
    return regex.sub(repl_func, text)

class FormatDictDiff:
    key_sep = ':'
    rm_prefix = '~'
    diff_sep = '>>'
    def fmt_add(self,key,new_val):
        return key+self.key_sep+str(new_val)
    def fmt_remove(self,key,old_val):
        return self.rm_prefix+key
    def fmt_modify(self,key,old_val,new_val):
        return key+self.key_sep+str(old_val)+self.diff_sep+str(new_val)
    def fmt_list(self,change_list):
        if not change_list:
            return ''
        return " + ("+", ".join(change_list)+")"

def compare_dict(label,base_d,target_d,fmt=FormatDictDiff()):
    'Return a string representation of changes to a dict.'
    all_keys = sorted(set(base_d.keys())|set(target_d.keys()))
    diffs = []
    for k in all_keys:
        if k not in base_d:
            diffs.append(fmt.fmt_add(k,target_d[k]))
        elif k not in target_d:
            diffs.append(fmt.fmt_remove(k,base_d[k]))
        elif base_d[k] != target_d[k]:
            diffs.append(fmt.fmt_modify(k,base_d[k],target_d[k]))
    return label + fmt.fmt_list([x for x in diffs if x])

def compare_set(base_s,target_s,include_unchanged=False):
    'Return a string representation of changes to a set.'
    same = base_s & target_s
    added = target_s - base_s
    removed = base_s - target_s
    result = ''
    if include_unchanged:
        result += ','.join(str(x) for x in sorted(same))
    if added:
        result += '+'+'+'.join(str(x) for x in sorted(added))
    if removed:
        result += '-'+'-'.join(str(x) for x in sorted(removed))
    return result

def setlist_compare_data(base_l,target_l):
    '''Return a triple describing changes to a set of sets.

    Results are a 3-tuple (adds,drops,modifies) where
    - adds and drops are lists of sets
    - modifies is a list of strings describing individual set modifications

    Sets from base_l and target_l are compared with their closest match,
    not by index order.
    '''
    base_l = [set(x) for x in base_l]
    target_l = [set(x) for x in target_l]
    # construct all possible pairings and score them
    from dtk.similarity import calc_jaccard
    pairs = []
    for i in range(len(base_l)):
        for j in range(len(target_l)):
            pairs.append((calc_jaccard(base_l[i],target_l[j]),i,j))
    pairs.sort(key=lambda x:(-x[0],x[1],x[2]))
    # process each pairing, and then remove all other pairings involving
    # either of the two processed sets
    modified = []
    while (pairs):
        c,i,j = pairs[0]
        if c == 1:
            pass # do nothing for identical sets
        elif c == 0:
            # everything remaining is an add or a drop
            break
        else:
            modified.append(compare_set(base_l[i],target_l[j],True))
        # remove matching elements
        pairs = [
                x for x in pairs[1:]
                if x[1] != pairs[0][1] and x[2] != pairs[0][2]
                ]
    # construct lists of all remaining unpaired sets
    add_idxs = set(x[2] for x in pairs)
    drop_idxs = set(x[1] for x in pairs)
    return (
            [target_l[x] for x in add_idxs],
            [base_l[x] for x in drop_idxs],
            modified,
            )

def compare_setlist(base_l,target_l):
    'Return a string representation of changes to a set of sets.'
    adds,drops,modifies = setlist_compare_data(base_l,target_l)
    result = ''
    for item in modifies:
        result += ',{'+item+'}'
    for item in adds:
        result += '+{'+','.join(sorted(item))+'}'
    for item in drops:
        result += '-{'+','.join(sorted(item))+'}'
    return result

def compare_wzs_settings(label,base_d,target_d):
    '''Return a string representation of changes to WZS settings.

    This is like compare_dict, but outputs HTML, and handles
    unusual comparisons specific to WZS parameters.
    '''
    class MyFormat(FormatDictDiff):
        key_sep = ': '
        diff_sep = ' >> '
        def fmt_modify(self,key,old_val,new_val):
            if key == 'auto_constraints':
                # This is a very special case. The values are json
                # representations of lists of lists (which actually
                # act like sets, i.e. unordered). Convert back from
                # json and use setlist compare tools to get a more
                # informative diff.
                import json
                adds,drops,modifies = setlist_compare_data(
                        json.loads(old_val),
                        json.loads(new_val),
                        )
                if not adds and not drops and not modifies:
                    # the formatter won't get called unless there's
                    # a textual change, but that might not translate
                    # to a logical change
                    return ''
                result = ''
                for item in modifies:
                    item = item.replace('-','<b> - </b>')
                    item = item.replace('+','<b> + </b>')
                    result += '<li>{'+item+'}</li>'
                for item in adds:
                    result += '<li>+{'+','.join(sorted(item))+'}</li>'
                for item in drops:
                    result += '<li>-{'+','.join(sorted(item))+'}</li>'
                return key+':<ul>'+result+'</ul>'
            return super().fmt_modify(key,old_val,new_val)
        def fmt_list(self,change_list):
            # strip any format-only change returned as '' by an
            # overridden formatter
            change_list = [x for x in change_list if x]
            if not change_list:
                return ''
            return " + <ul><li>"+"</li><li>".join(change_list)+"</li></ul>"
    from django.utils.safestring import mark_safe
    return mark_safe(compare_dict(label,base_d,target_d,fmt=MyFormat()))

def compare_refresh_wf_settings(label,base_d,target_d):
    '''Return a string representation of changes to Refresh WF settings.

    This is like compare_dict, but handles unusual comparisons specific
    to Refresh Workflow parameters.
    '''
    class MyFormat(FormatDictDiff):
        def fmt_modify(self,key,old_val,new_val):
            if key == 'refresh_parts':
                try:
                    old_val = set(int(x) for x in old_val)
                    new_val = set(int(x) for x in new_val)
                except ValueError:
                    old_val = set(str(x) for x in old_val)
                    new_val = set(str(x) for x in new_val)
                if old_val == new_val:
                    return None
                return key + self.key_sep + compare_set(old_val,new_val)
            return super().fmt_modify(key,old_val,new_val)
    return compare_dict(label,base_d,target_d,fmt=MyFormat())

