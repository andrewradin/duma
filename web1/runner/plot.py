


def slotify_timeline(procs):
    # Orders procs by time and lazily assigns to 'slots' on the timeline.
    # These don't have any real meaning, but help to visualize.
    q = []
    for p in procs:
        q.append(('start', p.started, p))
        q.append(('end', p.completed, p))
    
    slots = 0
    free_slots = []
    out = {}
    for ev, time, p in sorted(q, key=lambda x: x[1]):
        if ev == 'start':
            if not free_slots:
                slot = slots
                slots += 1
            else:
                slot = free_slots.pop()
            
            out[p.id] = slot
        else:
            slot = out[p.id]
            free_slots.append(slot)
    
    # Plotly tries to interpolate the vertical axis if it's just numbers, add some text.
    out = {k: f'slot{v}' for k, v in out.items()}
    return out

def make_segment_dicts(proc, slot):
    waits = proc.processwait_set.all()

    now = proc.started
    from datetime import timedelta
    segments = []
    for wait in sorted(waits, key=lambda x: x.start):
        segments.append(dict(start=now, end=wait.start, slot=slot, name=proc.role, short_name=proc.name))
        now = wait.start + wait.duration
    
    segments.append(dict(start=now, end=proc.completed, slot=slot, name=proc.role, short_name=proc.name))
    return segments

def plot_timeline(procs):
    slots = slotify_timeline(procs)
    data = []
    for i, proc in enumerate(procs):
        data.extend(make_segment_dicts(proc, slots[proc.id]))
    
    earliest = min(x['start'] for x in data)
    from datetime import timedelta
    def datestr(x):
        return x.strftime("%Y-%m-%d %H:%M:%S")
    for entry in data:
        duration = (entry['end'] - entry['start']) / timedelta(minutes=1)
        str_duration = f'{duration:.1f}m'
        entry['hover'] = f'{entry["name"]} {str_duration}'
        if duration < 1:
            entry['text'] = ''
        elif duration < 5:
            entry['text'] = str_duration
        else:
            entry['text'] = f'{entry["name"]} {str_duration}'
        entry['start'] = datestr(entry['start'])
        entry['end'] = datestr(entry['end'])


    import plotly.express as px
    from dtk.plot import PlotlyPlot
    fig = px.timeline(data, x_start="start", x_end="end", y="slot", color="short_name", text='text', hover_name='hover')
    fig = fig.to_dict()
    fig['layout']['width'] = 1920
    fig['layout']['height'] = 1080
    return PlotlyPlot(data=fig['data'], layout=fig['layout'])