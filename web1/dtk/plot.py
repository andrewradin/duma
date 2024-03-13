from __future__ import print_function
# Plotting convenience wrappers - the point of these functions is to hide
# plotting details from most of the code, exposing simplified interfaces
# for specific use cases that enforce a consistent look and interaction style.
#
# To find details of the underlying packages in order to modify or extend
# these functions, see:
# http://matplotlib.org/api/pyplot_api.html
# https://github.com/plotly/plotly.js
# https://github.com/plotly/plotly.py
# https://plot.ly/python/

from builtins import str
from builtins import zip
from builtins import range
from builtins import object
import logging
logger = logging.getLogger(__name__)
import os

# python3 can't directly json'ify numpy ints (python2 can supposedly
# only do one of np.int32 or np.int64 depending on sizeof(long))
# https://bugs.python.org/issue24313
def convert(x):
    import numpy as np
    if isinstance(x, np.int64):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, set):
        return list(x)
    raise TypeError("Couldn't convert %s" %  type(x))

def rand_jitter(l, jitter = 0.01):
    import numpy as np
    stdev = jitter*(max(l)-min(l))
    return list(np.array(l) + np.random.randn(len(l)) * stdev)

def mpl_non_interactive():
    try:
        import matplotlib
        matplotlib.use('Agg')
        # at some point, this became necessary to avoid having matplotlib
        # require python-tk; call before creating figures
    except ImportError:
        pass

class Color(object):
    default='#1A84C6' #blue
    default_light='#5AB4F6' #blue (slightly lighter)
    highlight='#F48725' #orange
    highlight_light='#FFB765' #orange (slightly lighter)
    highlight2='#009231' #green
    highlight3='#c89a00' #gold
    highlight4='#a10058' #magenta?
    negative='#7f7f7f' #gold
    ordered_colors = [default, highlight, highlight2, highlight3, highlight4, negative]
    # detailed here https://community.plot.ly/t/plotly-colours-list/11730/2
    plotly_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
    ]

    extended_plotly_colors = [default,highlight,
    'rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
    'rgb(148, 103, 189)', 'rgb(140, 86, 75)', 'rgb(227, 119, 194)','rgb(127,205,187)',
    'rgb(128,0,38)','rgb(5,10,172)','rgb(106,137,247)','rgb(190,190,190)',highlight2,
    'rgb(44,255,150)','rgb(255,255,255)','rgb(150,0,90)','rgb(0,0,200)','#8c564b',
    'rgb(255,0,0)','rgb(0,0,130)','rgb(255,234,0)']
# plus some more I added b/c we needed more for some pie charts
    many_pie_slice_colors= extended_plotly_colors+[
    'rgb(255,215,180)','rgb(128,128,0)',highlight4, 'rgb(255,0,255)',
    'rgb(210,245,60)'
    ]


# a common function for displaying the direction of a DPI
def dpi_arrow(direction):
    direction = int(direction)
    from dtk.html import glyph_icon
    if direction > 0:
        return glyph_icon('arrow-up',color=Color.default)
    if direction < 0:
        return glyph_icon('arrow-down',color=Color.highlight)
    return glyph_icon('arrow-right',color='black')

def validate_plotdir(plotdir,content_caches):
    '''Make sure plotdir exists and holds only valid plots.

    Any dynamic information that may affect the validity of
    a cached plot (like the current contents of a kt set)
    should be passed in content_caches. content_caches is
    a list of pairs of strings like

    (label_for_information, canonical_encoding_of_value).

    label_for_information should encode the parameter value
    (e.g. the name of a drugset) and a fixed string to
    disambiguate which parameter this is (in case the view
    might be affected by multiple parameters that need to
    be checked in this way).

    This will be checked against a stored copy that this
    function maintains within the directory. If the content
    has changed, all the plots in the directory are removed.

    When this function returns, plotdir is guaranteed to exist,
    and any content found within it is guaranteed to be valid.
    '''
    valid = True
    import os
    for name,chkstr in content_caches:
        chkpath = os.path.join(plotdir,name)
        try:
            got = open(chkpath).read().strip()
            valid = (got == chkstr)
        except IOError:
            valid = False
        if not valid:
            break
    if not valid:
        from dtk.files import remove_tree_if_present
        remove_tree_if_present(plotdir)
    from path_helper import make_directory
    make_directory(plotdir)
    for name,chkstr in content_caches:
        chkpath = os.path.join(plotdir,name)
        open(chkpath,'w').write(chkstr)

# plot a series of lines to a png file using matplotlib
def plot_vectors(fn,vecset,
                legend_space=0.2,
                legend_above=False,
                **kwargs):
    import matplotlib.pyplot as plt
    plt.clf()
    fig = plt.gcf()
    ax = plt.gca()
    for label,vec in vecset:
        ax.plot(vec,
                label=str(label),
                **kwargs
                )
    if legend_space:
        # All coordinates below run from 0,0 in the lower left
        # to 1,1 in the upper right.  'loc' is the attachment
        # point on the legend box.
        box = ax.get_position()
        if legend_above:
            offset = box.height * legend_space
            ax.set_position([
                    box.x0,
                    box.y0,
                    box.width,
                    box.height - offset,
                    ])
            ax.legend(loc='lower center', bbox_to_anchor=(0.5,1))
        else:
            offset = box.width * legend_space
            ax.set_position([
                    box.x0,
                    box.y0,
                    box.width - offset,
                    box.height,
                    ])
            ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    plt.savefig(fn+'.png')

# generate a scatterplot to a png file using matplotlib
def scatterplot(xlabel,ylabel,vec,fn=None,refline=True):
    if fn is None:
        fn = '%s_vs_%s' % (xlabel,ylabel)
    import matplotlib.pyplot as plt
    plt.clf()
    vx = [x[0] for x in vec]
    vy = [x[1] for x in vec]
    plt.scatter(vx,vy)
    if refline:
        plt.plot(vx,vx)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fn+'.png')

# a wrapper for plotly annotations
def annotations(text,text2=None):
    '''return an annotation configuration.

    Supports one or two text boxes above the plotting area.
    '''
    defaults = dict(
            # The 'paper' settings put us in a 0-1 coordinate system
            # overlaying the entire plotting area
            xref='paper',
            yref='paper',
            # these next two align the bottoms of the annotations with
            # the top of the plot area
            y=1,
            yanchor='bottom',
            # an innocuous default border
            bordercolor='gray',
            borderpad=4,
            # no arrows from the annotation
            showarrow=False,
            )
    mid = 0.5
    margin = 0.03
    d1 = dict(defaults)
    if not text2:
        # single annotation in center
        d1.update(x=mid,text=text)
        return [d1]
    # two annotations, top left and right
    d1.update(x=mid-margin,xanchor='right',text=text)
    d2 = dict(defaults)
    d2.update(x=mid+margin,xanchor='left',text=text2)
    return [d1,d2]
# a wrapper for legends annotations
def fig_legend(text, y):
    '''return an annotation configuration.

    Supports one text box below the plot.
    '''
    return dict(
            # The 'paper' settings put us in a 0-1 coordinate system
            # overlaying the entire plotting area
            xref='paper',
            yref='paper',
            # these next two align the bottoms of the annotations with
            # the top of the plot area
            y=y,
            yanchor='top',
            # no arrows from the annotation
            showarrow=False,
            x=0.5,
            xanchor='center',
            text='<br>'.join(text)
            )

def add_jitter(vec,amt):
    if amt is True:
        # auto-scale
        amt = 0.001 * (max(vec)-min(vec))
    import random
    return [x+random.uniform(-amt,+amt) for x in vec]

# the Plotly-based replacement for scatterplot above
def scatter2d(x_label,y_label,points,
        refline=True,
        class_idx=None,
        classes=None,
        title=None,
        text=None, # for multi-line, separate with '<br>'
        textposition=None, #should text be displayed on the plot
        ids=None, # (click_type,list)
        width=None,
        height=None,
        logscale=False,
        annotations=None,
        jitter=False,
        linestyle='markers',
        bins=None, #If True then creates a 2d hist scatter, if other than bool assumes provided bins
        hist_opacity=1, # opacity for hist part of hist scatter
        ):
    import numpy as np
    color = Color()
    colors = color.ordered_colors[:]
    data = []
    id_lookup = []
    if points:
        x,y = list(zip(*points))
    else:
        x, y = [], []
    if jitter:
        x = add_jitter(x,jitter)
        y = add_jitter(y,jitter)
    if refline:
        domain = (min(x),max(x))
        data.append( dict(
                x=domain,
                y=domain,
                mode='lines',
                showlegend=False,
                hoverinfo='skip',
                ))
        id_lookup.append( list() )
    if classes:
        for i,cfg in enumerate(classes):
            kwargs = {
                    'name':cfg[0],
                    'x':[v for idx,v in zip(class_idx,x) if idx == i],
                    'y':[v for idx,v in zip(class_idx,y) if idx == i],
                    'mode':linestyle,
                    'marker':cfg[1],
                    'type': 'scattergl',
                    'hoverinfo':'x+y',
                    }
            if text:
                kwargs['text'] = [v
                            for idx,v in zip(class_idx,text)
                            if idx == i
                            ]
                kwargs['hoverinfo'] = 'x+y+text'
            if textposition:
                kwargs['mode']+='+text'
                kwargs['textposition'] = textposition
            if ids:
                id_lookup.append( [v
                            for idx,v in zip(class_idx,ids[1])
                            if idx == i
                            ] )
            data.append( dict(**kwargs) )
            if bins:
                if type(cfg[1]['color']) != list or len(set(cfg[1]['color'])) == 1:
                    hist_color = cfg[1]['color']
                else:
                    hist_color = colors.pop(0)
                append_scatter_histograms(
                        data,
                        kwargs['x'],
                        kwargs['y'],
                        bins,
                        hist_color,
                        hist_opacity,
                        )
    else:
        kwargs = {
                'x':x,
                'y':y,
                'mode':linestyle,
                'showlegend':False,
                'hoverinfo':'x+y',
                }
        if text:
            kwargs['text'] = text
            kwargs['hoverinfo'] = 'x+y+text'
        if textposition:
            kwargs['mode']+='+text'
            kwargs['textposition'] = textposition
        if ids:
            id_lookup.append( list(ids[1]) )
        data.append( dict( **kwargs ) )
        if bins:
            append_scatter_histograms(
                    data,
                    x,
                    y,
                    bins,
                    colors.pop(0),
                    hist_opacity,
                    )
    layout={
            'hovermode':'closest',
            'xaxis':{'title':x_label},
            'yaxis':{'title':y_label},
            }
    if classes:
        layout['legend'] = {'traceorder':'reversed'}
    if title:
        layout['title'] = title
    if width:
        layout['width'] = width
    if height:
        layout['height'] = height
    if annotations:
        layout['annotations'] = annotations
    if logscale:
        if logscale != 'xaxis':
            layout.setdefault('yaxis',{})['type'] = 'log'
        if logscale != 'yaxis':
            layout.setdefault('xaxis',{})['type'] = 'log'
    if bins is not None: # Formatting for the histograms
        layout['xaxis'].update(dict(
            domain=[0, 0.85],
            showgrid=False,
            zeroline=False
            ))
        layout['yaxis'].update(dict(
            domain=[0, 0.85],
            showgrid=False,
            zeroline=False
                               ))
        layout['xaxis2']=dict(
            domain=[0.85, 1],
            showgrid=False,
            zeroline=False
            )
        layout['yaxis2']=dict(
            domain=[0.85, 1],
            showgrid=False,
            zeroline=False
            )
        fig = dict(data=data, layout=layout)
        data = fig['data']
        layout = fig['layout']
    if ids:
        return PlotlyPlot(data,layout,click_type=ids[0],id_lookup=id_lookup)
    return PlotlyPlot(data,layout)

def append_scatter_histograms(data,x,y,bins,color,opacity):
    base = {
            'histnorm':'probability',
            'showlegend' : False,
            'opacity': opacity,
            'type':'histogram',
            }
    import numpy as np
    if color:
        base['marker'] = {'color':color}
    else:
        base['marker'] = {'color':Color.default}
    xargs = dict(base)
    xargs.update(x=x,yaxis='y2')
    yargs = dict(base)
    yargs.update(y=y,xaxis='x2')
    if type(bins) == bool:
        xargs.update(autobinx=True)
        yargs.update(autobiny=True)
    else:
        xargs.update(xbins={
                'size':bins[0][1]-bins[0][0],
                'start':bins[0][0],
                'end':bins[0][-1],
                })
        yargs.update(ybins={
                'size':bins[1][1]-bins[1][0],
                'start':bins[1][0],
                'end':bins[1][-1],
                })
    data.append( dict(**xargs))
    data.append( dict(**yargs))

# XXX Maybe add other variations to cover boxplot cases in:
# ML/run_eval_weka
# - cross-validation (horizontal singles)
# nav/views
# - ds_dpi_cmp (2 side-by-side; 1 multiple side-by-side pairs)
# - score_cmp (side-by-side)
def boxplot_stack(datasets,title=None,width=1000,description=None,height_per_dataset=40, xaxis_range=None):
    from dtk.plot import PlotlyPlot
    data = [
            dict(
                    x=plot_data[0],
                    name=plot_data[1],
                    boxpoints='all',
                    jitter=0.5,
                    boxmean='sd',
                    marker=dict(size=3,opacity=0.5),
                    type='box'
                    )
            for plot_data in datasets
            ]
    max_label = max([len(x[1]) for x in datasets])
    margin = float(max(0,max_label-14))/130
    xaxis_dict = {'domain':[margin,1]}
    if xaxis_range:
        xaxis_dict['range']=xaxis_range
    layout = {
            'width':width,
            'height':len(datasets)*height_per_dataset,
            'legend':{
                    'traceorder':'reversed',
                    },
            'yaxis':{
                    'tickangle':-45,
                    },
            'xaxis': xaxis_dict
            }
    if description is not None:
            layout.update({
                'annotations':[annotations(description)[0]],
                'margin':dict(
                              l=60,
                              r=30,
                              b=150,
                              t=80,
                              pad=4
                              )
            })
    if title:
        layout['title'] = title
    return PlotlyPlot(data,layout)

def stripplot(x_label,y_label,data,title='',click=None,description=None):
    '''Return side-by-side strip plots.

    data param is an array with one element per strip; each element is
    an array of point descriptors; each descriptor consists of a y value,
    a hover text, and an optional click key.
    '''
    vpoints = max([len(x) for x in data])
    x = []
    y = []
    keys = []
    text = []
    bg_shading_x = []
    for i,col in enumerate(data):
        for j,t in enumerate(col):
            # The following is a synthesized x coordinate that places the
            # point in strip 'i', and offsets it by 'j' from the center of
            # the strip, so that points with the same y value don't hide
            # each other.  The 5.0 makes the strip 5 times as wide as the
            # occupied width.  Since this value is arbitrary, hoverinfo
            # below is set not to display it.
            x.append(5.0*vpoints*(i+0.5)+j)
            y.append(t[0])
            text.append(t[1])
            if click:
                keys.append(t[2])
        if i % 2 == 0:
            bg_shading_x.append( (5.0*vpoints*i,5.0*vpoints*(i+1)) )
    bg_shading_y = [min(y)*1.1, max(y)*1.1]
    data=[ dict(
            x=x,
            y=y,
            text=text,
            mode='markers',
            showlegend=False,
            hoverinfo='y+text',
            ) ]
    layout=dict(
            hovermode='closest',
            xaxis={'title':x_label,'showticklabels':False},
            yaxis={'title':y_label},
            title=title,
            showlegend=False,
            shapes = [{'type':'rect'
                            , 'x0':tup[0]
                            , 'y0':bg_shading_y[0]
                            , 'x1':tup[1]
                            , 'y1':bg_shading_y[1]
                            , 'line': {'width': 0,}
                            , 'fillcolor': 'rgba(220, 220, 220, 0.3)'
                            }
                            for tup in bg_shading_x
                           ],
            )
    if click:
        return PlotlyPlot(data,layout,click_type=click,id_lookup=[keys])
    return PlotlyPlot(data,layout)


def barplot_stack(datasets,title=None,width=1000,description=None):
    from dtk.plot import PlotlyPlot
    datasets = list(zip(*datasets))
    data = [
            dict(
                 x=datasets[0],
                 y=datasets[1],
                 orientation='h',
                 type='bar'
                )
            ]
    max_label = max([len(x) for x in datasets[1]])
    margin = float(max(0,max_label-14))/130
    layout = {
            'width':width,
            'height':len(datasets[1])*40,
            'legend':{
                    'traceorder':'reversed',
                    },
            'yaxis':{
                    'tickangle':-45,
                    },
            'xaxis':{
                    'domain':[margin,1],
                    },
            }
    if description is not None:
        layout.update({
            'annotations':[annotations(description)[0]],
            'margin':dict(
                          l=60,
                          r=30,
                          b=150,
                          t=80,
                          pad=4
                          )
        })
    if title:
        layout['title'] = title
    return PlotlyPlot(data,layout)

def _assure_dendro_vals(l):
    return [float(x)
            for x in range(4
                    , int(max(l))
                    , 10
                )
           ]

def plotly_heatmap(data_array, row_labels, precalcd = True,
                   Title = None, color_bar_title = "",
                   col_labels = None, width = None,
                   height = None, color_zero_centered = False,
                   hover_text = None, invert_colors = False,
                   zmin = None, zmax = None, reorder_cols = None,
                   colorscale = None, dendro_plot_portion = 0.1,
                   reversescale = False, max_len_r = None,
                   reorder_rows = True, click_type=None, id_lookup=None):
    """Returns a figure with a plotly heatmap and dendrograms

    Rows will be reordered due to dendrogram clustering.
    If reorder_cols is True, cols will also be reordered.

    id_lookup/click_type are keyed on rows, if specified.
    (if you want to add column support, need to modify this func and also
    the click_type support in PlotlyPlot click_handler)
    """
    from plotly import figure_factory as FF
    import numpy as np
    nrow,ncol = data_array.shape
    assert len(row_labels) == nrow
    if not col_labels:
        col_labels = row_labels
        for_col = data_array
        show_col_labs = False
    else:
        for_col = np.swapaxes(data_array, 0,1)
        show_col_labs = True
    assert len(col_labels) == ncol, f'{len(col_labels)} labels, but {ncol} cols'

    if reorder_cols is None and set(col_labels) == set(row_labels):
        reorder_cols = True

    if reorder_cols and len(col_labels) != len(set(col_labels)):
        # We index into the dendrogram labels for the reordering, which breaks if we have duplicate labels.
        # So we give them all unique labels if we see any dupes.
        # TODO: There is probably a better way to extract the clustering than text indexing.
        col_labels = [f'{i}: {v}' for i, v in enumerate(col_labels)]

    if reorder_rows and len(row_labels) != len(set(row_labels)):
        # See comment for cols.
        row_labels = [f'{i}: {v}' for i, v in enumerate(row_labels)]

    if nrow > 1:
        # Initialize figure by creating row dendrogram
        figure = FF.create_dendrogram(data_array, orientation='right', labels=row_labels).to_dict()
    else:
        print('WARNING: Only one row was provided. So that we can plot the data, that row will be plotted twice')
        figure = FF.create_dendrogram(np.concatenate((data_array,data_array), axis=0),
                                        orientation='right',
                                        labels=row_labels+row_labels
                                     ).to_dict()
    if not reorder_rows:
        figure['data'] = []
    # get the y values of the dendrogram ends
    row_dendro_leaves = [row_labels.index(x) for x in figure['layout']['yaxis']['ticktext']]
### I ran into an issue where the tickvals were not evenly split and thus the heatmaps weren't working
### this hack around fixed things
    figure['layout']['yaxis']['tickvals'] = _assure_dendro_vals(figure['layout']['yaxis']['tickvals'])
    for i in range(len(figure['data'])):
        figure['data'][i]['xaxis'] = 'x2'
    # repeat the process for the column Dendrogram
    if ncol > 1:
        dendro_col = FF.create_dendrogram(for_col, orientation='bottom', labels=col_labels).to_dict()
    else:
        ### Only one column was provided, we can't cluster that, so we just hack around it
        ### by doubling the original column to make the proper data structure then remove it
        print('WARNING: Only one column was provided. So that we can plot the data, that column will be plotted twice')
        dendro_col = FF.create_dendrogram(np.concatenate((for_col, for_col), axis=0),
                                            orientation='bottom',
                                            labels=col_labels+col_labels
                                          ).to_dict()
    if max_len_r is None:
        max_len_r = max([len(i) for i in figure['layout']['yaxis']['ticktext']])
    max_len_pix = max_len_r*11
    dendro_col['layout']['xaxis']['tickvals'] = _assure_dendro_vals(dendro_col['layout']['xaxis']['tickvals'])
    for i in range(len(dendro_col['data'])):
        dendro_col['data'][i]['yaxis'] = 'y2'

    if reorder_cols:
        # Add col Dendrogram Data to Figure only if we are reordering columns
        # based on the dendrogram.  Otherwise it will be incorrect.
        figure['data'].extend(dendro_col['data'])
    # Create Heatmap
    # organize the data
    if precalcd:
        if reorder_rows:
            heat_data = data_array[row_dendro_leaves,:]
        else:
            heat_data = data_array
            row_dendro_leaves = list(range(len(row_labels)))
            figure['layout']['yaxis']['ticktext'] = np.array(row_labels)
        if reorder_cols:
            col_dendro_leaves = [col_labels.index(x) for x in dendro_col['layout']['xaxis']['ticktext']]
            col_labels = dendro_col['layout']['xaxis']['ticktext']
            heat_data = heat_data[:,col_dendro_leaves]
        else:
            col_dendro_leaves = list(range(len(col_labels)))
    else:
        from scipy.spatial.distance import pdist, squareform
        col_dendro_leaves = [col_labels.index(x) for x in dendro_col['layout']['xaxis']['ticktext']]
        data_dist = pdist(data_array)
        heat_data = squareform(data_dist)
        heat_data = heat_data[row_dendro_leaves,:]
    if hover_text is not None:
        hover_text = hover_text[row_dendro_leaves,:]
        hover_text = hover_text[:,col_dendro_leaves]
        hover_text = hover_text

    # create the plot
    if color_zero_centered:
        zmax = max([np.amax(heat_data), abs(np.amin(heat_data))])
        zmin = -1.0 * zmax
    else:
        if zmin is None:
            zmin = np.amin(heat_data)
        if zmax is None:
            zmax = np.amax(heat_data)
    heatmap = [
        dict(
            type = 'heatmap',
            x = dendro_col['layout']['xaxis']['tickvals'],
            y = figure['layout']['yaxis']['tickvals'],
            z = heat_data,
            zmin = zmin,
            zmax = zmax,
            colorscale = colorscale or _heatmap_colorscale(invert_colors),
            text = hover_text,
            reversescale = reversescale,
        )
    ]
    # set dimensions dynamically from the size of the matrix
    if height is None:
        height = 110+20*nrow+9*max([len(x) for x in col_labels])
    if width is None:
        width = 100+20*ncol+10*max([len(x) for x in row_labels])
    ratio = float(height)/width if width > height else float(width)/height
    if width > height:
        colormap_len = dendro_plot_portion/ratio*1.5
    else:
        colormap_len = dendro_plot_portion*1.5
    heatmap[0]['colorbar']={
                                'yanchor' : 'top',
                                'y' : 1.+(dendro_plot_portion/5.),
                                'len' : colormap_len,
                                'xanchor' : 'left',
                                'x' : 0.-(dendro_plot_portion*ratio),
                           #     'width' : dendro_plot_portion
                               }
    # Add Heatmap Data to Figure
    figure['data'].extend(heatmap)

    if id_lookup:
        # Reorder according to the rows.
        id_lookup = np.array(id_lookup)[row_dendro_leaves]
        # Attach to the trace id corresponding to the actual heatmap
        id_lookup = { len(figure['data'])-1 : id_lookup }


    # Edit Layout
    figure['layout'].update({'width':width, 'height':height,
                             'showlegend':False, 'hovermode': 'closest',
                             })
    figure['layout']['yaxis']['ticktext'] = figure['layout']['yaxis']['ticktext']
    figure['layout']['yaxis']['tickmode'] = 'array'
    if show_col_labs:
        figure['layout']['xaxis']['tickvals'] = dendro_col['layout']['xaxis']['tickvals']
        if precalcd:
            figure['layout']['xaxis']['ticktext'] = col_labels
        else:
            figure['layout']['xaxis']['ticktext'] = dendro_col['layout']['xaxis']['ticktext']
        figure['layout']['xaxis']['tickmode'] = 'array'

    for i in range(len(figure['data'])):
        for ind in ['x', 'y', 'z']:
            try:
                figure['data'][i][ind] = figure['data'][i][ind]
            except (AttributeError, KeyError):
                pass

    figure['layout'] = _set_heatmap_layout(figure['layout'],
                                           show_col_labs,
                                           max_len_pix,
                                           height,
                                           width,
                                           ratio,
                                           dendro_plot_portion
                                          )

    if Title:
        figure['layout']['title'] = Title
    return PlotlyPlot(figure['data'], figure['layout'], click_type=click_type, id_lookup=id_lookup)
def _set_heatmap_layout(fl, show_col_labs, max_len_pix,height,width,ratio,margin):
    # Scaling custom_margin by ratio can cause extreme skewing of uneven plots, as
    # the dendrogram takes up far more margin than accounted for in that dimension,
    # squishing the actual plot area.
    # custom_margin = margin/ratio
    custom_margin = margin
    import plotly.graph_objs as go
    if width > height:
        x1 = [margin, 1.]
        x2 = [0., margin]
        y1 = [0, (1.-custom_margin)]
        y2 = [(1.-custom_margin), 1.0]
    else:
        x1 = [custom_margin, 1.]
        x2 = [0., custom_margin]
        y1 = [0, (1.-(margin * 0.75))]
        y2 = [(1.-margin), 1.0]
    fl['xaxis'].update({'domain': x1,
                        'mirror': False,
                        'showgrid': False,
                        'showline': False,
                        'anchor' : 'free',
                        'side' : 'bottom',
                        'position' : 0.,
                        'showticklabels': show_col_labs,
                        'zeroline': False,
                        'ticks':""})
    # Edit xaxis2 (row dendro)
    fl.update({'xaxis2': {'domain': x2,
                          'mirror': False,
                          'showgrid': False,
                          'showline': False,
                          'zeroline': False,
                          'showticklabels': False,
                          'ticks':""}})
    # Edit yaxis which is more complicated b/c of the color bar
    fl['yaxis'].update({'domain': y1,
                        'mirror': False,
                        'showgrid': False,
                        'showline': False,
                        'anchor' : 'free',
                        'side' : 'right',
                        'position' : 1.,
                        'showticklabels': True,
                        'zeroline': False,
                        'ticks': ""})
    # Edit yaxis2 (col dendro)
    fl.update({'yaxis2':{'domain':y2,
                         'mirror': False,
                         'showgrid': False,
                         'showline': False,
                         'zeroline': False,
                         'showticklabels': False,
                         'ticks':""}})
    fl['margin']=dict(
                      l=5,
                      r=max_len_pix,
                      b=50,
                      t=50,
                      pad=4
                     )
    return fl

def _heatmap_colorscale(invert_colors):
    import numpy as np
    if invert_colors:
        color_range = list(np.linspace(1.0, 0.0, 10))
    else:
        color_range = list(np.linspace(0.0, 1.0, 10))

    return [[color_range[9], 'rgb(165,0,38)']
           , [color_range[8], 'rgb(215,48,39)']
           , [color_range[7], 'rgb(244,109,67)']
           , [color_range[6], 'rgb(253,174,97)']
           , [color_range[5], 'rgb(254,224,144)']
           , [color_range[4], 'rgb(224,243,248)']
           , [color_range[3], 'rgb(171,217,233)']
           , [color_range[2], 'rgb(116,173,209)']
           , [color_range[1], 'rgb(69,117,180)']
           , [color_range[0], 'rgb(49,54,149)']
           ]

# the question is how to customize the histogram from here
def smart_hist(data, layout = {}, n_log = 3, nbins = 50):
    from math import log
    import numpy as np
    hist_data = np.histogram(np.array(data), nbins)
    range = max(hist_data[0]) - min(hist_data[0])
    if log(range, 10) >= n_log:
        if 'yaxis' not in layout:
            layout['yaxis'] = {}
        layout['yaxis'].update({'type':'log'})
    return PlotlyPlot([dict(x = data, nbinsx = nbins, type='histogram')]
                    , layout
                   )


def score_plot(ordering, marked_id_sets, dtc='wsa', base_color=None, marked_colors=None):
    from browse.models import WsAnnotation
    ids = []
    score_vals = []
    colors = [base_color or Color.default] * len(ordering)
    widths = [1] * len(ordering)

    marked_colors = marked_colors or Color.plotly_defaults[1:len(marked_id_sets)+1]
    id_colors = {}

    for marked_id_set, color in zip(marked_id_sets, marked_colors):
        for id in marked_id_set:
            id_colors[id] = color

    for i,x in enumerate(ordering):
        ids.append(str(x[0]))
        score_vals.append(x[1])
        if x[0] in id_colors:
            colors[i] = id_colors[x[0]]
            widths[i] = 3

    # ugly way to get ws object; assume id list not empty
    if dtc == 'wsa':
        ws = WsAnnotation.all_objects.get(pk=ids[0]).ws
        name_map = ws.get_wsa2name_map()
        drug_names = [name_map[int(id)] for id in ids]
    elif dtc == 'uniprot':
        from browse.views import get_prot_2_gene
        name_map = get_prot_2_gene(ids)
        drug_names = [name_map[id] for id in ids]
    trace = dict(
                 type='bar',
                 y = score_vals,
                 text = drug_names,
                 marker = dict(
                      color = colors,
                      line = dict(
                             width = widths,
                             color = colors
                         )
                     )
                )
    fig = {'data': [trace],
           'layout': {
           }}
    fig['layout'].update(height = 800
                       , width = 800
                       , title = 'Scores'
                       , showlegend = False
                       , xaxis = {'title': 'Drug Rank'}
                       , yaxis = {'title': 'Drug score'}
                       )
    return PlotlyPlot(fig['data'],
                fig['layout'],
                id_lookup=[ids,None,ids],
                click_type={
                        'wsa':'drugpage',
                        'uniprot':'protpage',
                        }[dtc]
                )


def color_with_opacity(color, opacity):
    if color[0] == '#':
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
    else:
        elements = color.split('(')[1].rstrip(')').split(',')
        r, g, b = elements[:3]

    return f'rgba({r},{g},{b},{opacity})'


def bar_histogram_overlay(x_data, names, bins=None, x_range=None, density=False, jitter=0, handle_nan=True, annotations=None, show_mean=True):
    """Generates a bar chart that mimics overlaid histograms.

    Normal plotly plots retain all of the underlying datapoints, which is
    undesired with huge amounts of data.
    This method precomputes the relevant summary statistics and keeps only those.

    If you specify a jitter, bars will be offset by that amount (in x-units,
    not pixels).  This can sometimes make multiple distributions easier to see.
    """
    traces = []
    import numpy as np

    if x_range is None:
        if not handle_nan:
            min_func = np.min
            max_func = np.max
        else:
            min_func = np.nanmin
            max_func = np.nanmax
        mins = [min_func(xs) for xs in x_data]
        maxs = [max_func(xs) for xs in x_data]
        if not mins and not maxs:
            x_range = (0, 1)
        else:
            x_range = (min_func(mins), max_func(maxs))
    if bins is None:
        bins = 20

    for trace_idx, x_i in enumerate(x_data):
        color = Color.plotly_defaults[trace_idx % len(Color.plotly_defaults)]

        mean = np.mean(x_i)

        # note that np.histogram's density argument is not what we want, it
        # creates a PDF whose integral sums to 1, rather than having the
        # individual bars sum to 1.
        hist, edges = np.histogram(x_i, bins=bins, range=x_range)

        if density:
            hist = np.array(hist, dtype=float) / np.sum(hist)

        bar_x = []
        bar_y = []
        mean_x = []
        mean_y = []
        for i in range(len(hist)):
            if mean >= edges[i] and mean < edges[i+1]:
                mean_x.extend([mean, mean])
                mean_y.extend([hist[i], 0])

            bar_x.extend([edges[i], (edges[i]+edges[i+1])/2, edges[i+1]])
            bar_y.extend([hist[i], hist[i], hist[i]])
        """
        # Kernel density equivalent for future reference.
        # Bandwidth estimation is hard, though, and the absolute Y values
        # become very hard to interpret, and area under curve becomes more
        # important than peakiness which is hard to intuit.
            from scipy.stats import gaussian_kde
            bin_width = (x_range[1] - x_range[0]) / bins
            kernel = gaussian_kde(x_i, bw_method=bin_width*2)
            bar_x = np.arange(x_range[0], x_range[1] + bin_width - 1e-9, bin_width / 10)
            bar_y = kernel(bar_x)
        """

        trace = {
            'x': bar_x,
            'y': bar_y,
            'name': names[trace_idx],
            'legendgroup': names[trace_idx],
            'type': 'scattergl',
            'fill': 'tozeroy',
            'fillcolor': color_with_opacity(color, 0.05),
            'line':dict(
                color=color_with_opacity(color, 0.5),
                width=2.0,
                ),
        }

        traces.append(trace)
        if show_mean:
            mean_trace = {
                'x': mean_x,
                'y': mean_y,
                'name': '',
                'type': 'scattergl',
                'showlegend': False,
                'legendgroup': names[trace_idx],
                'line':dict(
                    color=color_with_opacity(color, 0.5),
                    width=2.0,
                    dash='dot',
                    ),
            }
            traces.append(mean_trace)
    dtick = (x_range[1] - x_range[0]) / bins
    layout = {
        'barmode': 'overlay',
        'bargap': 0,
        'xaxis': {
            'dtick': dtick,
            'tick0': x_range[0] + dtick / 2,
            }
        }
    if annotations:
        layout['annotations'] = annotations
    return traces, layout


class Controls:
    """Some prebaked plotly controls you can add to the plot.

    Also serves as useful templates/examples for overriding for more custom.
    See https://plotly.com/python/dropdowns/
    """
    log_linear_y = {
                "buttons": [
                    dict(label='Linear Scale',
                        method='relayout',
                        args=[{'yaxis': {'type': 'linear'}}]),
                    dict(label='Log Scale',
                        method='relayout',
                        args=[{'yaxis': {'type': 'log'}}]),
                    ],
                "active": 0, # which of the buttons starts active
                "type": "dropdown",
            }


# new class for using plotly.js
# - instantiate this class in any of several ways
# - pass it to the template in the context
# - make sure the template loads plotly.js
# - expand it using _plotly_div.html
class PlotlyPlot(object):
    # see https://plot.ly/javascript/configuration-options/
    # and plotly.js src/plot_api/plot_config.js
    config_default={
            'modeBarButtonsToRemove':['sendDataToCloud'],
            'displaylogo': False,
            'showLink': False,
            }
    def __init__(self,
                data=[],
                layout={},
                config=dict(config_default),
                id_lookup=None,
                click_type=None,
                ):
        self._data = data
        self._layout = layout
        self._config = config
        self._id_lookup = id_lookup
        self._click_type = click_type
        self._thumbnail_stem = ''
        # This was to guard against a '&alpha;' in an ATC code description
        # that was crashing plotly (and so hanging up the xvfb job). But
        # it turns out the '&' character appears in drug names without
        # causing any issues. So, this is disabled until we identify a more
        # specific case to test for.
        #assert '&' not in self.data_json()

    def update_layout(self, layout):
        self._layout.update(layout)
    
    def add_control(self, control_data):
        """See Controls class above for examples of things to add."""
        menus = self._layout.setdefault('updatemenus', [])
        menus.append(control_data)

    #####
    # extraction methods for django templates
    #####
    @staticmethod
    def _as_json(x):
        import json
        from django.utils.safestring import mark_safe
        return mark_safe(json.dumps(x, default=convert))
    def width(self): return self._layout.get('width',640)
    def height(self): return self._layout.get('height',480)
    def data_json(self): return self._as_json(self._data)
    def layout_json(self): return self._as_json(self._layout)
    def config_json(self): return self._as_json(self._config)
    def handles_click(self): return bool(self._click_type)
    def id_lookup(self): return self._as_json(self._id_lookup)
    def as_figure(self):
        from plotly.graph_objects import Figure
        return Figure(self.as_dict(), skip_invalid=False)
    def click_handler(self):
        from django.utils.safestring import mark_safe
        if self._click_type == 'drugpage':
            return mark_safe('''
                    var point = data.points[0];
                    var pointNumber = point.pointNumber;
                    if (Array.isArray(pointNumber)) {
                        // Heatmap usually.
                        pointNumber = pointNumber[0];
                    }
                    var id = idLookup[point.curveNumber][pointNumber];
                    if (id != 0){
                        var url = '/mol/'+wsId+'/annotate/'+id+'/';
                        window.open(url,'_blank');
                    }
                    ''')
        elif self._click_type == 'synergypage':
            return mark_safe('''
                    var point = data.points[0];
                    var pointNumber = point.pointNumber;
                    if (Array.isArray(pointNumber)) {
                        // Heatmap usually.
                        pointNumber = pointNumber[0];
                    }
                    var pair = idLookup[point.curveNumber][pointNumber];
                    var url = '/score/'+wsId+'/score_synergy/?x='+pair[0]+'&y='+pair[1];
                    window.open(url,'_blank');
                    ''')
        elif self._click_type == 'protpage':
            return mark_safe('''
                    var point = data.points[0];
                    var pointNumber = point.pointNumber;
                    if (Array.isArray(pointNumber)) {
                        // Heatmap usually.
                        pointNumber = pointNumber[0];
                    }
                    var id = idLookup[point.curveNumber][pointNumber];
                    var url = '/'+wsId+'/protein/'+id+'/';
                    window.open(url,'_blank');
                    ''')
        elif self._click_type == 'pathwaypage':
            return mark_safe('''
                    var point = data.points[0];
                    var pointNumber = point.pointNumber;
                    if (Array.isArray(pointNumber)) {
                        // Heatmap usually.
                        pointNumber = pointNumber[0];
                    }
                    var id = idLookup[point.curveNumber][pointNumber];
                    var url = `/pathway_network/#{"initPathway":"${id}"}`; 
                    window.open(url,'_blank');
                    ''')
        
        elif self._click_type == 'sigpage':
            return mark_safe('''
                    var point = data.points[0];
                    var id = idLookup[point.curveNumber][point.pointNumber];
                    var url = '/cv/'+wsId+'/sigprot/'+id+'/';
                    window.open(url,'_blank');
                    ''')
        elif self._click_type == 'anyjobpage':
            return mark_safe('''
                    var point = data.points[0];
                    var pair = idLookup[point.curveNumber][point.pointNumber];
                    var wsId = pair[0];
                    var id = pair[1];
                    var url = '/cv/'+wsId+'/progress/'+id+'/';
                    window.open(url,'_blank');
                    ''')

        raise Exception("unknown click type '%s'" % self._click_type)
    #####
    #  png thumbnail support
    #####
    def thumbnail(self):
        return bool(self._thumbnail_stem)
    def thumb_loc(self):
        """Used by e.g. progress page to find URL for displaying"""
        # strip initial '/' to make life easier for template
        from path_helper import PathHelper
        return PathHelper.url_of_file(self._thumbnail_stem)[1:]

    thumbnail_pending = {}
    @classmethod
    def get_background(cls):
        if not getattr(cls, 'executor', None):
            from concurrent.futures import ThreadPoolExecutor
            cls.executor = ThreadPoolExecutor(max_workers=1)
        return cls.executor

    @classmethod
    def block_if_thumbnailing(cls, path):
        if os.path.exists(path):
            return
        if path.endswith('.png'):
            # There is a simpler way to do this in-process, but
            # platform might have multiple processes running.

            path = os.path.abspath(path)

            if path in cls.thumbnail_pending:
                logger.info("Waiting on thumbnail future: %s", path)
                # This is in-process, just wait on the future and return.
                cls.thumbnail_pending[path].result()
                return

            lock_path = path + '.lock'
            if os.path.exists(lock_path):
                logger.info("Waiting on thumbnail lock: %s", lock_path)
                # Wait for the other process to give up the lock.
                from dtk.lock import FLock
                with FLock(lock_path):
                    pass

    def _build_thumbnail(self,path,force=False):
        suffix = '.plotly'
        if path.endswith('.gz'):
            suffix += '.gz'
        assert path.endswith(suffix)
        path = os.path.abspath(path)
        if path in self.thumbnail_pending:
            return

        self._thumbnail_stem = path[:-len(suffix)]
        thumbnail_path = self._thumbnail_stem+'.png'
        if os.path.exists(thumbnail_path) and not force:
            return
        lock_path = thumbnail_path + '.lock'
        from dtk.lock import FLock
        lock = FLock(lock_path)
        lock.acquire()
        logger.info("Locked for thumbnailing: %s", lock_path)
        def run():
            import time
            now = time.time()

            try:
                import plotly.io as pio
                # Plotly defaults to some new theme, but it's inconsistent with
                # how we see it in our javascript, so turn it off for now.
                pio.templates.default = 'none'
                from plotly.graph_objects import Figure
                from plotly.io import write_image
                from path_helper import PathHelper
                with FLock(PathHelper.timestamps+"ploty_orca_lock"):
                    write_image(self.as_dict(), thumbnail_path, validate=False)
                    #fig = Figure(self.as_dict(), skip_invalid=True)
                    #fig.write_image(thumbnail_path, validate=False)
            except (OSError, ValueError) as e:
                import traceback
                traceback.print_exc()
                logger.error("Failed to write thumbnail - %s", e)
            finally:
                os.unlink(lock_path)
                lock.release()
                del self.thumbnail_pending[thumbnail_path]

            then = time.time()
            print("Wrote %s in %.2f seconds" % (thumbnail_path, then - now))
        future = self.get_background().submit(run)
        self.thumbnail_pending[thumbnail_path] = future
    def as_dict(self):
        return {
            'data':self._data,
            'layout':self._layout,
            'config':self._config,
            'id_lookup':self._id_lookup,
            'click_type':self._click_type,
        }
    #####
    # save and restore from file
    #####
    def save(self,path,thumbnail=False):
        import json
        s = json.dumps(self.as_dict(), default=convert)
        from dtk.files import open_cmp
        with open_cmp(path, 'w') as f:
            f.write(s+'\n')
        if thumbnail:
            self._build_thumbnail(path,force=True)
    @classmethod
    def build_from_file(cls,path,thumbnail=False):
        from dtk.files import open_cmp
        with open_cmp(path, 'r') as f:
            s = f.read()
        import json
        d = json.loads(s)
        data = d['data']

        # Working around an issue where the default changed in plotly 5.0, causing any attempts
        # to render old bar charts with text hovers take forever, due to it trying to render them on the bars.
        # While we've fixed up newer plots to explicitly include 'textposition'='none', for older plots we use this
        # shim here to ensure that if unspecified we set it to none.
        # https://github.com/plotly/plotly.js/issues/5764
        for trace in data:
            if trace.get('type') == 'bar':
                if 'textposition' not in trace:
                    trace['textposition'] = 'none'

        obj = cls(data,d['layout'],d['config'],
                d.get('id_lookup'),
                d.get('click_type'),
                )
        if thumbnail:
            obj._build_thumbnail(path,force=False)
        return obj
    #####
    # other construction methods
    #####
    @classmethod
    def build_from_plotly_figure(cls,fig):
        fig_d = fig.to_dict()
        return PlotlyPlot(fig_d['data'], fig_d['layout'])
    @classmethod
    def build_from_mpl(cls,fig):
        import plotly
        p=plotly.tools.mpl_to_plotly(fig)
        return PlotlyPlot(p['data'],p['layout'])
    @classmethod
    def build_from_R(cls,path):
        with open(path) as f:
            for line in f:
                if 'application/json' in line:
                    end_tag='</script>\n'
                    line=line[line.index('>')+1:]
                    assert line.endswith(end_tag)
                    line = line[:-len(end_tag)]
                    import json
                    d = json.loads(line)
                    d = d['x']
                    d['config']['displaylogo'] = False
                    d['config']['showLink'] = False
                    return cls(d['data'],d['layout'],d['config'])
        raise RuntimeError("plotly data not found in '%s'" % path)

