import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import itertools

try:
    from pygbox import ops
except ImportError as e:
    import ops

def set_plot_style(pltstyle = 'seaborn-paper', kwargs = {}):
    plt.style.use(pltstyle)
    pltparams = {
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'font.size': 20,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'legend.fontsize': 16,
        'figure.figsize': (5, 5),
        'font.family': 'sans-serif',
        'font.sans-serif': ['Tahoma'],
        "text.usetex": False,
        "axes.facecolor": "white",
        'image.cmap': 'viridis'}
    pltparams.update(kwargs)
    mpl.rcParams.update(pltparams)
    #plt.set_cmap("viridis")

def make_fig(nrows = 1, ncols = 1, fig_kw = {}):
    """tba"""
    subplot_kw = {}
    gridspec_kw={}
    fig_kw_ = { "dpi": 300, "tight_layout": True }
    fig_kw_.update(fig_kw)
    fig, ax = plt.subplots(nrows, ncols, **fig_kw_)
    return fig, ax

def wait(timeout = 5):
    """ Helper fun to avoid having to call `waitforbuttonpress` """
    plt.show()
    plt.waitforbuttonpress(timeout)
    plt.close('all')

def name_ax(ax, xlab = None, ylab = None, titl = None, legend = None):
    ax.set_title(titl)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if legend: ax.legend(legend)

def annotate_ax(ax, txt, x = 0.2, y = 0.85):
    """tba"""
    if not isinstance(txt, (str)):
        # assume it is datta
        txt = r"$\mu$ = {:.2f} \n$\nu$ = {:.2f} \n$\sigma$ = {:.2f}".\
            format(np.mean(txt), np.median(txt), np.std(txt))
    plt.text(x, y, txt, transform = ax.transAxes)

def plot_hist(ax, x, bins = 'fd', density = True, kwargs = {}):
    """Plot histogram of distributions"""
    #n, bins, patches = ax.hist( x, bins, density = True, histtype='step',
    #                            cumulative = False, alpha = 0.5)

    nn, bins = np.histogram(x, bins, density = density)
    #nn = nn / nn.max() # normalize to 1
    #bins = bins[:-1] + np.abs(np.diff(bins)) / 2
    kwargs_.update(kwargs)
    ax.stairs(nn, bins, **kwargs_)

def plot_hist2(ax, x, bins = 'fd', txt = None, xlab = None, kwargshist = {}, kwargskde = {}):
    kwargs_hist = { 'ec': 'black',
                    'color':  'darkgrey',
                    'density': True
                    }
    kwargs_hist.update(kwargshist)
    n, bins, parts = ax.hist(x, bins = bins, **kwargs_hist)

    kwargs_kde = {  'fill': False,
                    'lw': 3,
                    'c': 'black'
                    }
    kwargs_kde.update(kwargskde)
    axs= sns.kdeplot(x, ax = ax, **kwargs_kde)

    # annotate ax
    if not isinstance(txt, (str)):
        #txt = "$\mu$={:.1f}\n$\pm${:.1f} min\n n={:d}".format(xmean_, xerr_, len(x))
        xmean_, (xstd_, xerr_) = ops.summarize_data(x)
        txt = "$\mu$={:.1f}$\pm${:.2f}".format(xmean_, xstd_)
    annotate_ax(ax, txt = txt, x = .6, y = .8)

    # beatify axes
    ax.set_ylabel('frequency')
    ax.set_yticklabels([])
    ax.set_yticks([])
    #ax.set_xlabel('$t_{eq}$ [min]')

    if not xlab: xlab = '$Rg/Rg_{end}$ [-]'
    ax.set_xlabel(xlab)
    ax.xaxis.set_ticks_position('both')
    ax.xaxis.set_tick_params(direction='in')

    #  adjust axes limtis
    #fig.tight_layout()
    return ax

def plot_ecdf(ax, x, legend = '', xlabel = ''):
    """Empirical cumulative distribution plot"""
    # option 1
    # n, bins, patches = ax.hist(x, bins, density = True, alpha = 0.5)
    # option 2
    if not isinstance(x, (list, tuple)): x = [x]
    for i in x:
        i = np.sort(i)
        i_cdf = np.cumsum(i / np.sum(i))
        p = ax.plot(i, i_cdf, alpha = 0.75)

    # set colors
    clr = p[0].get_color()
    ax.tick_params(axis='y', colors= clr)
    ax.yaxis.label.set_color(clr)
    ax.spines['left'].set_color(clr)

    plt.legend(legend)
    ax.set_ylabel("eCDF [-]")
    ax.set_xlabel(xlabel)

def plot_violin(ax, x, xlabels, ylabel, stat = 'median'):
    """Violin plot
    Consider seaborn violin plot in the future
    """
    x = [np.sort(y) for y in x]
    parts = ax.violinplot(  x, showmeans = False, showmedians = False,
                            quantiles = [[0.1, 0.9]] * len(x), showextrema = False)

    set_xaxis_labels(ax, xlabels)
    ax.set_ylabel(ylabel)
    customize_violin(ax, x, parts, stat)
    label_sample_size(ax, x)
    return parts

def set_xaxis_labels(ax, labels, start = 1, do_rotate = True):
    ax.xaxis.set_tick_params(direction='in')
    ax.xaxis.set_ticks_position('both')
    if start:
        ax.set_xticks(np.arange(start, start + len(labels)))
    else:
        ax.set_xticks(np.arange(0, len(labels)))
    if (np.max([len(x) for x in labels]) > 3) and do_rotate:
        ax.set_xticklabels(labels, rotation = 45)
    else:
        ax.set_xticklabels(labels)
    ax.set_xlim(start - 0.75, start + len(labels) - 0.25)


def customize_violin(ax, x, parts, stat = 'median'):
    """customize violin plot"""
    ax.tick_params(axis = 'both', direction = 'in')
    ax.yaxis.set_ticks_position('both')
    # get colors, but drpp the first color
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #colors = ['#9467bd', '#2ca02c', '#ff7f0e', '#ff7f0e']
    #import pdb; pdb.set_trace()
    if len(x) > 2:
        colors = np.roll(np.array(colors), 1)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(.8)

    parts['cquantiles'].set_color('black')
    parts['cquantiles'].set_alpha(0.5)

    qLs = [np.quantile(y, .25) for y in x]
    qHs = [np.quantile(y, .75) for y in x]
    if stat == 'median':
        q5s = [np.quantile(y, .5) for y in x]
    elif stat == 'mean':
        q5s = [np.mean(y) for y in x]
    elif stat == 'logmode': # most common value
        q5s = [10**ops.get_mode(np.log10(y))[0] for y in x]
    elif stat == 'mode':
        q5s = [ops.get_mode(y)[0] for y in x]

    whiskers = np.array([
        adjacent_values(sorted_array, qL, qH)
        for sorted_array, qL, qH in zip(x, qLs, qHs)])
    inds = np.arange(1, len(q5s) + 1)
    ax.scatter(inds, q5s, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, qLs, qHs, color='k', linestyle='-', lw=5)
    ax.vlines(  inds, [np.quantile(y, .1) for y in x],
                [np.quantile(y, .9) for y in x], color='k', linestyle='-',
                lw=1, alpha = 0.5)

    if np.mean(q5s) > 1e3:
        ax.set_yscale('log')
        ax.yaxis.set_tick_params(direction='in', which = 'both')

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def label_sample_size(ax, x, y = 0.8, kwargs = {}):
    """Add sample size info per each tick
    Use in combination with violin, bar plots

    Reference:
        https://www.python-graph-gallery.com/58-show-number-of-observation-on-violinplot
        https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html
    """
    kwargs_ = {  'fontsize': 16, 'fontfamily': 'sans-serif',
                'horizontalalignment': 'center', 'rotation': 90}
    kwargs_.update(**kwargs)
    trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for tick, label, x_ in zip(ax.get_xticks(), ax.get_xticklabels(), x):
        ax.text(tick-0.5, y, "n=" + str(len(x_)), transform = trans,
                **kwargs_)

def plot_spokes(spokes, coords, kwargs = {}):
    """Plot radial intensity profiles as spokes"""

    #spokes = np.ma.masked_where(spokes, np.isnan(spokes))
    # take on any further arguments
    _kwargs = {'alpha': .5}
    _kwargs.update(kwargs)
    # TODO: insteead of averaging - draw a randoom spoke

    ## Option 1 - Plot in 2D / 3D
    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')

    # unpack coordinates - AZIMUTHALLY
    ph, ri = np.meshgrid(*coords[1:], indexing = 'ij')
    # Express the mesh in the cartesian system.
    xx, yy = (ri*np.cos(ph), ri*np.sin(ph))
    # Combine info along polar angle:
    zz = np.median(spokes, axis = 0)
    # Plot the surface.
    ax1.plot_surface(xx, yy, zz, cmap='viridis', **_kwargs)

    # unpack coordinates - POLAR
    th, ri = np.meshgrid(coords[0], coords[2], indexing = 'ij')
    # Express the mesh in the cartesian system.
    xx, yy = (ri*np.cos(th), ri*np.sin(th))
    # Combine info along polar angle:
    zz = np.median(spokes, axis = 1)
    # Plot the surface.
    ax2.plot_surface(xx, yy, zz, cmap='viridis', **_kwargs)

    ## Option 2 - simple plot in 2D
    #combine polar and azimuthal information
    spokes = np.reshape(spokes, (-1, spokes.shape[-1])).T
    ax3 = fig.add_subplot(223)
    ax = plt.plot(coords[-1], spokes, **_kwargs)

    plt.show()

def plot_rg_sum(ax, rg, sm, kwargs = {}):
    """ Plot radius of gyration against sum intensity

    Intensity should be logged (not sure if natural or base-10)
    Can make nondimensional?

    TODO: add fiting? See : "~\git\gbox\matlab\plots\plotIntensity.m"
    """
    kwargs_ = {'marker': '.', 'ls': '', 'markersize': 10}
    kwargs_.update(kwargs)
    ax.plot(rg, sm, **kwargs_)

    #ax.set_ylim(bottom = 0)
    ax.xaxis.set_tick_params(direction='in',  which = 'both')
    ax.yaxis.set_tick_params(direction='in', which = 'both')
    ax.tick_params(axis = 'both', direction = 'in')
    if np.mean(sm) > 1e3:
        ax.set_yscale('log')
        ax.yaxis.set_tick_params(direction='in', which = 'both')
    return ax


def plot_conditions(ax, x1, x2, titl = "", ylab = "", ticks = "", xerr1 = [], xerr2 = [],
                    scale = False):
    """plot mass spec results

    References:
        [1]: https://stackoverflow.com/a/56212312"""
    error_kw = {'capsize': 10, 'elinewidth': 2, 'ecolor': 'k', 'barsabove': False}

    # set up axes values
    vals = np.asarray(x1 + x2)
    errs = np.asarray(xerr1 + xerr2)
    if len(errs) == 0: errs = None
    x = np.arange(1, len(vals) + 1)
    if scale:
        scaler =  vals.max()
        vals = vals/scaler
        errs = errs/scaler

    # set up colors
    c1, c2 = plt.rcParams['axes.prop_cycle'].by_key()['color'][:2]
    cs = [c1] * len(x1) + [c2] * len(x2)
    # Plot and set up tick and labels
    ax.bar(x, vals, width = 0.4, color = cs, yerr = errs, error_kw = error_kw)
    ax.grid(False)
    ax.set_ylabel(ylab)
    set_xaxis_labels(ax, ticks)
    ax.set_title(titl)
    ax.tick_params(axis = 'both', direction = 'in')
    ax.yaxis.set_ticks_position('both')

    return ax


def plot_msd(ax, tau, msd, msd_std, kwargs = {}):
    """plot MSD curve"""
    ax.plot(tau, msd, **kwargs)
    #ax.fill_between(tau, msd - msd_std, msd + msd_std, alpha=0.2, **kwargs)
    ax.set_xlim(left = 0); ax.set_ylim(bottom = 0)

    name_ax(ax, xlab = 'lag time [s]', ylab = 'MSD $[\mu m^2$]')

    # force ticks
    ax.yaxis.set_ticks(ticks = np.arange(0, 1.5, 0.5))
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(axis = 'both', direction = 'in')
    ax.xaxis.set_tick_params(direction='in',  which = 'both')
    ax.yaxis.set_tick_params(direction='in', which = 'both')


def errswarmplot(ax, y, yerr, colors = None):
    """add errors on top of swarm plot"""
    xax = [[i]*len(y_) for i, y_ in enumerate(y)]
    #import pdb; pdb.set_trace()
    error_kw = {'capsize': 5, 'capthick': 1.5, 'elinewidth': 1.5,
                'ecolor': 'darkgrey', 'barsabove': False, 'fmt': 'none',
                'alpha': 1}
    for xax_, y_, yerr_, c_ in zip(xax, y, yerr, colors):
        #_ =ax.errorbar(np.array(xax).flatten(),
        #                np.array(y).flatten(), np.array(yerr).flatten(),
        #                **error_kw)
        error_kw.update({'ecolor': c_})
        #import pdb; pdb.set_trace()
        _ = ax.errorbar(xax_, y_, yerr_, **error_kw)
    return ax

def swarmplot(  ax, x, xlabs = None, ylab = None, colors = None, xerr = None,
                show_mean = False, add_box = True, how ='swarm', swkwargs = {}):
    """seaborn swarmplot"""

    swkwargs_ = {
        'alpha': 1,
        'size': 5,
    }
    swkwargs_.update(swkwargs)

    if colors is None:
        colors = ['#9467bd', '#2ca02c', '#ff7f0e']
        #colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(x)]
        if len(x) > 3:
            colors2 = plt.rcParams['axes.prop_cycle'].by_key()['color']
            colors2 = [c2 for c2 in colors2 if c2 not in colors]
            colors += colors2
            #colors = np.roll(np.array(colors), 1)
    elif len(colors) == 1:
        colors = colors * len(x)
    colors = colors[:len(x)]

    if add_box:
    #ax = sns.violinplot(ax = ax, data = x, saturation = 1, cut = 2, inner = None)
        boxprops = {
            'boxprops':{'facecolor':'none', 'edgecolor':'darkgrey'},
            'medianprops':{'color':'darkgrey'},
            'whiskerprops':{'color':'darkgrey'},
            'capprops':{'color':'darkgrey'},
            'fliersize': 0
        }
        ax = sns.boxplot(   ax = ax, data = x, whis = 3, color = "white",
                            saturation = .2, width = .5, **boxprops)
    # iterate over boxes
    # for i,box in enumerate(ax.artists):
    #     box.set_edgecolor('black')
    #     box.set_facecolor('white')
    #     plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
    #     plt.setp(ax.lines, color='k')

    if show_mean:
        ax = sns.pointplot( ax = ax, data = x, estimator=np.nanmean, color = 'darkgrey',
                           join = False, ci = None, markers = "-", scale = 0.5)
    if how.lower().startswith('swarm'):
        # https://stackoverflow.com/a/64636825
        try:
            ax = sns.swarmplot(ax = ax, data = x, **swkwargs_)
        except Exception as e:
            how = 'point'
    elif how.lower().startswith('snspoint'):
        ax = sns.pointplot( ax = ax, data = x, estimator=np.nanmean,
                            errorbar = 'se',
                            join = False, ci = None, markers = "o", scale = 3)
    elif how.lower().startswith('point'):
        raise NotImplementedError('Swarmplot from pyplot points not implemented')

    for color, collection in zip(colors, ax.collections[-len(x):]):
        collection.set_facecolor(color)
        #import pdb; pdb.set_trace()

    if xerr:
        ax = errswarmplot(ax, x, xerr, colors)
    try:
        if np.mean(np.concatenate(x)) > 1e3: ax.set_yscale('log')
    except ValueError as e:
        if np.mean(x) > 1e3: ax.set_yscale('log')


    #sample size:
    #ax.set_ylim(bottom = 0)
    label_sample_size(ax, x, y = .5)
    # beatify axes
    if xlabs is not None: set_xaxis_labels(ax, xlabs, start = 0, do_rotate = False)
    if ylab is not None: name_ax(ax, ylab = ylab)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(axis = 'both', direction = 'in')
    ax.xaxis.set_tick_params(direction='in',  which = 'both')
    ax.yaxis.set_tick_params(direction='in', which = 'both')
    return ax

def get_whiskers(x, whis = 1.5):
    """Get whiskers position from data

    lower: Q1 - whis*(Q3-Q1)
    upper: Q3 + whis*(Q3-Q1)

        Q1-whis*IQR   Q1   median  Q3   Q3+whis*IQR
                      |-----:-----|
      o      |--------|     :     |--------|    o  o
                      |-----:-----|
    flier             <----------->            fliers
                           IQR

    Reference:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html
    """
    x = np.sort(x)
    q1, q3 = np.quantile(x, (.25, .75))
    lw_ = q1 - whis*(q3-q1)
    uw_ = q3 + whis*(q3-q1)

    # the lower whisker is at the lowest datum above lw_
    lw = x[np.argwhere(x >= lw_)[0]]
    # the upper whisker at the highest datum below uw_
    uw = x[np.argwhere(x <= uw_)[-1]]
    return lw[0], uw[0] # return as tuple of floats

def first_nonzero_index(x):
    """
    Return the index of the first non-zero element of array x.
    If all elements are zero, return -1.
    """

    fnzi = -1 # first non-zero index
    indices = np.flatnonzero(x)

    if (len(indices) > 0):
        fnzi = indices[0]

    return fnzi

def annotate_stat(ax, data, pos = None, text="*"):
    """
     Annotate plot with result of statistical test.
    """
    if not pos:
        pos = ax.get_xaxis()
    y = np.max(np.concatenate(data))
    h = y * 0.1
    ax.plot(np.repeat(pos, 2), [y+h, y+2*h, y+2*h, y+h], lw = 1.5, c ='k')
    ax.text((pos[0]+pos[1])/2, y+2*h, text, ha = 'center', va = 'bottom', color = 'k')

## EXPANSION PLOTS

def grayscale_plot_markers():
    """an iterator over marker/color combinations"""
    import itertools
    colors = ['#000000', '#bebebe', '#6a6a6a']
    marks = ["D", "^", "s", ".", "o"]
    # return endless iterator
    return itertools.cycle(map(''.join, itertools.product(colors, marks)))


def plot_expansion_scatter(ax, plotpairs, legends):
    """Plot all indvidual measurements"""
    markersmap = grayscale_plot_markers()

    for i, (x, y) in plotpairs:
        imarker = next(markersmap)
        ax.plot(x, y, c = imarker[:-1], marker = imarker[-1], ls = "",
                markersize = 3, alpha = 0.8, label = str(legends[i]))

    # Put a legend to the right of the current axis
    leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol = 2, fontsize = 6)
    #leg = ax.legend(loc='lower right', ncol = 2, fontsize = 8)
    leg.get_frame().set_alpha(0)

    return ax

def plot_expansion_shaded(ax, x, y, yerr, kwargs_mean = {}, kwargs_err = {}):
    """Plot mean and confidence interval"""

    kwargs_mean_ = {
        'color': 'red',
        'ls': '-',
        'lw': 2.5,
        'alpha': 1.0,
    }
    kwargs_mean_.update(kwargs_mean)
    kwargs_err_ = {
        'color':  kwargs_mean_['color'],
        'alpha': 0.1,
        'ls': '-',
    }
    kwargs_err_.update(kwargs_err)


    ax.plot(x, y, **kwargs_mean_)
    ax.fill_between(x, y - yerr, y + yerr, **kwargs_err_)
    return ax

def plot_expansion(data, legends = None):
    """wrapper plot function"""
    fig, ax = make_fig(fig_kw = {"tight_layout": True})
    set_plot_style()

    if isinstance(data[0][1], (tuple, )):
        ax = plot_expansion_scatter(ax, data, legends)
    else:
        ax = plot_expansion_shaded(ax, *data)

    # make plot look nice
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1]*1.05)
    #plots.name_ax(ax, "$t\ [min]$", "$Rg\ [\mu m]$")
    name_ax(ax, "$t\ [min]$", "$Rg/Rg_0\ [-]$")
    #plots.name_ax(ax, "$t\ [min]$", "$No.\ clusters$")

    ylim = ax.get_ylim()
    #ax.set_ylim(ylim[0], 1.50)

    return ax
