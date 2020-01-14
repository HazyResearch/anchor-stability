import matplotlib
import matplotlib.pyplot as plt
import pandas 

markers = ['s', 'v', '^', '<', '>', 'o']

def plt_single(df, vals, val_tag, xtag, dist, ylog=False, ylabel='', xlabel='', title='', val_tag_label='', legend=False, color='C4', marker='s', line_label=None):
    if val_tag_label == '':
        val_tag_label=val_tag
    for i, val in enumerate(vals): 
        df_sub = df.loc[df[val_tag] == val]
        xval = df_sub[xtag]
        yval = df_sub[(dist, 'mean')]
        if ('overlap' in dist and 'sym' not in dist and 'anchor' not in dist) or 'knn' in dist: 
                yval = 1 - yval 
        yerr = df_sub[(dist, 'std')]
        if len(vals) == 1: 
            if line_label is None: 
                line_label = f'{val_tag_label}={val}'
            plt.errorbar(xval, yval, yerr, label=line_label, capsize=5, marker=marker, linestyle='--', color=color)
        else:
            plt.errorbar(xval, yval, yerr, label=f'{val_tag_label}={val}', capsize=5, marker=markers[i], linestyle='--')
        if xtag == 'dim': 
            plt.xscale('log')
            plt.minorticks_off()
            plt.xticks([25, 50, 100, 200, 400, 800])
            plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        elif xtag == 'bitrate':
            plt.xscale('log', basex=2)
            plt.minorticks_off()
            plt.xticks([1, 2, 4, 8, 16, 32])
            plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        elif xtag == 'space':
            plt.xscale('log')
    plt.title(title)
    if legend:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylog: 
        plt.yscale('log')

def plt_correlations(df, values, val_tag, metrics, xmetric, ylog=False, ylabel='', title='', xlabel='', xlog=False, legend=False):
    if len(metrics) > 1: 
        fig = plt.figure(figsize=(20,30))
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
    num = 321 
    for dist in metrics: 
        if len(metrics) > 1: 
            plt.subplot(num)
        for i,val in enumerate(values): 
            df_sub = df.loc[df[val_tag] == val]
            xval = df_sub[(xmetric, 'mean')]
            yval = df_sub[(dist, 'mean')]
            yerr = df_sub[(dist, 'std')]
            if ('overlap' in xmetric and 'sym' not in xmetric and 'anchor' not in xmetric) or 'knn' in xmetric: 
                xval = 1 - xval 
            plt.errorbar(xval, yval, yerr, label=f'b={val}', capsize=5, marker=markers[i], linestyle='--')
        if xlog: 
            plt.xscale('log')
            plt.minorticks_off()
        plt.title(title)
        if legend: 
            plt.legend(ncol=2)
        if xlabel == '':
            xlabel = xmetric
        plt.xlabel(xlabel)
        plt.ylabel('% Disagreement')
        num += 1 
        if ylog: 
            plt.yscale('log')

# load csv results
def plt_csv(xlabel, filepath):
    df = pandas.read_csv(filepath)
    plt.errorbar(df['Disagreement|x'], df['Disagreement|y'], df['Disagreement|y_std'], capsize=5, marker='s', linestyle='--', color='C4')
    if xlabel == 'Precision': 
        plt.xscale('log', basex=2)
        plt.minorticks_off()
        plt.xticks([1, 2, 4, 8, 16, 32])
        plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    else:
        plt.xscale('log')
        plt.minorticks_off()
        plt.xticks([192, 384, 768, 1536, 3072])
        plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlabel(xlabel)
    plt.ylabel('% Disagreement')
