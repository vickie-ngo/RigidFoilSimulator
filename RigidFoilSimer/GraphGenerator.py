import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import fftpack, interpolate, signal
import pandas as pd
import lmfit
import math
import os

def linear(x, m, b):
    return m*x+b
    
def parabolic(x, a, b, c):
    return a*np.power(x,2)+b*x+c

def cubic(x, a, b, c, d):
    return a*np.power(x,3)+b*np.power(x,2)+c*x+d

def asymptotic(x, a, b, c, d): 
    return a*np.exp(b*x)+c*np.exp(d*x)

def powerLaw(x, a, b, c, d):
    # return a*np.power(x,b)+c*np.power(x,d)
    return a*np.power(x,b)+c

def logCurve(x, a, b, c): 
    # return (a*x*np.log(x) + b )/(c*x*np.log(x) + d)
    # return (a)/(b*x+c)
    # return (a*x*np.log(x) + b )/(c*x*np.log(x)+ f*x + d)
    # return (x+a)/(x+b)
    return a*np.exp(b*x) + c

def pressure(x, a, b):
    return 1/(1+(1/a-1)*np.exp(x))

def cubicRatio(x, a, b, c, d):
    return a*np.power(np.log(x),3)+b*np.power(np.log(x),2)+c*np.log(x)+d

def is_integer(n):
    try:
        float(n)
        return float(n)
    except ValueError:
        return n

def readData(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            stripped_data = line.strip().split()
            if isinstance(is_integer((stripped_data)[-1]),float):
                cols = np.array([is_integer(i) for i in stripped_data])
                if len(cols) != data.shape[-1]:
                    data = np.empty([0,len(cols)])
                data = np.vstack((data, cols))
            else:
                if '(' in line:
                    variables = [''.join(e for e in i if e.isalnum()) for i in line.strip().split()]
                    data = np.empty([0, len(variables)])
                continue
    return variables, data

def index_containing_substring(mainstring, substring):
    for i, s in enumerate(mainstring):
        if substring in s:
            return i - len(mainstring)
    return -1

def drawTrendlines(file_name, x_term, x_label, y_term, y_label, legend_term, outlier = range(0,8), coeff_count = 2):
    xint = 0.005
    yint = 0.01
    
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    pol_coeff = np.empty((0,coeff_count))

    ## Reading Data
    variables, data = readData(file_name + '.txt')    
    col_leg = variables.index(''.join(e for e in legend_term if e.isalnum()))
    col_x = variables.index(''.join(e for e in x_term if e.isalnum()))
    col_y = variables.index(''.join(e for e in y_term if e.isalnum()))
    leg_set = np.unique(data[:,col_leg])

    ## Generating Plot
    fig = plt.figure(figsize=(10.5, 6.5), dpi = 500)
    axs = fig.add_axes([0.1, 0.2, 0.45, 0.75])

    for sets in range(leg_set.shape[0]):
        plot_data = data[data[:,col_leg] == leg_set[sets],:]
        plot_data = plot_data[plot_data[:,col_x].argsort()]
        x = plot_data[:,col_x].astype(float)
        y = plot_data[:,col_y].astype(float)
        axs.plot(x[outlier],y[outlier], marker='o', linestyle = 'None', color = color[sets], label = 'k='+leg_set[sets])
        if coeff_count == 2:
            popt, pcov = curve_fit(linear, x[outlier], y[outlier])
            axs.plot(x, linear(x, *popt), '--', color = color[sets]) #, label='fit: slope=%5.3f, y-intercept=%5.3f' % tuple(popt))
        elif coeff_count == 3:
            popt, pcov = curve_fit(parabolic, x[outlier], y[outlier])
            axs.plot(x, parabolic(x, *popt), '--', color = color[sets]) #, label='fit: slope=%5.3f, y-intercept=%5.3f' % tuple(popt))
        pol_coeff = np.append(pol_coeff, [popt], axis = 0)
    
    axs.set_xlabel(x_label, fontsize=18)
    plt.xticks(np.arange(data[:,col_x].astype(np.float).min()-data[:,col_x].astype(np.float).min()%xint, max(data[:,col_x].astype(np.float))+xint, xint))
    axs.set_ylabel(y_label, fontsize=18)
    plt.yticks(np.arange(data[:,col_y].astype(np.float).min()-data[:,col_y].astype(np.float).min()%yint, data[:,col_y].astype(np.float).max()+yint, yint))
    axs.grid()
    axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)
    # plt.title(y_term.replace('_',' ') + ' vs. ' + x_term.replace('_',' '), fontsize=18)
    png_name = file_name+'_'+ ''.join(e for e in y_term if e.isalnum()) + '_vs_' + ''.join(e for e in x_term if e.isalnum())
    fig.savefig(png_name + '.png', transparent=True)
    plt.close()
    return png_name, legend_term, leg_set, pol_coeff

def curveTrend(file_name, legend_term, leg_set, pol_coeff):

    fig = plt.figure(figsize=(10.5, 6.5), dpi = 500)
    axs = fig.add_axes([0.1, 0.2, 0.45, 0.75])
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    leg_set = leg_set.astype(float)

    axs.plot(leg_set, pol_coeff[:,0], marker='o', linestyle = 'None', color = color[0], label='slope trend')
    popt, pcov = curve_fit(linear, leg_set, pol_coeff[:,0])
    axs.plot(leg_set, linear(leg_set, *popt), '--', color = color[0], label='fit: slope=%5.3f, y-int=%5.3f' % tuple(popt))
    axs.legend(bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0., fontsize=13)
    axs.set_ylabel('slope', fontsize=18)

    ax2 = axs.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(leg_set, pol_coeff[:,1], marker='o', linestyle = 'None', color = color[1], label='y-intercept trend')
    popt, pcov = curve_fit(linear, leg_set, pol_coeff[:,1])
    ax2.plot(leg_set, linear(leg_set, *popt), '--', color = color[1], label='fit: slope=%5.3f, y-int=%5.3f' % tuple(popt))
    ax2.legend(bbox_to_anchor=(1.15, 0.85), loc='upper left', borderaxespad=0., fontsize=13)
    ax2.set_ylabel('y-intercept', fontsize=18)

    axs.set_xlabel('$k$', fontsize=18)
    fig.tight_layout()
    plt.title('Trend vs. $k$', fontsize=18)
    fig.savefig(file_name+'_curvetrends.png', transparent=True)

def log_trend(x, y, axs, plot_col):
    # poly_coeff = np.polyfit(np.log(x), y, 1)
    # xx = np.linspace(x.min(), x.max(), 100)
    # yy = poly_coeff[0]*np.log(xx) + poly_coeff[1]

    # poly_coeff = np.polyfit(x, y, 3)
    # poly_eqn = np.poly1d(poly_coeff)
    # xx = np.linspace(x.min(), x.max(), 100)
    # yy = poly_eqn(xx)

    model = logCurve
    popt, pcov = curve_fit(model, x, y, bounds = ([0, -np.inf, y.min()-0.1],[np.inf, 0, y.min()+0.1]) , maxfev=500000)
    xx = np.linspace(x.min(), 0.15, 1000)
    yy = model(xx, *popt)
    # axs[-1, plot_col].scatter(x, y, marker = '.', color = 'b')
    axs[1, plot_col].plot(xx, yy, color='b')
    slope_goal = -100
    x_goal = (1/popt[1])*np.log(slope_goal/(popt[0]*popt[1]))
    y_goal = model(x_goal, *popt)
    return x_goal, y_goal

def liftDataProcessing(files, dyn, geo, axs, x):
    steps_per_cycle = dyn.steps_per_cycle
    f_s = steps_per_cycle/dyn.freq

    file_path = files.data_path + '\lift-rfile.out'
    # file_path = r'C:\Users\vicki\Google Drive\Lab CFD\ProcessedData\LiftData\GEO1_PM_008.txt'
    variables, data = readData(file_path)
    variables = np.hstack((variables, ['Cycle Number']))
    data = np.hstack((data, np.transpose([np.floor(data[:,0]/steps_per_cycle)])))
    cycle_count = np.unique(data[:,-1])

    lift_variable_index = index_containing_substring(variables, 'lift')
    
    for cycle in cycle_count:
        cycle_data = data[data[:,-1] == cycle, :]
        if cycle_data.shape[0] > 1 and cycle == 2:
            axs[0,x].plot(cycle_data[:,0]%steps_per_cycle/steps_per_cycle, cycle_data[:,lift_variable_index], linewidth = 5, label = ' CFD Cycle = ' + str(cycle+1))    
            forceData = np.hstack((np.vstack(cycle_data[:,0]%steps_per_cycle/steps_per_cycle), np.vstack(cycle_data[:,lift_variable_index])))
        else:
            cycle_data = data[data[:,-1] == cycle -1, :]
            
    axs[0,x].legend()
    axs[0,x].set(title = geo.geo_name,xlabel = '$Time$ [t/T]', ylabel = '$C_L$')

    lift = cycle_data[:,lift_variable_index]
    X = fftpack.fft(lift)
    freqs = fftpack.fftfreq(len(lift)) * f_s
    axs[1,x].stem(freqs, np.abs(X), use_line_collection = True)
    axs[1,x].scatter(freqs[np.abs(X)>2], (np.abs(X))[np.abs(X)>2], marker = 'x', color = 'r')
    axs[1,x].set(xlabel = 'Frequency in Hertz [Hz]',xlim = (0, 30), ylabel = 'Frequency (Spectrum) Magnitude', ylim = (-5, 110), title = 'CFD Frequency Domain')   
            
# def apply_lmfit(x, y, model1, model2=):