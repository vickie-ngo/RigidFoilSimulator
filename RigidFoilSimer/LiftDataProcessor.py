import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

def readData(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if (line.strip().split())[0].isdigit():
                cols = np.array([float(i) for i in line.strip().split()])
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
            print(len(mainstring))
            return i - len(mainstring)
    return -1

steps_per_cycle = 1000
f = 1.6
f_s = steps_per_cycle/f

file_path = r'C:\Users\vicki\Google Drive\Lab CFD\ProcessedData\LiftData\GEO1_PM_012.txt'
# file_path = r'C:\Users\vicki\Google Drive\Lab CFD\ProcessedData\LiftData\GEO1_PM_008.txt'
variables, data = readData(file_path)
variables = np.hstack((variables, ['Cycle Number']))
data = np.hstack((data, np.transpose([np.floor(data[:,0]/steps_per_cycle)])))
cycle_count = np.unique(data[:,-1])

lift_variable_index = index_containing_substring(variables, 'lift')
fig, axs = plt.subplots(ncols = 2, nrows = 2, figsize=(10.5, 6.5), constrained_layout = True)
fig.suptitle('k = 0.08')
for cycle in cycle_count:
    cycle_data = data[data[:,-1] == cycle, :]
    if cycle_data.shape[0] > 1:
        axs[0,0].plot(cycle_data[:,0]%steps_per_cycle/steps_per_cycle, cycle_data[:,lift_variable_index], label = 'Cycle = ' + str(cycle))    
    else:
        cycle_data = data[data[:,-1] == cycle -1, :]
axs[0,0].legend()
axs[0,0].set(title = 'CFD Results')

lift = cycle_data[:,lift_variable_index]
X = fftpack.fft(lift)
freqs = fftpack.fftfreq(len(lift)) * f_s
axs[0,1].stem(freqs, np.abs(X), use_line_collection = True)
axs[0,1].scatter(freqs[np.abs(X)>2], (np.abs(X))[np.abs(X)>2], marker = 'x', color = 'r')
axs[0,1].set(xlabel = 'Frequency in Hertz [Hz]',xlim = (0, 30), ylabel = 'Frequency Domain (Spectrum) Magnitude', ylim = (-5, 110), title = 'CFD Frequency Domain')
# axs[0,1].set_xlim(0, 30)
# axs[0,1].set_ylim(-5, 110)
plt.show()
