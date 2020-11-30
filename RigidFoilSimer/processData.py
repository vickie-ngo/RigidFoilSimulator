import os, sys
import numpy as np
from . import Parameters, GraphGenerator
from scipy.interpolate import interp1d, interp2d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl

def convert_2_txt(file_path):
    """Identifies if file needs to be converted to txt"""
    if (file_path.find(".txt") < 0):
        new_path = file_path + ".txt"
        os.rename(file_path, new_path)
        file_path = new_path
    return file_path 
    
def add_data_columns(file_path, chord, theta, h, cutoff):
    """Check to see if new columns of rotated data have been added and add if needed"""    
    file_object = open(file_path,"r")
    headers = file_object.readline()
    variable_names = np.array(headers.replace(",", " ").replace("_","-").strip().split())
    var_count = len(variable_names)   
    x_wallshear_col = np.where(variable_names == "x-wall-shear")
    y_wallshear_col = np.where(variable_names == "y-wall-shear")
    if (headers.find("x-rotated")<0):
        #If this header column does not exist, it means the data has not yet been processed       
        c, s= np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))        
        variable_names = [np.append(variable_names, np.array(['x-rotated', 'y-rotated', 'calculated-wallshear']))]
        data = variable_names
        scatterPlot = np.empty((0,3))
        
        for line in file_object:
            # Get data from each line and calculate the rotated position
            cols = np.array([float(i) for i in line.replace(","," ").strip().split()])
            xy = cols[1:3]
            cols[2] = cols[2] - h
            xyR = np.dot(R, cols[1:3]) + [chord/2, 0]      
            # Filter to only collect data for the leading edge of the correct surface
            top_bottom = int(1 if xyR[1] > 0 else -1)
            frontal_region = int(1 if xyR[0] < cutoff *chord else -1) 
            if top_bottom*theta > 0 and frontal_region == 1: 
                wallshear = cols[x_wallshear_col]*np.cos(theta)-cols[y_wallshear_col]*np.sin(theta)
                cols = np.concatenate((cols, xyR, wallshear))
                data = np.append(data, [cols], axis=0) 
            scatterPlot = np.vstack((scatterPlot, np.array([xyR[0], xyR[1], top_bottom]))) 
            
        x_rotated_col = var_count
        
        # Sort data 
        set_data = data[1:,:].astype(float)
        sorted_data = set_data[set_data[:, var_count].argsort()]
        final_data = np.append(variable_names, sorted_data, axis=0)
        
    else:
        final_data = [np.array(variable_names)]
        x_rotated_col = int(np.where(variable_names == "x-rotated")[0])
        for line in file_object:
            cols = np.array([float(i) for i in line.replace(","," ").strip().split()])
            
            # Filter to only collect data for the leading edge of the correct surface
            top_bottom = int(1 if cols[-2] > 0 else -1)
            frontal_region = int(1 if cols[-3] < cutoff *chord else -1)
                
            if top_bottom*theta > 0 and frontal_region == 1:              
                final_data = np.append(final_data, [cols], axis=0)  
    
    file_object.close()
    return final_data, scatterPlot

def main(Files, FoilDyn, FoilGeo, axs, plot_col=1, dataOutput = False, cutoff = 0.15):
    """Go into wall shear folder and process raw data"""
    FoilDyn.cutoff = cutoff
    data_path = Files.data_path
    print('\n' + Files.project_name)
    
    if Files.org_path == 'None':
        savePath = Files.folder_path+"\\_mod-"
    else:
        savePath = Files.org_path + "\\" + FoilGeo.geo_name + "-" + "{:.2f}".format(FoilDyn.reduced_frequency).replace(".","") + "-"

    if os.path.isdir(data_path) == True:
        file_names = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
        file_names = list(filter(lambda x:(x.find("les") >= 0 or x.find("wall") >= 0 or x.find("wss") >= 0), file_names))

        if data_path == os.path.dirname(os.path.realpath(__file__)) + r"\Tests\Assets":
            FoilDyn.update_totalCycles(2,0)
            modfiles = list(filter(lambda x:(x.find("mod-") >= 0), file_names))
            for x in modfiles:
                os.remove(data_path+"\\"+x)
            file_names = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
            file_names = list(filter(lambda x:(x.find("les") >= 0 or x.find("wall") >0 or x.find(FoilGeo.geo_name) >= 0), file_names))
        
        if len(file_names) > 0:
            FoilDyn.tau0_database = np.empty([0,4])
            ws_ct, dp_ct, plotting_range = 0, 0, 5
            space, window_size, poly_order = 1000, 11, 1
            
            file_names = sorted(file_names)
            last_time_step = int(file_names[-1].split('-')[-1].split('.')[0])
            if last_time_step > round(last_time_step, -3):
                start_time_step = round(last_time_step, -3)   
            else:
                start_time_step = round(last_time_step, -3) - FoilDyn.steps_per_cycle
            
            times = [int(file_names[f].split('-')[-1].split('.')[0]) for f in range(len(np.array(file_names)))]
            
            for x in range(len(file_names)):
                time_step = times[x]

                if time_step > start_time_step and time_step < start_time_step + 200 and round(FoilDyn.theta[time_step],3) != 0: # and time_step % 10 == 0:
                    if ws_ct<plotting_range or dp_ct<plotting_range or time_step % 5 == 0:
                        file_path = convert_2_txt(data_path+"\\"+file_names[x])
                        (final_data, scatterPlot) = add_data_columns(file_path, FoilDyn.chord, FoilDyn.theta[time_step], FoilDyn.h[time_step], 1)
                        # np.savetxt(savePath + str(time_step) + '.txt', final_data[:-1,:], fmt="%s")
                        # processed data is of rotated x, rotated y, and calculated wallshear data
                        processed_data = final_data[1:,-3:].astype(float)
                        processed_LE = final_data[1:,:].astype(float)
                        processed_LE = processed_LE[processed_LE[:,-3] <= FoilDyn.cutoff*FoilDyn.chord, :]
                        
                        wallshear = processed_LE[1:, -1]
                        if np.min(wallshear) < 0 and wallshear[0] > 0 and ws_ct < plotting_range:
                            if ws_ct == 0:
                                ws_time = time_step
                            ws_ct = ws_ct + 1
                        
                        pressure_name = "pressure-coefficient"
                        pressure_term = np.where(final_data[0,:] == pressure_name)[0]
                        if pressure_term.size > 0:
                            if FoilDyn.tau0_database.shape[-1] < 5:
                                FoilDyn.tau0_database = np.insert(FoilDyn.tau0_database, 4, 0, axis = 1)
                                FoilDyn.dp_database = np.empty([0,5])
                                FoilDyn.dpdx_max = np.empty([0,5])
                            processed_data = np.append(processed_data, final_data[1:,pressure_term].astype(float), axis=1)
                            p = processed_LE[:,pressure_term]
                            x = processed_LE[:,-3]
                            y = processed_LE[:,-2]
                            x_mid_panel = (x[:-1] + x[1:])/2
                            y_mid_panel = (y[:-1] + y[1:])/2
                            dp = p[1:]-p[:-1]
                            dx = x[1:]-x[:-1]
                            dy = y[1:]-y[:-1]
                            xx = np.linspace(x_mid_panel.min(), x_mid_panel.max(), space)
                            yy = np.linspace(y_mid_panel.min(), y_mid_panel.max(), space)
                            itp = interp1d(x_mid_panel, (np.squeeze(dp)*FoilDyn.chord/dx), kind='linear')
                            # Calculate smoothed curve of dpdx
                            dpdx_filt = savgol_filter(itp(xx), window_size, poly_order)
                            # popt, pcov = curve_fit(GraphGenerator.pressure, xx, itp(xx), maxfev=10000)
                            # dpdx_filt = GraphGenerator.pressure(xx, *popt)
                            ddpdx = dpdx_filt[1:]-dpdx_filt[:-1]
                            dp_data = np.column_stack((xx, yy, itp(xx), dpdx_filt, np.full((space,1),time_step).astype(int)))
                            FoilDyn.dp_database = np.vstack((FoilDyn.dp_database, dp_data))
                            if np.argmax(dpdx_filt) < round(space*0.5) and ddpdx[np.argmax(dpdx_filt)] < 0 and ddpdx[0] > 0 and dpdx_filt[0] < 0 and dp_ct < plotting_range:# and np.max(dpdx_filt) > 0
                                if dp_ct == 0:
                                    dp_time = time_step
                                    # xy = dp_data[:,:2].astype(float)[np.argmax(dpdx_filt)]
                                    # pressure_details = Parameters.relation_eqns(FoilDyn, FoilGeo, 'dpdx', ws_ct, xy) 
                                dp_ct = dp_ct + 1
                        processed_data = np.append(processed_data, np.full((processed_data.shape[0],1), time_step).astype(int), axis=1)
                        FoilDyn.tau0_database = np.append(FoilDyn.tau0_database, processed_data, axis=0)
                        
            desired_steps = np.unique(FoilDyn.tau0_database[FoilDyn.tau0_database[:,-1]% 10 == 0,-1])
            desired_steps = desired_steps[np.logical_and(desired_steps%1000>25,desired_steps<ws_time + 20)].astype(int)
            size = len(desired_steps)
            for step in range(size):
                if desired_steps[step] != 0:
                    tau0_filtered = FoilDyn.tau0_database[np.logical_and(FoilDyn.tau0_database[:,0] <= FoilDyn.chord*FoilDyn.cutoff, FoilDyn.tau0_database[:,-1]==desired_steps[step]),:]
                    dpdx_filtered = FoilDyn.dp_database[np.logical_and(FoilDyn.dp_database[:,0] <= FoilDyn.chord*FoilDyn.cutoff, FoilDyn.dp_database[:,-1]==desired_steps[step]),:]            
                    tT = (desired_steps[step] % FoilDyn.steps_per_cycle)/FoilDyn.steps_per_cycle
                    # FoilGeo.find_r(tau0_filtered[:,0], tau0_filtered[:,1])
                    # tau0_r2 = 1/2+FoilGeo.r2/FoilDyn.chord
                    # FoilGeo.find_r(dpdx_filtered[:,0], dpdx_filtered[:,1])
                    # dpdx_r2 = 1/2+FoilGeo.r2/FoilDyn.chord
                    axs[0, plot_col].plot(tau0_filtered[:,0]/FoilDyn.chord, tau0_filtered[:,2], label = tT, color = (220/255, 68/255, 0, (step+2)/20))
                    # axs[0, plot_col].plot(tau0_r2,tau0_filtered[:,2], label = tT)
                    axs[1, plot_col].plot(dpdx_filtered[:,0]/FoilDyn.chord, dpdx_filtered[:,3], label = tT, color = (220/255, 68/255, 0, (step+2)/20))
                    # axs[1, plot_col].plot(dpdx_r2, dpdx_filtered[:,3], label = tT)
            plt.setp(axs, xlim=[0, 0.15])
            axs[0, plot_col].set_ylim([-0.1, 0.8])  
            axs[1, plot_col].set_ylim([-50, 150])   
            
        else:
            print('\n' + Files.project_name + ' does not have data')

        ## Tau data
        desired_steps = np.arange(ws_time - 1, ws_time + 1)
        filtered = FoilDyn.tau0_database[(FoilDyn.tau0_database[:,-1]>= ws_time-1) & (FoilDyn.tau0_database[:,-1] <= ws_time+1) & (FoilDyn.tau0_database[:,0] <= FoilDyn.chord*FoilDyn.cutoff)][1:,[0,1,2,-2,-1]]
        x = np.linspace(max([filtered[filtered[:,-1] == step, 0].min() for step in desired_steps]), min([filtered[filtered[:,-1] == step, 0].max() for step in desired_steps]), space)
        y_interp = interp1d(filtered[:,0], filtered[:,1])
        y = y_interp(x)
        interp_array = np.empty([0,space])
        interp_array2 = interp_array
        point_array = np.empty([0,2])
        for step in range(len(desired_steps)):
            step_filt = filtered[filtered[:,-1]==desired_steps[step],:]
            interp_ws = interp1d(step_filt[:,0], step_filt[:,2])
            interp_dp = interp1d(step_filt[:,0], step_filt[:,3])
            interp_array = np.vstack((interp_array, interp_ws(x)))
            interp_array2 = np.vstack((interp_array2, interp_dp(x)))
            point_array = np.vstack((point_array,[x[np.argmin(interp_ws(x))], (interp_ws(x))[np.argmin(interp_ws(x))] ]))
        slope = -point_array[0,1]/(point_array[1,1]-point_array[0,1])
        interp_array = interp_array[0,:600] + (interp_array[1,:600]-interp_array[0,:600])*slope
        interp_array2 = interp_array2[0,:] + (interp_array2[1,:]-interp_array2[0,:])*slope
        ws_xy = [x[np.argmin(interp_array)], y[np.argmin(interp_array)], interp_array[np.argmin(interp_array)], interp_array2[np.argmin(interp_array)]]
        axs[0, plot_col].plot(x[:600]/FoilDyn.chord, interp_array, 'k')
        axs[0, plot_col].scatter(ws_xy[0]/FoilDyn.chord, ws_xy[2], marker='x', s=50, c='k')
        ws_time = desired_steps[0] + slope
        wallshear_details = np.hstack((Parameters.relation_eqns(FoilDyn, FoilGeo, 'tau0', ws_time, ws_xy[0:2]), [['tau0_wallshear','tau0_P'],[ws_xy[2], ws_xy[3]]]))
        tau0_headline = FoilGeo.geo_name + ', \nt/T=' + str(round((ws_time % FoilDyn.steps_per_cycle)/FoilDyn.steps_per_cycle, 3)) + ', \nx = ' + str(round(ws_xy[0]/FoilDyn.chord,4))
        axs[0, plot_col].set(xlabel = 'Position along Chord, [x/C]', ylabel='Wall Shear', title=tau0_headline)
        
        ## dpdx data             
        desired_steps = np.arange(min(ws_time, dp_time), max(ws_time, dp_time))
        for step in desired_steps:
            dpdx_filtered = FoilDyn.dp_database[np.logical_and(FoilDyn.dp_database[:,0] <= FoilDyn.chord*FoilDyn.cutoff, FoilDyn.dp_database[:,-1]==step),:]
            FoilDyn.dpdx_max = np.vstack((FoilDyn.dpdx_max, dpdx_filtered[np.argmax(dpdx_filtered[:,3]),:]))
        FoilGeo.find_r(FoilDyn.dpdx_max[:,0],FoilDyn.dpdx_max[:,1])
        goal_x, goal_dpdx = GraphGenerator.log_trend(FoilDyn.dpdx_max[:,0]/FoilDyn.chord, FoilDyn.dpdx_max[:,3], axs, plot_col)
        # goal_x, goal_dpdx = GraphGenerator.log_trend((1/2+FoilGeo.r2/FoilDyn.chord), FoilDyn.dpdx_max[:,3], axs, plot_col)
        axs[1, plot_col].scatter(goal_x, goal_dpdx, marker='x', s= 50, c = 'k')
        
        desired_steps = np.unique(FoilDyn.dp_database[FoilDyn.dp_database[:,-1] < ws_time,-1])
        FoilDyn.dp_database[:,0] = FoilDyn.dp_database[:,0]/FoilDyn.chord
        for step in desired_steps:
            x1_less = FoilDyn.dp_database[(FoilDyn.dp_database[:,-1] == step) & (FoilDyn.dp_database[:,0] < goal_x),:][-1,[0, 3]]
            x1_more = FoilDyn.dp_database[(FoilDyn.dp_database[:,-1] == step) & (FoilDyn.dp_database[:,0] > goal_x),:][0,[0, 3]]
            x2_less = FoilDyn.dp_database[(FoilDyn.dp_database[:,-1] == step + 1) & (FoilDyn.dp_database[:,0] < goal_x),:][-1,[0, 3]]
            x2_more = FoilDyn.dp_database[(FoilDyn.dp_database[:,-1] == step + 1) & (FoilDyn.dp_database[:,0] > goal_x),:][0,[0, 3]]
            if (x1_less[-1] < goal_dpdx or x1_more[-1] < goal_dpdx) and (x2_less[-1] > goal_dpdx or x2_more[-1] > goal_dpdx):
                desired_steps = np.arange(step, step + 2)
                break

        filtered = FoilDyn.dp_database[(FoilDyn.dp_database[:,-1]>= desired_steps[0]) & (FoilDyn.dp_database[:,-1] <= desired_steps[1]) & (FoilDyn.dp_database[:,0] <= FoilDyn.cutoff)][:,[0,1,3,-1]]
        x = np.linspace(max([filtered[filtered[:,-1] == step, 0].min() for step in desired_steps]), min([filtered[filtered[:,-1] == step, 0].max() for step in desired_steps]), space)
        y_interp = interp1d(filtered[:,0], filtered[:,1])
        y = y_interp(x)
        interp_array = np.empty([0,space])
        point_array = np.empty([0,2])
        for step in desired_steps:
            step_filt = filtered[filtered[:,-1]==step,:]
            interp_eqn = interp1d(step_filt[:,0], step_filt[:,2])
            interp_array = np.vstack((interp_array, interp_eqn(x)))
            point_array = np.vstack((point_array,[goal_x, interp_eqn(goal_x) ])) 
        slope = (goal_dpdx-point_array[0,1])/(point_array[1,1]-point_array[0,1])
        interp_array = interp_array[0,:] + (interp_array[1,:]-interp_array[0,:])*slope
        dp_xy = [x[np.argmax(interp_array)], y[np.argmax(interp_array)], interp_array[np.argmax(interp_array)]]
        axs[1, plot_col].plot(x, interp_array, 'k')
        dp_time = desired_steps[0] + slope
        pressure_details = np.hstack((Parameters.relation_eqns(FoilDyn, FoilGeo, 'dpdx', dp_time, [dp_xy[0]*FoilDyn.chord, dp_xy[1]]), [['dpdx_max'],[dp_xy[2]]]))
        dpdx_headline = 't/T=' + str(round((dp_time % FoilDyn.steps_per_cycle)/FoilDyn.steps_per_cycle, 3)) + ', \nx = ' + str(round(dp_xy[0],4))
        axs[1, plot_col].set(xlabel='Position along Chord, [x/C]', ylabel='dP/dx', title=dpdx_headline)
        
        plt.draw()
        
    else:
        print('\n' + Files.project_name + ' is not a folder')

    try:
        shed_time
    except NameError:
        shed_time = 0
        x_wallshear = -1
        print("Vortex has not shed within the simulated time line.")
  
    if dataOutput == True:
        return np.hstack((wallshear_details, pressure_details))
              
    return shed_time, x_wallshear
