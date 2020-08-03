import os, sys
import numpy as np
from . import Parameters
import matplotlib.pyplot as plt

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
        
        data = np.empty((0,var_count+3))
        variable_names = [np.append(variable_names, np.array(['x-rotated', 'y-rotated', 'calculated-wallshear']))]
        data = variable_names
        
        for line in file_object:
            # Get data from each line and calculate the rotated position
            cols = np.array([float(i) for i in line.replace(","," ").strip().split()])
            cols[2] = cols[2] - h
            xyR = np.dot(R, cols[1:3]) + [chord/2, 0]
            
            # Filter to only collect data for the leading edge of the correct surface
            top_bottom = int(1 if xyR[1] > 0 else -1)
            frontal_region = int(1 if xyR[0] < cutoff *chord else -1)
    
            if top_bottom*theta > 0 and frontal_region == 1: 
                wallshear = cols[x_wallshear_col]*np.cos(theta)-cols[y_wallshear_col]*np.sin(theta)
                cols = np.concatenate((cols, xyR, wallshear))
                data = np.append(data, [cols], axis=0) 
                
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
    return final_data

def wallshearData(Files, FoilDyn, FoilGeo, cutoff = 0.2):
    """Go into wall shear folder and process raw data"""
    
    data_path = Files.data_path
    
    if Files.org_path == 'None':
        savePath = Files.folder_path+"\\_mod-"
    else:
        savePath = Files.org_path + "\\" + FoilGeo.geo_name + "-" + "{:.2f}".format(FoilDyn.reduced_frequency).replace(".","") + "-"
    
    file_names = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    file_names = list(filter(lambda x:(x.find("les") >= 0 or x.find("wall") >= 0 or x.find("wss") >= 0), file_names))

    if data_path == os.path.dirname(os.path.realpath(__file__)) + r"\Tests\Assets":
        FoilDyn.update_totalCycles(2,0)
        modfiles = list(filter(lambda x:(x.find("mod-") >= 0), file_names))
        for x in modfiles:
            os.remove(data_path+"\\"+x)
        file_names = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
        file_names = list(filter(lambda x:(x.find("les") >= 0 or x.find("wall") >0 or x.find("wss") >= 0 or x.find(FoilGeo.geo_name) >= 0), file_names))
    
    temp_database = np.empty([0,4])
    ct = 0
    file_names = sorted(file_names)
    last_time_step = int(file_names[-1].split('-')[-1].split('.')[0])
    if last_time_step > round(last_time_step, -3):
        start_time_step = round(last_time_step, -3)   
    else:
        start_time_step = round(last_time_step, -3) - 1000

    for x in range(len(file_names)):
        if ct == 4 or x == len(file_names)-1:
            fig, axs = plt.subplots(4)
            desired_steps = np.unique(temp_database[:,-1]).astype(int)[-9:]
            temp_set = np.empty([0,4])
            filtered_data = np.empty([0,4])
            for step in desired_steps:
                filtered_data = temp_database[temp_database[:,-1]==step,:]
                axs[0].plot(filtered_data[:,0]/FoilDyn.chord,filtered_data[:,2], label = step)
                temp_x = temp_database[temp_database[:, -1] == step,:][np.argmin(temp_database[temp_database[:, -1] == step,2]),0]
                temp_y = temp_database[temp_database[:, -1] == step,:][np.argmin(temp_database[temp_database[:, -1] == step,2]),1]
                temp_ws = np.min(temp_database[temp_database[:, -1] == step,2])
                temp_set = np.append(temp_set, [[step, temp_x, temp_y, temp_ws]], axis=0)
            axs[1].plot(temp_set[:,0], temp_set[:,3])
            axs[2].plot(temp_set[:,0], temp_set[:,2]/FoilDyn.chord)
            axs[0].legend()
            axs[0].grid()
            axs[1].grid()
            axs[2].grid()
            fig.suptitle(FoilGeo.geo_name + ", k = " + str(FoilDyn.reduced_frequency))
            axs[0].set(xlabel='x position along the chord, [x/C]', ylabel='Wall Shear')
            axs[1].set(xlabel='time step [s]', ylabel='Wall Shear')
            axs[2].set(xlabel='time step [s]', ylabel='x position along the chord, [x/C]')
            axs[3].plot(shed[:,0].astype(float), tang)
            axs[3].grid()
            
            if Parameters.specialCase() == False:  
                print("Exit plots to end procedure")
                plt.draw()   
            break
            
        file_path = convert_2_txt(data_path+"\\"+file_names[x])
        time_step = int(file_names[x].split('-')[-1].split('.')[0])
        
        if time_step > start_time_step and time_step < start_time_step + 1000 and round(FoilDyn.theta[time_step],3) != 0: # and time_step % 10 == 0:
            final_data = add_data_columns(file_path, FoilDyn.chord, FoilDyn.theta[time_step], FoilDyn.h[time_step], 1)
            # np.savetxt(savePath + str(time_step) + '.txt', final_data[:-1,:], fmt="%s")
            final_data = final_data[1:,:].astype(float)
            processed_data = np.transpose(np.append([final_data[:,-3]], [final_data[:,-2], final_data[:,-1]], axis=0))
            processed_data2 = np.append(processed_data[processed_data[:,0] <= cutoff*FoilDyn.chord ], np.full((processed_data[processed_data[:,0] <= cutoff*FoilDyn.chord ].shape[0],1), time_step).astype(int) , axis=1)
            temp_database = np.append(temp_database, processed_data2, axis=0)
            wallshear = processed_data2[1:,-2].astype(float)
            
            if np.min(wallshear) < 0 and wallshear[0] > 0 and ct < 4:
                if ct == 0: 
                    shed_time = time_step
                    shed_x = processed_data2[1:,0].astype(float)[np.argmin(wallshear)]
                    shed_y = processed_data2[1:,1].astype(float)[np.argmin(wallshear)]
                    shed_wallshear = wallshear
                    x_wallshear = shed_x/FoilDyn.chord
                    FoilDyn.theta_inf_hdot = np.arctan(-FoilDyn.h_dot[time_step]/FoilDyn.velocity_inf)
                    eff_AoA = FoilDyn.theta[time_step] - FoilDyn.theta_inf_hdot
                    shed = processed_data[:,:2].astype(float)
                    theta_tp = [FoilGeo.find_tangent(x[0], x[1]) for x in shed]
                    tang = FoilDyn.theta[time_step] - [FoilGeo.find_tangent(x[0], x[1]) for x in shed]
                    FoilDyn.theta_t = FoilDyn.theta[time_step] - FoilGeo.find_tangent(shed_x,shed_y)
                    FoilGeo.find_r(shed_x, shed_y)
                    FoilDyn.theta_p_r2 = FoilDyn.theta[time_step] + FoilGeo.theta_r2
                    FoilDyn.u_thetadot = FoilGeo.r2*FoilDyn.theta_dot[time_step]
                    # u_thetadot is the pitching velocity at the vortex shed location [magnitude, x, y]
                    FoilDyn.u_thetadot = [FoilDyn.u_thetadot, FoilDyn.u_thetadot*np.sin(FoilDyn.theta_p_r2), FoilDyn.u_thetadot*np.cos(FoilDyn.theta_p_r2)]
                    FoilDyn.theta_inf_thetadot = np.arctan(-FoilDyn.u_thetadot[2]/(FoilDyn.velocity_inf - FoilDyn.u_thetadot[1]))
                    FoilDyn.theta_inf_hdot_thetadot = np.arctan(-(FoilDyn.u_thetadot[2]+FoilDyn.h_dot[time_step])/(FoilDyn.velocity_inf - FoilDyn.u_thetadot[1]))
                    print('\n' + Files.project_name)
                    print("\nOutput Results:\n\tVortex is shed at time step = %s\n\tVortex Position = %s" % (shed_time,x_wallshear))
                    print('\nTheta values:\n\tPitching Angle = %s\n\tTangent Angle = %s\n\tInf+h_dot = %s\n\tInf+theta_dot = %s\n\tInf+h_dot+theta_dot = %s' % (FoilDyn.theta[time_step],FoilDyn.theta_t,FoilDyn.theta_inf_hdot,FoilDyn.theta_inf_thetadot,FoilDyn.theta_inf_hdot_thetadot))
                    print('\nr values:\n\tr1 = %s\n\tr2 = %s' % (FoilGeo.r1, FoilGeo.r2))
                ct = ct + 1

    try:
        shed_time
    except NameError:
        shed_time = 0
        x_wallshear = -1
        print("Vortex has not shed within the simulated time line.")
      
    return shed_time, x_wallshear
