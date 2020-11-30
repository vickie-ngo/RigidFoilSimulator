import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import fsolve
from sympy import symbols, Eq, solve
import pickle
import sys, os
import shutil
import __main__


pi = np.pi
cos = np.cos

class FilePath(object):
    """Establishes paths that are referenced throughout the package"""
    def __init__(self, folder_parentpath, folder_name="RigidFoilSimer_Example", project_name="Project_Example", default = 0):
        self.folder_path = path_check((folder_parentpath + "\\" + folder_name).replace("/","\\"), "\nStore simulation files to %s?\nA) Yes, use/create the folder and save to it \nB) No, I want to specify a different folder directory \nC) No, I want to cancel this process\nPick an answer of A, B, or C: ", default)
        self.folder_name = folder_name
        self.project_path = (self.folder_path + "\\" + project_name).replace("/","\\")
        self.project_name = project_name
        self.wbjnMesh_path = self.project_path + "_genFileGeomMesh.wbjn"
        self.wbjnFluent_path = self.project_path + "_genFileFluent.wbjn"
        if 'google' in self.project_path.lower():
            self.data_path = self.project_path     
        elif 'none' in self.project_name.lower():
            self.data_path = self.folder_path
        else:
            self.data_path = self.project_path + r"_files\dp0\FFF\Fluent"
            
        self.org_path = 'None'

        fluent_path = shutil.which("fluent")
        if fluent_path == None:
            print("IMPORTANT: ANSYS Fluent application could not be found. The rest of this package will operate without interacting with live simulations until ANSYS is installed and file paths are reestablished.")
        else:
            self.WB_path = fluent_path[0:int(fluent_path.find("fluent"))] + r"Framework\bin\Win64\RunWB2.exe"
            self.version = int(fluent_path.split('\\')[-5][1:])
        if self.folder_name == "RigidFoilSimer_Example":
            self.data_path =  os.path.dirname(os.path.realpath(__file__)) + r"\Tests\Assets"
    
    def newFolderPath(self, folder_path):
        self.folder_path = folder_path.replace("/","\\")
        self.project_path = (self.folder_path + "\\" + self.project_name).replace("/","\\")
        self.wbjnMesh_path = (self.project_path + "_genFileGeomMesh.wbjn").replace("\\","/")
        self.wbjnFluent_path = self.project_path + "_genFileFluent.wbjn"
        if not self.data_path == os.path.dirname(os.path.realpath(__file__)) + r"\Tests\Assets":
            self.data_path =  self.project_path + r"_files\dp0\FFF\Fluent"
        
    def __repr__(self):
        output = ("\nFile Paths: \n \
        Folder path : \t\t % s \n \
        Project path : \t % s \n \
        Data path : \t\t % s \n \
        " % (self.folder_path, self.project_path, self.data_path))
        if hasattr(self, 'WB_path'):
            output = output + "Workbench path : \t % s " % (self.WB_path)
        return output

class Geometry(object):
    """Foil geometry conditions are used to explore different sizes and shapes"""
    
    def __init__(self, chord=0.15, leading_ellipse_y = 0.15*0.075, leading_ellipse_x = 0.15*0.3, trailing_ellipse_y = 0.001, trailing_ellipse_x=0.006):
        """Initializes the main parameters for DesignModeler, default parameters are for the flat rigid plate geometry"""
        self.geo_name = 'None'
        self.leading_ellipse_y = leading_ellipse_y
        self.leading_ellipse_x = leading_ellipse_x
        self.leading_ellipse_origin = -chord/2 + self.leading_ellipse_x
        self.trailing_ellipse_y = trailing_ellipse_y
        self.trailing_ellipse_x = trailing_ellipse_x
        self.trailing_ellipse_origin = chord/2 - self.trailing_ellipse_x
        self.chord = chord
        
        # These equations are necessary if you decide to switch to SpaceClaim Geometry Scripting. 
        # These are not required for parameterization
        #Solve for tangent lines to the leading and trailing edge ellipses
        m, k = symbols('m k')
        eq1 = Eq((self.leading_ellipse_x**2*k*m - self.leading_ellipse_origin*self.leading_ellipse_y**2)**2 - (self.leading_ellipse_y**2+self.leading_ellipse_x**2*m**2)*(self.leading_ellipse_origin**2*self.leading_ellipse_y**2+self.leading_ellipse_x**2*k**2-self.leading_ellipse_y**2*self.leading_ellipse_x**2),0)
        eq2 = Eq((self.trailing_ellipse_x**2*k*m - self.trailing_ellipse_origin*self.trailing_ellipse_y**2)**2 - (self.trailing_ellipse_y**2+self.trailing_ellipse_x**2*m**2)*(self.trailing_ellipse_origin**2*self.trailing_ellipse_y**2+self.trailing_ellipse_x**2*k**2-self.trailing_ellipse_y**2*self.trailing_ellipse_x**2),0)
        sol_dict = solve((eq1,eq2),(m,k))
  
        # Define the equation for tangent lines
        x, y = symbols('x y')
        eqT = Eq(sol_dict[1][0]*x+sol_dict[1][1]-y,0)
        
        # Solve for the intersection point at the leading edge ellipse
        eqE = Eq((x-self.leading_ellipse_origin)**2/self.leading_ellipse_x**2 + y**2/self.leading_ellipse_y**2 - 1,0)
        sol_xy = solve((eqT,eqE),(x,y))
        self.leading_ellipse_xT = abs(sol_xy[0][0])
        self.leading_ellipse_yT = abs(sol_xy[0][1])
        
        # Solve for the intersection point at the trailing edge ellipse
        eqE = Eq((x-self.trailing_ellipse_origin)**2/self.trailing_ellipse_x**2 + y**2/self.trailing_ellipse_y**2 - 1,0)
        sol_xy = solve((eqT,eqE),(x,y))
        self.trailing_ellipse_xT = abs(sol_xy[0][0])
        self.trailing_ellipse_yT = abs(sol_xy[0][1])

        # Solve for tangent angle
        # Jordan is awesome :)
        if not self.leading_ellipse_yT - self.trailing_ellipse_yT == 0:
            self.tangentline_angle = np.arctan(float((self.trailing_ellipse_yT - self.leading_ellipse_yT)/(self.trailing_ellipse_xT+self.leading_ellipse_xT)))
        else:
            self.tangentline_angle = 0
        
    def find_theta_t(self, shed_x, shed_y):
        ## finds the tangent angle along the foil at (shed_x,shed_y) relative to the chord line
        x = shed_x-self.chord/2
        if x <= -self.leading_ellipse_xT:
            self.theta_t = np.arctan(-float((self.leading_ellipse_y**2*(x-self.leading_ellipse_origin))/(self.leading_ellipse_x**2*shed_y)))
        elif x <= self.trailing_ellipse_xT:
            self.theta_t = self.tangentline_angle
            if shed_y < 0:
                self.theta_t = -self.theta_t
        else:
            self.theta_t = np.arctan(-float((self.trailing_ellipse_y**2*(x-self.trailing_ellipse_origin))/(self.trailing_ellipse_x**2*shed_y)))
        return self.theta_t
    
    def find_r(self, shed_x, shed_y):
        x = shed_x-self.chord/2
        self.r1 = np.sqrt((self.leading_ellipse_origin - x)**2 + (shed_y)**2)*shed_y/abs(shed_y)
        self.r2 = np.sqrt(x**2 + shed_y**2)*shed_y/abs(shed_y)
        self.theta_r2 = np.tan(-shed_y/x)
    
    def __repr__(self):
        import numpy.random as rnd
        from matplotlib.patches import Ellipse

        ells = [Ellipse(xy=np.array([self.leading_ellipse_origin, 0]), width=2*self.leading_ellipse_x, height=2*self.leading_ellipse_y, angle=0),
                Ellipse(xy=np.array([self.trailing_ellipse_origin, 0]), width=2*self.trailing_ellipse_x, height=2*self.trailing_ellipse_y, angle=0)]
        fig = plt.figure(0)
        ax = fig.add_subplot(111, aspect='equal')
        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(1)
            e.set_edgecolor('tab:blue')
            e.set_facecolor('none')
        ax.plot([-self.leading_ellipse_xT,self.trailing_ellipse_xT],[self.leading_ellipse_yT, self.trailing_ellipse_yT], color='tab:blue')    
        ax.plot([-self.leading_ellipse_xT,self.trailing_ellipse_xT],[-self.leading_ellipse_yT, -self.trailing_ellipse_yT], color='tab:blue')    
        ax.set_xlim(-self.chord/2, self.chord/2)
        ax.set_ylim(-self.chord/2, self.chord/2)
        plt.axis('off')
        plt.show()
        return "Foil Geometry Parameters [M]: \n \
        chord length : \t % s \n \
        LE height : \t\t % s \t\t\n \
        LE width : \t\t % s \t\t\n \
        LE Tangent y : \t % s \t\t\n \
        LE Tangent x : \t % s \t\t\n \
        TE height : \t\t % s \t\t\n \
        TE width : \t\t % s \t\t\n \
        TE Tangent y : \t % s \t\t\n \
        TE Tangent x : \t % s \t\t\n \
        " % (self.chord, self.leading_ellipse_y, self.leading_ellipse_x, self.leading_ellipse_yT, -self.leading_ellipse_xT, self.trailing_ellipse_y, self.trailing_ellipse_x, self.trailing_ellipse_yT, self.trailing_ellipse_xT)

      
class Dynamics(object):
    """Foil parameters are all parameters involved in the motion generation"""
    # class body definition
    
    def __init__(self, k=0.12, total_cycles=0.003, plot_steps=2, chord = 0.15, f=1.6, h0=0.075, theta0=70, steps_per_cycle=1000, density=1.225):
        self.reduced_frequency = k
        self.freq = f                    
        self.theta0 = np.radians(theta0)
        self.steps_per_cycle = steps_per_cycle
        self.dt = 1/(f*steps_per_cycle)
        self.total_cycles = total_cycles
        #self.T = round(total_cycles/f,6)
        self.rho = density                          #fluid density
        self.chord = chord
        self.velocity_inf = f*chord/k
        self.h0 = h0
        self.just_steps = int(np.ceil(round(total_cycles/f,6)/self.dt)) 
        self.plot_steps = plot_steps
        samp = np.array([x for x in range(self.just_steps + self.plot_steps +1)])
        self.time = [round(x*self.dt,5) for x in samp]
        self.h = [self.h0*cos(2*pi*x/steps_per_cycle)-self.h0 for x in samp]
        self.theta = [self.theta0*cos(2*pi*x/steps_per_cycle+pi/2) for x in samp]
        self.h_dot = [2*pi*f*self.h0*cos(2*pi*f*self.time[x]+pi/2) for x in samp]
        self.theta_dot = [2*pi*f*self.theta0*cos(2*pi*f*self.time[x]) for x in samp]
        self.alpha_eff = [self.theta[x] - np.arctan(self.h_dot[x]/self.velocity_inf) for x in samp]
        self.relations = {}
    
    def update_totalCycles(self, total_cycles, plot_steps):   
        self.just_steps = int(np.ceil(round(total_cycles/self.freq,6)/self.dt)) 
        self.plot_steps = plot_steps
        samp = np.array([x for x in range(self.just_steps + self.plot_steps +1)])
        self.time = [round(x*self.dt,5) for x in samp]
        self.h = [self.h0*cos(2*pi*x/steps_per_cycle)-self.h0 for x in samp]
        self.theta = [self.theta0*cos(2*pi*x/steps_per_cycle+pi/2) for x in samp]
        self.h_dot = [2*pi*f*self.h0*cos(2*pi*f*self.time[x]+pi/2) for x in samp]
        self.theta_dot = [2*pi*f*self.theta0*cos(2*pi*f*self.time[x]) for x in samp]
        
    def e_theta(self, timestep):
        return self.theta0*cos(2*pi*timestep/self.steps_per_cycle+pi/2)
        
    def e_h_dot(self, timestep):
        return 2*pi*self.freq*self.h0*cos(2*pi*self.freq*timestep*self.dt+pi/2)
    
    def e_theta_dot(self, timestep):
        return 2*pi*self.freq*self.theta0*cos(2*pi*self.freq*timestep*self.dt)

    def __repr__(self):
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Cycles [-]', fontsize=18)
        ax1.set_ylabel('Heave Position [m]', color=color, fontsize=18)
        ax1.plot(np.asarray(self.time)/(1/np.asarray(self.freq)), self.h, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Pitching Angle [rad]', color=color, fontsize=18)  # we already handled the x-label with ax1
        ax2.plot(np.asarray(self.time)/(1/np.asarray(self.freq)), self.theta, '--', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        plt.title('Heaving and Pitching Profiles Across 3 Cycles', fontsize=18)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
        
        return "Foil Dynamic Parameters: \n \
        reduced frequency [-]: \t % s \n \
        chord length [M]: \t\t % s \n \
        heaving frequency [Hz]: \t % s \n \
        heaving amplitude [M]: \t % s \n \
        pitching amplitude [rad]: \t % s \n \
        steps per cycle [N]: \t\t %s \n \
        total cycles [-]: \t\t %s \n \
        fluid density [kg/m^3]: \t %s \n \
        " % (self.reduced_frequency, self.chord, self.freq , self.h0, self.theta0, self.steps_per_cycle, self.total_cycles, self.rho)

def relation_eqns(FoilDyn, FoilGeo, term, time_step, xy):
    time = (time_step % FoilDyn.steps_per_cycle)/ FoilDyn.steps_per_cycle
    FoilDyn.theta_inf_hdot = np.arctan(-FoilDyn.e_h_dot(time_step)/FoilDyn.velocity_inf)
    xC = xy[0]/FoilDyn.chord
    eff_AoA = FoilDyn.e_theta(time_step) - FoilDyn.theta_inf_hdot
    FoilDyn.theta_t  = FoilGeo.find_theta_t(xy[0], xy[1])
    FoilDyn.theta_txy = FoilDyn.e_theta(time_step) - FoilDyn.theta_t
    FoilGeo.find_r(xy[0], xy[1])
    FoilDyn.theta_p_r2 = FoilDyn.e_theta(time_step) + FoilGeo.theta_r2
    FoilDyn.u_thetadot = FoilGeo.r2*FoilDyn.e_theta_dot(time_step)
    # u_thetadot is the pitching velocity at the vortex shed location [magnitude, x, y]
    FoilDyn.u_thetadot = [FoilDyn.u_thetadot, FoilDyn.u_thetadot*np.sin(FoilDyn.theta_p_r2), FoilDyn.u_thetadot*np.cos(FoilDyn.theta_p_r2)]
    FoilDyn.theta_inf_thetadot = np.arctan(-FoilDyn.u_thetadot[2]/(FoilDyn.velocity_inf - FoilDyn.u_thetadot[1]))
    FoilDyn.theta_inf_hdot_thetadot = np.arctan(-(FoilDyn.u_thetadot[2]+FoilDyn.e_h_dot(time_step))/(FoilDyn.velocity_inf - FoilDyn.u_thetadot[1]))
    Alpha_eff = FoilDyn.e_theta(time_step)-FoilDyn.theta_inf_hdot
    Alpha_inf_hdot =  FoilDyn.theta_txy-FoilDyn.theta_inf_hdot
    Alpha_inf_thetadot = FoilDyn.theta_txy-FoilDyn.theta_inf_thetadot
    Alpha_inf_hdot_thetadot = FoilDyn.theta_txy-FoilDyn.theta_inf_hdot_thetadot
    FoilDyn.relations = ['_Time_(t/T)','_Position_Along_Chord_(x/C)','_Pitching_Angle_(rad)','_Tangent_Angle_(rad)','_Theta_inf_+_hdot_(rad)','_Theta_inf_+_thetadot_(rad)','_Theta_inf_+_hdot_+_thetadot_(rad)','_Alpha_eff','_Alpha_inf_+_hdot','_Alpha_inf_+_thetadot','_Alpha_inf_+_hdot_+_thetadot','_r1','_r2']
    FoilDyn.relations = np.vstack(([term + headers for headers in FoilDyn.relations], np.array([time, xC, FoilDyn.e_theta(time_step), FoilDyn.theta_txy, FoilDyn.theta_inf_hdot, FoilDyn.theta_inf_thetadot, FoilDyn.theta_inf_hdot_thetadot, Alpha_eff, Alpha_inf_hdot, Alpha_inf_thetadot, Alpha_inf_hdot_thetadot, FoilGeo.r1, FoilGeo.r2])))
    return FoilDyn.relations  

def query_yes_no(question, default=None):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        if specialCase() == False:
            sys.stdout.write(question + prompt)
            choice = input().lower()
        else:
            choice = "y"
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def path_check(path, prompt, default):
    """figure out whether file exists and if so, how to handle it"""
    while True:
        if specialCase() == True:
            data = 'a'
        elif default == 1:
            data = 'd'
        else:
            data = input(prompt % (path))
        if data.lower() not in ('a', 'b', 'c', 'd'):
            print("Not an appropriate choice.")
        elif data.lower()=='a':
            try:
                os.mkdir(path)
            except OSError:
                #print ("Creation of the directory %s failed" % path)
                if os.path.exists(path):
                    if query_yes_no("\nFolder already exists, is it okay to replace existing files?")==False:
                        path = input("\nEnter the full path of the folder you would like the file to be saved w/o quotations: ")
                    else:
                        break
                else:    
                    sys.exit("\nDirectory for the simulation files could not be created/processed. Please check your directory inputs in the input form")
            else:
                print ("\nSuccessfully accessed the directory, %s " % path)
            break
        elif data.lower()=='b':
            path = input("\nEnter the full path of the folder you would like the file to be saved w/o quotations: ")
        elif data.lower()=='c':
            sys.exit("\nDirectory needs to be defined in order to proceed")
            
        elif data.lower()=='d':
            if not os.path.exists(path):
                if query_yes_no("\nOutput folder does not exist, enter new file path?")==True:
                    path = input("\nEnter the full path of the folder you would like the file to be saved w/o quotations: ")
                else:
                    sys.exit("Data could not be found.")
            else:
                break
    return path
    
def specialCase():
    if hasattr(__main__, '__file__'):  
        if "test" in __main__.__file__.lower() or "batch" in __main__.__file__.lower():
            return True
        else:
            return False
    else:
        return False
