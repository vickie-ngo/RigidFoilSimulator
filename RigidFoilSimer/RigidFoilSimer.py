from RigidFoilSimer import Parameters, talkToAnsys, CFile_Generation, processData, GraphGenerator
import numpy as np
import sys
import shutil
import os

def yesNo(prompt):
    if Parameters.query_yes_no(prompt) == False:
        sys.exit("Done.")

def main(files, geo, dyn, axs=2, x=2, CollectData = False):
    """Runs simulation from reading in input form to processing end data"""
    
    if os.path.isdir(files.data_path) == False:
        Folder_Path = Parameters.path_check(files.folder_path, "\nStore simulation files to %s?\nA) Yes, use/create the folder and save to it \nB) No, I want to specify a different folder directory \nC) No, I want to cancel this process\nPick an answer of A, B, or C: ", 0)
        files.newFolderPath(Folder_Path)

        ## Generate Journal Files
        talkToAnsys.generateMesh_wbjn(files, geo, 1)
        talkToAnsys.generateFluent_wbjn(files, dyn, 1)

        if hasattr(files, 'WB_path'):
            talkToAnsys.run_wbjn(files.WB_path, files.wbjnMesh_path, '-B')
            yesNo("Project with Mesh file has been generated. Begin simulation? (This will take a long time)")
            ## Generate C File
            files = CFile_Generation.genCFile(files, geo, dyn)        
            talkToAnsys.run_wbjn(files.WB_path, files.wbjnFluent_path, '-B')
    
    dataString = np.hstack(([['Geometry_Name', 'Radius_of_Curvature', 'Leading_y', 'Leading_x', 'Reduced_Frequency'],[geo.geo_name, geo.radius_of_curvature, geo.leading_ellipse_y, geo.leading_ellipse_x, dyn.reduced_frequency]], processData.main(files, dyn, geo, axs, x, CollectData)))
    try:
        files.dataOutput = np.vstack((files.dataOutput, dataString[1,:]))
    except AttributeError:
        files.dataOutput = dataString