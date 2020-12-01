from RigidFoilSimer import Parameters, talkToAnsys, CFile_Generation, processData, GraphGenerator
import matplotlib.pyplot as plt
import sys
import shutil

def yesNo(prompt):
    if Parameters.query_yes_no(prompt) == False:
        sys.exit("Done.")

def main(FilePath, FoilGeo, FoilDyn, axs=2, x=2):
    """Runs simulation from reading in input form to processing end data"""

    Folder_Path = Parameters.path_check(FilePath.folder_path, "\nStore simulation files to %s?\nA) Yes, use/create the folder and save to it \nB) No, I want to specify a different folder directory \nC) No, I want to cancel this process\nPick an answer of A, B, or C: ", 0)
    FilePath.newFolderPath(Folder_Path)

    ## Generate Journal Files
    talkToAnsys.generateMesh_wbjn(FilePath, FoilGeo, 1)
    talkToAnsys.generateFluent_wbjn(FilePath, FoilDyn, 1)

    if hasattr(FilePath, 'WB_path'):
        talkToAnsys.run_wbjn(FilePath.WB_path, FilePath.wbjnMesh_path, '-B')
        yesNo("Project with Mesh file has been generated. Begin simulation? (This will take a long time)")
        ## Generate C File
        FilePath = CFile_Generation.genCFile(FilePath, FoilGeo, FoilDyn)        
        talkToAnsys.run_wbjn(FilePath.WB_path, FilePath.wbjnFluent_path, '-B')
    
    # processData.main(FilePath, FoilDyn, FoilGeo, axs, x)