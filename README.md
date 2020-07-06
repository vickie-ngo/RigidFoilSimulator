[![DOI](https://zenodo.org/badge/257502195.svg)](https://zenodo.org/badge/latestdoi/257502195)


# RigidFoilSimulator
Python-based oscillating rigid foil simulation package enables novice users to get meaningful computational fluid dynamics results through a streamlined command line interface.

## Installation
This package can be installed via:

    pip install git+https://github.com/SoftwareDevEngResearch/RigidFoilSimulator
    
[See instructions for Environment Variables setup here](#Environment-Variables-Setup)  

## Dependencies
The simulation itself is ran through the ANSYS Workbench software package and therefore relies on the successful installation of ANSYS and its dependencies. Some of the dependencies include:
  1. This package was developed using Windows, use with caution when working on other systems
  2. ANSYS 2019 or later. 
  3. Visual Studio 2017. VS 2019 is not yet compatible.

Installing the package also requires Git.  
    
  4. Git
  
## Warnings and Limitations
**It is not recommended to run the Rigid Foil Simulator package without prior working knowledge** of how computational fluid dynamics is performed. Output results should be analyzed with discretion before being used to expand and inform on scientific understanding.

## Usage Examples
The example integrated into this package is of a NACA0015-_like_ airfoil at a reduced frequency of k=0.08. To use the rigid foil simulation package, start by defining the 3 class objects to your workspace and pick one of the 5 options that follow.
    
    from RigidFoilSimer import Parameters
    
    filepaths = Parameters.FilePath(r <path to folder*>, <folder name**> )
    Geo = Parameters.Geometry()
    Dyn = Parameters.Dynamics()

\* Required, for example, "C:\Users\<username>\Desktop" will save an example folder to the desktop. 

\*\* If the folder name is left as the default, "RigidFoilSimer_Example", the package WILL treat it as an example so be sure to change the folder name for non-example operations.

### Usage #1: Input Parameters, run Journals and get Processed Data (Recommended)
Option #1 is a single statement that runs Options #2-#5 all together. Import the main code module to the workspace and add the module statement:

    from RigidFoilSimer import RigidFoilSimer
    
    RigidFoilSimer.main(filepaths, Geo, Dyn)

**If ANSYS is not installed,** the example case will create an example folder and store all new files into the folder. The example will proceed to run an example case of post-processing on existing data to show what the output would look if the simulation had been completed.

**If ANSYS is installed,** the example case will do the same things as stated previously, but will also run the first 5 time steps of the simulation within ANSYS (note: post-processing of the example does not use data generated from the example simulation, as simulations take over 10hrs of run time to reach the time of interest)

### Usage #2: Create C-File
Import the CFile_Generation module and add the module statement:

    from RigidFoilSimer import CFile_Generation
    
    CFile_Generation.genCFile(filepaths, Geo, Dyn)

A \*.c file will be generated in the defined folder.

### Usage #3: Create ANSYS Journals
There are two different ANSYS journals (\*.wbjn) that are built off templates: the Mesh generation template and the Fluent simulation template. Having two separate journals allows the user to just get the mesh file if that is what they want, rather than having to run the entire simulation first. For ANSYS related functions, import the talkToAnsys module:

    from RigidFoilSimer import talkToAnsys
    
Generate the Journal to create an ANSYS Fluent project and a \*.msh file with the command
    
    talkToAnsys.generateMesh_wbjn(filepaths, Geo)
    
Likewise, generate the Journal to run the Fluent calculations with the command
    
    talkToAnsys.generateFluent_wbjn(filepaths, Dyn)

For the purpose of this example, the default number of time steps to run the simulation is 5.

### Usage #4: Run Pre-Existing Journals
This option only works if the associated Journals already exist and their directories have been defined within the filepaths class and, more importantly, if ANSYS has been installed. To run the Mesh Journal and the Fluent Journal, the statements are

    talkToAnsys.run_wbjn(filepaths.WB_path, filepaths.wbjnMesh_path, '-B')
    talkToAnsys.run_wbjn(filepaths.WB_path, filepaths.wbjnFluent_path, '-B')
    
respectively. Caution must be taken if the Journals are not stored in the same place the Project will be / has been generated, that was not what the package was designed to navigate. The output is then stored in the <project name>_files folder generated by ANSYS.

### Usage #5: Post-Processing Data
If the data has already been generated, the location of the file can be identified using

    filepaths = Parameters.FilePath(r <path to folder***>, <folder name****> )

and the following module and statement will determine the points of interest:

    from RigidFoilSimer import processWallShear
    
    processWallshear.wallshearData(filepaths.data_path, Dyn)

For running the example case, the output is demonstrated using data that is included with the package.

\*\*\* Required, for example, "C:\Users\<username>\Desktop", but for the example case, this directory will not be generated or referenced.

\*\*\*\* If the folder name is left as the default, "RigidFoilSimer_Example", the package WILL treat it as an example so be sure to change the folder name for non-example operations.

# Environment Variables Setup
To ensure that ANSYS and Visual Studio are installed correctly, check to verify the following environment variables are correctly called out:

| Variable     | Allowed Values                                                                                         |
|--------------|--------------------------------------------------------------------------------------------------------|
| `FLUENT_INC` | C:\Program Files\\\<ANSYS Application Name>\v\<version number>\fluent                                     |
| `INCLUDE`    | C:\Program Files (x86)\Microsoft Visual Studio\\\<Year>\Enterprise\VC\Auxiliary\VS\include               |
| `LIB`        | C:\Program Files (x86)\Microsoft Visual Studio\\\<Year>\Enterprise\VC\Auxiliary\VS\lib                   |
| `Path`       | C:\Program Files (x86)\Microsoft Visual Studio\\\<Year>\Enterprise                                       |
|              | C:\Program Files\\\<ANSYS Application Name>\v\<version number>\fluent\ntbin\win64                         |
|              | C:\Program Files (x86)\Microsoft Visual Studio\\\<Year>\Enterprise\VC\Tools\MSVC\14.24.28314\bin\Hostx64 |
|              | C:\Program Files (x86)\Microsoft Visual Studio\\\<Year>\Enterprise\VC\Tools\MSVC\14.24.28314\bin\Hostx86 |
|              | C:\Users\\\<username>\AppData\Local\Programs\Git\cmd                                                     |
