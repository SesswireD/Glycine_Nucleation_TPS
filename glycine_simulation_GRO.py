import os
import numpy as np

def minimize_system(system):
    """Minimizes a molecular system using Gromacs. Parameters are defined in Data/Input/Simulation/minim.mdp.
    This operation assembles the .gro and .top file of a system together with the minimization instructions defined in the .mdp.
    and does the minimization procedure. As a result the following files are created:

    {system}_minim.tpr
        result of gmx grompp. Assembly of .top, .gro and .mdp. Used as input to gmx mdrun. option -o
    {system}_minim.mdp 
        result of gmx grommp. Contains all parameters that were used during minimization.
    {system}_minim.gro
        result of gmx mdrun. Resulting coordinates after minimization. option -c
    {system}_minim.edt
        result of gmx mdrun. Contains energies, the temperature, pressure, etc. option -e
    {system}_minim.log
        result of gmx mdrun. Contains a log of each step during minimization. option -g
    {system}_minim.trr
        result of gmx mdrun. Contains the entire trajectory of minimization. option -o
    
    PARAMETERS:
    -----------
    system: str
        The complete path to the system.
    """

    #Separate input path from system
    input_path, system = os.path.split(system)
    input_path += "/"

    #Simulation output is saved in a different folder
    output_path = "Data/Output/Simulation/"

    #Assemble .gro and .top with parameters in .mdp for  minimization
    os.system(f"gmx grompp -f Data/Input/Simulation/minim.mdp -c {input_path+system}.gro -p {input_path+system}.top -o {output_path+system}_minim.tpr -po {output_path+system}_minim")

    #Run the assembly 
    print(f"Minimizing system {system}...")
    os.system(f"gmx mdrun -s {output_path+system}_minim.tpr -c {output_path+system}_minim -e {output_path+system}_minim -g {output_path+system}_minim -o {output_path+system}_minim")


def equilibrate_system(system):
    """Equilibrates a molecular system using Gromacs. Parameters are defined in Data/Input/Simulation/npt.mdp.
    This operation assembles the .gro and .top file of a system together with the minimization instructions defined in the .mdp.
    and does the equilibration procedure. As a result the following files are created:

    {system}_npt.tpr
        result of gmx grompp. Assembly of .top, .gro and .mdp. Used as input to gmx mdrun. option -o
    {system}_npt.mdp 
        result of gmx grommp. Contains all parameters that were used during minimization.
    {system}_npt.gro
        result of gmx mdrun. Resulting coordinates after equilibration. option -c
    {system}_npt.edt
        result of gmx mdrun. Contains energies, the temperature, pressure, etc. option -e
    {system}_npt.log
        result of gmx mdrun. Contains a log of each step during minimization. option -g
    {system}_npt.trr
        result of gmx mdrun. Contains the entire trajectory of minimization. option -o
    
    PARAMETERS:
    -----------
    system: str
        The complete path to the system.
    """

    #Separate input path from system
    input_path, system = os.path.split(system)
    input_path += "/"

    #Output is saved in a different folder
    output_path = "Data/Output/Simulation/"

    #Assemble .gro and .top with parameters in .mdp for  minimization
    os.system(
        f"gmx grompp "                          #File assembly command
        f"-f Data/Input/Simulation/npt.mdp "    #Input parameters
        f"-c {output_path+system}_minim.gro "   #Input coordinate file
        f"-p {input_path+system}.top "          #Input toology
        f"-o {input_path+system}_npt.tpr "      #Result of assembly
        f"-po {output_path+system}_npt"         #All parameters used
    )

    #Run the assembly 
    print(f"Equilibrating system {system}...")
    os.system(
        f"gmx mdrun "                      #Run simulation command
        f"-s {input_path+system}_npt "     #Input assembly to simulation
        f"-c {output_path+system}_npt "    #Coordinates after simulation
        f"-e {output_path+system}_npt "    #Energy file
        f"-g {output_path+system}_npt "    #Log of simulation
        f"-o {output_path+system}_npt "    #Full precision trajectory
        f"-cpo {output_path+system}_npt "  #Checkpoint files
        f"-x {output_path+system}_npt"     #Compressed trajectory
    )


def run_simulation(system):
    """Runs a molecular simulation of a system using Gromacs. Parameters are defined in Data/Input/Simulation/inputParametersTest.mdp.
    This operation assembles the .gro and .top file of a system together with the minimization instructions defined in the .mdp.
    and does the equilibration procedure. As a result the following files are created:

    {system}_md.tpr
        result of gmx grompp. Assembly of .top, .gro and .mdp. Used as input to gmx mdrun. option -o
    {system}_md.mdp 
        result of gmx grommp. Contains all parameters that were used during minimization.
    {system}_md.gro
        result of gmx mdrun. Resulting coordinates after minimization. option -c
    {system}_md.edt
        result of gmx mdrun. Contains energies, the temperature, pressure, etc. option -e
    {system}_md.log
        result of gmx mdrun. Contains a log of each step during minimization. option -g
    {system}_md.trr
        result of gmx mdrun. Contains the entire trajectory of minimization. option -o
    
    PARAMETERS:
    -----------
    system: str
        The complete path to the system.
    """

    #Separate input path from system
    input_path, system = os.path.split(system)
    input_path += "/"

    #Output is saved in a different folder
    output_path = "Data/Output/Simulation/"

    #Assemble .gro and .top with parameters in .mdp for simulation, include .cpt
    os.system(
        f"gmx grompp "                          #File assembly command
        f"-f Data/Input/Simulation/md.mdp "     #Input parameters
        f"-c {output_path+system}_npt.gro "     #Input coordinate file
        f"-p {input_path+system}.top "          #Input toology
        f"-o {input_path+system}_md.tpr "       #Result of assembly
        f"-po {output_path+system}_md "          #All parameters used
        f"-t {output_path+system}_npt.cpt"      #Checkpoint from npt equilibration
    )

    #Run the assembly 
    print(f"Equilibrating system {system}...")
    os.system(
        f"gmx mdrun "                     #Run simulation command
        f"-s {input_path+system}_md "     #Input assembly to simulation
        f"-c {output_path+system}_md "    #Coordinates after simulation
        f"-e {output_path+system}_md "    #Energy file
        f"-g {output_path+system}_md "    #Log of simulation
        f"-o {output_path+system}_md "    #Full precision trajectory
        f"-cpo {output_path+system}_md "  #Checkpoint files
        f"-x {output_path+system}_md"     #Compressed trajectory
    )


#PARAMETERS
# rx, ry, rz = 3, 1, 3
# morph_type = "alpha"
# box_size = 5.0
# distance = 1.8

#CHOOSE SYSTEM
# system = f"Data/Output/System/Crystal/{morph_type}_glycine_crystal_{rx}_{ry}_{rz}_box_{box_size}_solv"

#CHOOSE SIMULATION TYPE
# minimize_system(system)
# equilibrate_system(system)
# run_simulation(system)



