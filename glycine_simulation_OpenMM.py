#OpenMM imports
from openmm import *
from openmm.app import *
from openmm.unit import *

#NumPy
import numpy as np

#Simulation output
from sys import stdout

def list_parameters(system):

    #List all force objects and their parameters
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        print(f"Force {force_index}: {force.__class__.__name__}")
        for parameter_name in dir(force):
            # if not parameter_name.startswith("_"):  # Exclude private attributes
            parameter_value = getattr(force, parameter_name)
            print(f"  {parameter_name}: {parameter_value}")


def build_simulation(gro_file, top_file, temp = 300):
    """Loads coordinate and topology files and initializes a molecular dynamics simulation.

    PARAMETERS:
    -----------
        gro_file: str
            path to .gro file to load
        top_file: str
            path to .top file to load

    RETURNS:
    --------
        simulation: obj
            a simulation object
    """
    #Load Gro and Top files
    gro = GromacsGroFile(gro_file)
    top = GromacsTopFile(top_file, periodicBoxVectors=gro.getPeriodicBoxVectors(), includeDir='/usr/local/gromacs/share/gromacs/top')

    #Create a system out of the topology by specifying bonds and interactions
    system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)

    #Specify integrator to use for advancing the system
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)

    #Combine topology, system adn integrator to initialize the system
    simulation = Simulation(top.topology, system, integrator)
    simulation.context.setPositions(gro.positions)

    #Do local energy minimization for stability
    print("minimizing energy...")
    simulation.minimizeEnergy()

    #Define output measures
    simulation.reporters.append(PDBReporter('New_File_Test.pdb', 1000))
    simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))

    #Run the simulation for a number of steps
    print("starting simulation...")
    simulation.step(10000)


build_simulation("Data/Output/gly_box_120_3.6_solv.gro", "Data/Output/gly_box_120_3.6_solv.top")