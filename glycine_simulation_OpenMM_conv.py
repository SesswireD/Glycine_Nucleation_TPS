#OpenMM imports
from openmm import *
from openmm.app import *
from openmm.unit import *

#NumPy
import numpy as np

#Simulation output
from sys import stdout

#Parmed imports
import parmed as pmd
from parmed.openmm.reporters import NetCDFReporter
from parmed import unit as u
from parmed import gromacs

def list_parameters(system):

    #List all force objects and their parameters
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        print(f"Force {force_index}: {force.__class__.__name__}")
        for parameter_name in dir(force):
            # if not parameter_name.startswith("_"):  # Exclude private attributes
            parameter_value = getattr(force, parameter_name)
            print(f"  {parameter_name}: {parameter_value}")


def build_simulation(gro_file, top_file, out_file, temp, nsteps, pres= 1, pcoupl= 500):
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
    gro = GromacsGroFile(f"{gro_file}.gro")
    top = GromacsTopFile(f"{top_file}.top", periodicBoxVectors=gro.getPeriodicBoxVectors(), includeDir='/home/danielb/OPENMM/top')

    #Create a system out of the topology by specifying bonds and interactions
    system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*u.nanometer, constraints=HBonds, ewaldErrorTolerance=1e-05*u.nanometer)

    #Specify integrator to use for advancing the system
    integrator = LangevinMiddleIntegrator(temp*u.kelvin, 1.0/u.picoseconds, 0.002*u.picoseconds)

    #Specify Barostat for pressure coupling and add to system
    barostat = MonteCarloBarostat(pres, temp, pcoupl)
    system.addForce(barostat)

    #Simulation compute options (GPU)
    platform = Platform.getPlatformByName('CUDA')
    prop = dict(CudaPrecision='mixed') # Use mixed single/double precision
    
    #Simulation compute options (CPU)
    # platform = Platform.getPlatformByName('CPU')
    # prop = dict(OpenMPThreads=16)

    #Combine topology, system and integrator to initialize the system
    simulation = Simulation(top.topology, system, integrator, platform=platform)
    simulation.context.setPositions(gro.positions)

    #Minimize the system for stability
    print("minimizing energy...")
    simulation.minimizeEnergy(maxIterations=1000)

    #Define output measures and save every picosecond
    simulation.reporters.append(StateDataReporter(f'{out_file}.csv', 500, step=True, potentialEnergy=True, temperature=True, volume = True,density= True, progress=True, remainingTime=True, totalSteps=nsteps)) #system state variable
    # simulation.reporters.append(DCDReporter(f'{out_file}.dcd', 500))    #compressed trajectory file
    simulation.reporters.append(XTCReporter(f'{out_file}.xtc', 500))    #compressed trajectory file
    simulation.reporters.append(NetCDFReporter(f'{out_file}.nc', 500, vels=True)) #NetCDF file for path sampling

    #Run the simulation for a number of steps
    print("starting simulation...")
    simulation.step(nsteps)

    #Save final coordinates to a GRO file using Parmed
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    positions = state.getPositions()
    box = state.getPeriodicBoxVectors()

    #Load topology and system into ParmEd
    parm = pmd.openmm.load_topology(top.topology, system)
    parm.positions = positions
    parm.box = [box[0][0], box[1][1] , box[2][2], 90.0, 90.0, 90.0]

    #Write the GRO file using ParmEd
    parm.save(f"{out_file}.gro", format='gro', overwrite=True)


temp = 300      # temperature in K
nsteps = 500000 # nsteps * 0.002ps = 1000ps = 1ns
system = "gamma_glycine_crystal_3_2_3_box_5.0_1.8nm_sphere_insert_solv" # system name
gro_file = f"{system}_minim"          # path to gro file
top_file = f"{system}"          # path to top file
out_file = f"{system}_minim_npteq{temp}K" # output path

build_simulation(gro_file, top_file, out_file, temp, nsteps)