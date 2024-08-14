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

from simtk.openmm.app import StateDataReporter
from simtk.openmm import unit

class StateDataReporterWithPressure(StateDataReporter):
    def __init__(self, file, reportInterval, **kwargs):
        super().__init__(file, reportInterval, **kwargs)

    def report(self, simulation, state):
        # Get the pressure from the system
        state = simulation.context.getState(getEnergy=True, getPressure=True)
        pressure = state.getPressure()  # Pressure is in units of pascals

        # Convert pressure to bar for convenience (1 bar = 100,000 Pa)
        pressure_in_bar = pressure / (100 * unit.kilopascal)
        
        # Add pressure to the quantities being reported
        self._appendValue(pressure_in_bar, 'pressure')

        # Call the original report method to handle the rest
        super().report(simulation, state)

    def _constructHeaders(self):
        headers = super()._constructHeaders()
        headers.append('Pressure (bar)')
        return headers


def list_parameters(system):

    #List all force objects and their parameters
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        print(f"Force {force_index}: {force.__class__.__name__}")
        for parameter_name in dir(force):
            # if not parameter_name.startswith("_"):  # Exclude private attributes
            parameter_value = getattr(force, parameter_name)
            print(f"  {parameter_name}: {parameter_value}")


def get_atom_names_and_coordinates(simulation):
    """Extracts atom names and coordinates from an OpenMM simulation object.

    PARAMETERS:
    -----------
        simulation: Simulation
            OpenMM simulation object

    RETURNS:
    --------
        atom_names: list of str
            List of atom names
        coordinates: list of tuple
            List of coordinates (x, y, z) in nanometers
        box_vectors: list of float
            Periodic box vectors in nanometers
    """
    # Get the state of the system with positions
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    positions = state.getPositions()
    box = state.getPeriodicBoxVectors()
    
    # Extract atom names from the topology
    atom_names = []
    for residue in simulation.topology.residues():
        for atom in residue.atoms():
            atom_names.append(atom.name)
    
    # Extract coordinates
    coordinates = [(pos.x,
                    pos.y,
                    pos.z) for pos in positions]
    
    # Extract box vectors
    box_vectors = [box[0][0],
                   box[1][1],
                   box[2][2]]
    
    return atom_names, coordinates, box_vectors


def build_simulation(pdb_file, out_file, temp, nsteps, pres= 1, pcoupl= 500):
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
    #Load PDB file
    pdb = PDBFile(pdb_file)

    #Specify forcefield
    forcefield = ForceField('amber03.xml', 'spce.xml')

    #Create a system out of the topology by specifying bonds and interactions
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=1*u.nanometer, constraints=HBonds, ewaldErrorTolerance=1e-05*u.nanometer)

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
    # prop = dict(OpenMPThreads=4)

    #Combine topology, system and integrator to initialize the system
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    #Minimize the system for stability
    print("minimizing energy...")
    # simulation.minimizeEnergy(maxIterations=1000)

    #Define output measures and save every picosecond
    simulation.reporters.append(StateDataReporterWithPressure(f'{out_file}.csv', 500, step=True, potentialEnergy=True, temperature=True, volume = True,density= True, progress=True, remainingTime=True, totalSteps=nsteps)) #system state variable
    # simulation.reporters.append(DCDReporter(f'{out_file}.dcd', 500))    #trajectory file
    simulation.reporters.append(XTCReporter(f'{out_file}.xtc', 500))    #compressed trajectory file
    simulation.reporters.append(NetCDFReporter(f'{out_file}.nc', 500, vels=True)) #NetCDF file for path sampling

    #Run the simulation for a number of steps
    print("starting simulation...")
    simulation.step(nsteps)

    #Save final coordinates to PDB
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    positions = state.getPositions()

    with open('f{out_file}.pdb', 'w') as f:
        PDBFile.writeFile(simulation.topology, positions, f)


temp = 298     # temperature in K
nsteps = 3 # nsteps * 0.002ps = 10ps
system = "gly_box_195_5.0_solv"
pdb_file = "Data/Output/System/Sphere/alpha_glycine_crystal_3_1_3_box_5.0_1.6nm_sphere_insert_260_solv.pdb"
out_file = f"alpha_glycine_crystal_3_1_3_box_5.0_1.6nm_sphere_insert_260_solv_test"

build_simulation(pdb_file, out_file, temp, nsteps)

