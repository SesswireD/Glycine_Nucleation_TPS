from __future__ import division, print_function

import sys

# OpenMM Imports
import openmm as mm
import openmm.app as app
from openmmtools import integrators


from parmed import unit as u
import mdtraj as md

# ParmEd Imports
from parmed import load_file
from parmed.openmm.reporters import NetCDFReporter
from parmed import unit as u
from parmed import gromacs

from simtk.unit import *
from sys import stdout
import warnings
warnings.filterwarnings("ignore")


#Notes:
#VVVR integrator or langevin 
#montecarlobarostat
#report everything I can report (temp,pressure)
#N,P,T system  300K 1 atm 
#check first with a short run

# Load the Gromacs files
print('Loading Gromacs files...')
gro = load_file('/home/magdak/OpenMM/gromacs_new/posre.gro')
top = load_file('/home/magdak/OpenMM/gromacs_new/topol.top')

top.box = gro.box[:]
print(top.box)

# Simulation Options and Prepare the Simulation
platform = mm.Platform.getPlatformByName('CUDA')
prop = dict(CudaPrecision='mixed') # Use mixed single/double precision

topology = top.topology

tinit = 0
dt = 0.002*u.picoseconds
# nsteps = 5000000 #we have 10 nanoseconds sim

nsteps = 10000000 #we have 20 nanoseconds sim

# nsteps = 100
# nsteps = 100000 #job id 119090
temperature = 298*u.kelvin
pressure = 1*u.atmosphere
collision_rate = 1.0 / u.picoseconds
frequency = 500 #every 1 ps

print(gromacs.GROMACS_TOPDIR)

forcefield = app.ForceField('amber99sbildn.xml', 'tip3p.xml')

system = forcefield.createSystem(
                        topology,
                        nonbondedMethod=app.PME,
                        nonbondedCutoff=1.1*u.nanometer,
                        ewaldErrorTolerance=1e-05*u.nanometer,
                        constraints=app.HBonds)
                        
                        # implicitSolvent=app.GBn2,
                        # implicitSolventSaltConc=0.2*u.moles/u.liter


barostat = mm.MonteCarloBarostat(pressure,temperature,frequency)
system.addForce(barostat)


for i in range(3):

    # integrator = mm.VerletIntegrator(dt)
    integrator = integrators.VVVRIntegrator(temperature,collision_rate,dt)
    

    simulation = app.Simulation(topology, system, integrator,platform=platform, platformProperties=prop)
    simulation.context.setPositions(gro.positions)

    # Minimize energy
    simulation.minimizeEnergy()

    #step = 2500 so 0.5 ps

    simulation.step(1)

     # Save the first frame as a PDB file
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()
    with open(f'first_frame_{i}.pdb', 'w') as pdb_file:
        app.PDBFile.writeFile(topology, positions, pdb_file)

    #set reporters to save info every 100 steps.
    simulation.reporters.append(app.StateDataReporter(f'sim_{i}.csv', 5000, step=True, potentialEnergy=True, temperature=True, volume = True,density= True, progress=True, remainingTime=True, totalSteps=nsteps))
    simulation.reporters.append(app.DCDReporter(f'sim_{i}.dcd', 5000)) #step of 10 picoseconds
    
    # simulation.reporters.append(md.reporters.NetCDFReporter(f'sim_{i + 1}.nc', 5000))
    simulation.reporters.append(NetCDFReporter(f'sim_{i + 1}.nc', 5000, vels=True))

    # simulation.reporters.append(app.StateDataReporter(f'log_{i}.log', 2500, step=True, time=True, potentialEnergy=True,
    #                                                   kineticEnergy=True, totalEnergy=True, temperature=True, volume=True,
    #                                                   density=True, progress=True, remainingTime=True, speed=True, totalSteps=nsteps,
    #                                                   separator='\t'))  # Log file with detailed performance info
    
    
    simulation.reporters.append(app.PDBReporter(f'sim_{i}.pdb', 5000))  # Specify your file name and frequency
    

    # Run simulation
    simulation.step(nsteps)

