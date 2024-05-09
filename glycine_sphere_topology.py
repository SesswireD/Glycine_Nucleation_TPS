import numpy as np
import os

#Import own conversion functions
from glycine_monomer_topology import convert_topology

#Import own crystal functions
from glycine_crystal_topology import count_molecules

#Import position restrain functions
from position_restraints import read_gro_file, get_posre_indices, atoms_within_range

def write_gro_file(filename, atom_coordinates, atom_names, distance, box_size):
    """
    Writes the coordinates of the atoms to a gro file. Works slightly different from the 
    same named function defined in 'glycine_crystal_topology.py', because of the input datastructure.

    PARAMETERS:
    -----------
        filename: str
            The name of the output gro file.
        cell_grid: numpy.ndarray
            An array containing tuples where each tuple contains two lists: one for coordinates and one for atom names.
        lattice: numpy.ndarray
            An array containing the lattice definition.
    """

    #initialize index count
    index = 1

    #Open the file
    with open(filename, 'w') as file:
        #Calculate number of atoms
        num_elements = len(atom_coordinates)

        #Write title of the title of the file
        file.write(f"Spherical Glycine cluster of radius {distance} nm\n")

        #Write the number of atoms
        file.write('   ' + str(num_elements) + '\n')

        #Loop through the cell_grid and write the coordinates and atom names to file
        for coord, atom in zip(atom_coordinates, atom_names):
                    
            #Use f string to get correct spacing between entries
            line = f"    1GLY{atom:>7}{index:>5}{float(coord[0]):>8.3f}{float(coord[1]):>8.3f}{float(coord[2]):>8.3f}\n"
            file.write(line)
            #Increment index
            index+=1

        #Write box size to file
        file.write(f"   {box_size[0]:<9.5f}{box_size[1]:<9.5f}{box_size[2]:<9.5f}")


def remove_molecules(posre_indices, atom_coordinates, atom_names):
    """Removes atom coordinates based on a list of indices.

    PARAMETERS:
    -----------
        posre_indices: list
            List of atom indices to keep.
        atom_coordinates: list
            List of np.arrays containing the xyz coordinates of the atoms.
        atom_names: list
            List containing the atom_names .
    
    RETURNS:
    --------
        atom_coordinates: list
            Shortened list of np.arrays containing the xyz coordinates of the atoms.
        atom_names: list
            Shortened list containing the atom_names. 
    """

    # Create a mask to keep track of which indices to keep
    mask = [False] * len(atom_coordinates)

    # Set the mask for indices in posre_indices to True
    for idx in posre_indices:
        mask[idx] = True

    # Filter out coordinates and atom names based on the mask
    atom_coordinates = [coord for i, coord in enumerate(atom_coordinates) if mask[i]]
    atom_names = [name for i, name in enumerate(atom_names) if mask[i]]

    return atom_coordinates, atom_names


def build_spherical_cluster(system, distance, point = None):
    """Builds a molecular system of a spherical glycine cluster of certain diameter,
    based on a crystal system and saves it to a .gro file. 

    PARAMETERS:
    -----------
        system: str
            Path to the crystal .gro system to use as basis.
        distance: float
            Radius (in nanometer) within molecules are kept.
        point: np.array
            Coordinate from which to draw the radius.
    """
    #Path to crystal folder
    input_path = "Data/Output/System/Crystal/"

    #Path to sphere folder
    output_path = "Data/Output/System/Sphere/"

    #Read the square crystal system
    atom_coordinates, atom_names, box_size, num_atoms = read_gro_file(input_path+system)

    #By default the point from which to include molecules is the center of the box
    if point == None:
        point = box_size/2

    #Get the atoms that are within the range of the point
    atom_range_bools = atoms_within_range(atom_coordinates, point, distance)

    #Get the indices of the molecules within range
    posre_indices = get_posre_indices(atom_range_bools, atom_names)

    #Remove the molecules that are not within range
    atom_coordinates, atom_names = remove_molecules(posre_indices, atom_coordinates, atom_names)

    #Write the coordinates to file
    write_gro_file(f"{output_path}{system}_{distance}nm_sphere.gro", atom_coordinates, atom_names, distance, box_size)


def build_spherical_system(system, distance, point = None):
    """Builds a molecular system of a spherical glycine cluster from a perfect crystal
    that is surrounded by free floating glycine molecules and water. Preserves the concentration
    of the initial crystal system.

    PARAMETERS:
    -----------
        system: str
            Path to the crystal .gro system to use as basis.
        distance: float
            Radius (in nanometer) within molecules are kept.
        point: np.array
            Coordinate from which to draw the radius.
    """
    #Path to crystal folder
    input_path = "Data/Output/System/Crystal/"

    #Path to sphere folder
    output_path = "Data/Output/System/Sphere/"

    #Build a spherical cluster by cutting a sphere out of a square circle
    build_spherical_cluster(system, distance, point)

    #Get number of glycine molecules in original crystal system
    num_gly_or, _ = count_molecules(f"{input_path}{system}.gro")

    #Get number of glycine molecules in spherical cluster system
    num_gly_sp, _ = count_molecules(f"{output_path}{system}_{distance}nm_sphere.gro")

    #Insert free floating glycine molecules to preserve concentration
    os.system(
        f"gmx insert-molecules "                                     #Gromacs insert command
        f"-f {output_path}{system}_{distance}nm_sphere.gro "         #System to insert to
        f"-ci Data/Input/System/glycine_match.pdb "                  #Molecule to insert
        f"-nmol {num_gly_or - num_gly_sp} "                          #Number of molecules to insert
        f"-o {output_path}{system}_{distance}nm_sphere_insert.gro"   #Output file
        )

    #Solvate the system
    os.system(
        f"gmx solvate "                                                  #Gromacs solvate command
        f"-cp {output_path}{system}_{distance}nm_sphere_insert.gro "     #System to solvate
        f"-cs spc216.gro "                                               #Water model to use
        f"-o {output_path}{system}_{distance}nm_sphere_insert_solv.gro") #Output file

    #Create topology file for the UNsolvated system
    convert_topology(f"{output_path}{system}_{distance}nm_sphere_insert.top", num_gly_or, 0)

    #Get the number of glycine and water molecules from the solvated system 
    num_gly, num_sol = count_molecules(f"{output_path}{system}_{distance}nm_sphere_insert_solv.gro")

    #Create topology file for the solvated system
    convert_topology(f"{output_path}{system}_{distance}nm_sphere_insert_solv.top",num_gly, num_sol)


#PARAMETERS:
# rx, ry, rz = 3, 1, 3
# morph_type = "alpha"
# box_size = 5.0
# distance = 0.4
# system = f"{morph_type}_glycine_crystal_{rx}_{ry}_{rz}_box_{box_size}"

#Build sphere system with different cluster radii
# for i in np.linspace(1.8, 0.4, 8 ):
#     build_spherical_system(system, round(i,1))