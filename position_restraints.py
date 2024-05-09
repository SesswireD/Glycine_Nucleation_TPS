import numpy as np
import os

def read_gro_file(system):
    """
    Read a .gro file and collect coordinates, atom names, and box size.

    PARAMETERS:
    -----------
        file_path : str 
            Path to the .gro file.

    RETURNS:
    --------
        atom_coordinates : list
            List of arrays containing atom coordinates.
        atom_names : list
            List of atom names.
        box_size : np.array
            Array containing box size.
    """

    atom_coordinates = []
    atom_names = []

    #Add .gro extension
    file_path = system + ".gro"

    #Open the .gro file
    with open(file_path, 'r') as file:
        
        #Ignore the title
        _ = file.readline()

        #Read number of atoms 
        num_atoms = int(file.readline())

        #Read lines containing atom information
        for i in range(num_atoms):
        
            #Extract atom information
            atom_info = file.readline().split()

            #Extract atom coordinates and convert to float
            coord = np.array(atom_info[3:6], dtype=float)
            atom_coordinates.append(coord)

            #Extract atom name
            atom_name = atom_info[1]
            atom_names.append(atom_name)

        #Read the last line containing box size
        box_size = np.array(file.readline().split(), dtype=float)

    return atom_coordinates, atom_names, box_size, num_atoms


def atoms_within_range(atom_coordinates, point, range_threshold):
    """
    Identify atoms within range of a certain point.

    PARAMETERS:
    -----------
        atom_coordinates : list 
            List of coordinate arrays.
        point : np.array
            Coordinate of center point.
        range_threshold : float
            Distance from point to include atoms.
            
    RETURNS:
    --------
        atom_within_range : np.array
            Array where 1 indicates the atom is within range, 0 not.
    """

    #Empty list for molecule indices within range
    atoms_within_range = np.zeros(len(atom_coordinates))

    #Loop through the coords 
    for i, atom_coord in enumerate(atom_coordinates):
        distance = np.linalg.norm(atom_coord - point)
        if distance <= range_threshold:
            atoms_within_range[i] = 1

    return atoms_within_range


def get_posre_indices(atoms_within_range, atom_names):
    """
    Identifies the molecules that are within range.
    Returns the indices of all the atoms of these molecules.

    PARAMETERS:
    -----------
        atom_within_range : np.array
            Array where 1 indicates the atom is within range, 0 not.
            
    RETURNS:
    --------
        posre_indices : list
            A list of indices of all the atoms that need to be position restrained.
    """
    posre_indices = []
    
    #Loop through atoms array
    for i in range(0, len(atom_names), 10):   #The molecule consists of 10 atoms
        #If the central carbon atom is within range, add the entire molecule
        if atoms_within_range[i:i+10][1] == 1:  #CA is second element of the slice
            posre_indices.extend(range(i, i+10))

    return posre_indices


def write_posre_file(filename, posre_indices, point, distance):
    """
    Writes a position restrain file (.itp) given a list of indices.

    PARAMETERS:
    -----------
        posre_indices : list
            A list of indices of all the atoms that need to be position restrained.
    """
    #Separate input path from system
    input_path, system = os.path.split(filename)
    input_path += "/"

    #Open the file
    with open(filename + ".itp", 'w') as file:
        #Write itp information
        file.write(f'; This file restrains the molecules that are within {distance} nm of point {point}\n')
        file.write(f'; for the glycine cluster in system {system}.\n\n')

        #Write position restraint block
        file.write('[ position_restraints ]\n')

        #Write column header
        file.write('; atom  type      fx      fy      fz\n')

        #write all the indices with force constants
        for i, index in enumerate(posre_indices, start=1):
            file.write(f'{index:>6}     1  1000  1000  1000\n')


# #DEFINE SYSTEM:
# system = "Data/Output/Simulation/alpha_glycine_cluster_square_2_1_2_box_5.0_minim"
# point  = [2.5, 2.5, 2.5]
# distance = 1

# atom_coordinates, atom_names, box_size, num_atoms = read_gro_file(system)

# atom_range_bools = atoms_within_range(atom_coordinates, point, distance)

# posre_indices = get_posre_indices(atom_range_bools,atom_names)

# write_posre_file(system, posre_indices, point, distance)

