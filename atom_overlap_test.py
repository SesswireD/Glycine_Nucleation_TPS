import numpy as np
import os
from position_restraints import read_gro_file
from numba import njit

# @njit #does not work because of the set in find_duplicate_indices
def find_duplicate_indices(coordinates_array, tolerance=0):
    """
    Finds the indices of duplicate coordinates in the given list of coordinates.
    
    Parameters:
    -----------
    coordinates : list of np.array
        List of atomic coordinates.
    tolerance : float
        Tolerance for considering two coordinates as equal due to floating-point precision.
    
    Returns:
    --------
    duplicates : dict
        Dictionary where the key is the index of the original coordinate and the value is a list of indices of duplicate coordinates.
    """
    # coordinates_array = np.array(coordinates)
    num_coords = len(coordinates_array)
    duplicates = {}

    # Compare each coordinate with every other coordinate
    for i in range(num_coords):
        print(f"At coordinate:{i} of {num_coords}")
        if i in duplicates:
            continue
        duplicates[i] = []
        for j in range(i + 1, num_coords):
            if np.allclose(coordinates_array[i], coordinates_array[j], atol=tolerance):
                duplicates[i].append(j)
    
    # Remove entries that have no duplicates
    duplicates = {k: v for k, v in duplicates.items() if v}
    
    return duplicates


#Specify path
filepath = "Data/Output/System/Crystal/gamma_glycine_crystal_3_2_3_box_5.0"

#Read gro file
atom_coordinates, atom_names, box_size, num_atoms = read_gro_file(filepath)

#Convert to array and find duplicates
coordinates_array = np.array(atom_coordinates)
duplicates = find_duplicate_indices(coordinates_array)

#Print duplicates
print(duplicates)
