import numpy as np
import ast
import os

#Import own conversion functions
from glycine_monomer_topology import convert_coordinates, convert_topology, calc_concentration

def read_xyz_file(filename):
    """Reads an XYZ file and extracts the coordinates of the atoms
    as well as the lattice definition and atom names of the unit cell

    PARAMETERS:
    -----------
        filename: str 
            The name of the XYZ file.

    RETURNS:
    --------
        coordinates: numpy.ndarray
            An array containing the coordinates of the atoms in the unit cell.
        lattice:  numpy.ndarray)
            An array containing the lattice definition.
        atoms: list
            List containing the atom names for each coordinate entry
    """
    with open(filename, 'r') as file:
        #Read the first line to get the number of atoms
        num_atoms = int(file.readline())

        #Empty list for storing atom names
        atoms = []
        
        #Read lattice definition line
        line = file.readline().split("\"")

        #Convert the list to a numpy array
        lattice = np.array(line[1].split(),dtype=float).reshape(3,3)

        #Read the coordinates of the atoms
        coordinates = np.empty((num_atoms, 3))
        for i in range(num_atoms):
            line = file.readline().split()
            coordinates[i] = list(map(float, line[:3]))
            atoms.append(line[-1])

    return coordinates, lattice, atoms


def calculate_lattice_length(lattice):
    """Computes the norm of the lattice definition i.e. the size of the unit cell

    PARAMETERS:
    -----------
        lattice: numpy.ndarray 
            lattice definition of the unit cell in 3x3 matrix

    RETURNS:
    --------
        unit_cell_size: numpy.ndarray
            An array containing the lengths of the unit cell given a lattice vector
    """

    #Empty array for storing the size
    unit_cell_size = np.empty(3)
    
    #Calculate norm (\length) for each entry
    for i in range(3):
        unit_cell_size[i] = np.linalg.norm(lattice[i])

    return unit_cell_size


def initialize_grid(rx, ry, rz):
    """Builds a grid datastructure of correct dimensions. Each element of the grid is a unit 
    cell that contains the coordinates of the atoms for the specific location on the grid.

     PARAMETERS:
    -----------
        coordinates: numpy.ndarray
            An array containing the coordinates of the atoms. Each row represents an atom and the columns
            contain the x, y, and z coordinates.
        r: int
            Number of unit cells to put in each direction from the origin

    RETURNS:
    --------
        cell_grid: numpy.ndarray
            An array containing empty unit cell coordinates
    """

    #Define the shape of the 3D grid and the size of each ndarray
    x_size  = (rx * 2 ) + 1
    y_size  = (ry * 2)  + 1
    z_size  = (rz * 2)  + 1
    grid_shape = (x_size, y_size, z_size)  

    #Create the 3D grid
    cell_grid = np.empty(grid_shape, dtype=object)

    #Initialize each element of the grid with an ndarray of the specified size
    for index in np.ndindex(grid_shape):
        cell_grid[index] = ([],[])
    
    return cell_grid


def create_mask(shape):
    """Creates a mask that determines which unit cells need to be built.

    PARAMETERS:
    -----------
        shape: tuple
            The shape of the mask grid in (x, y, z) format.

    RETURNS:
    --------
        mask: numpy.ndarray
            A mask grid where elements inside the shape are 1 and outside are 0.
    """

    #Initialize empty mask
    mask = np.zeros(shape)
    
    #Fill mask with ones
    mask.fill(1)

    return mask 


def copy_unit_cell(unit_coordinates, atoms, lattice, rx, ry, rz):
    """Given atom coordinates for a single unit cell, builds a rectangular cluster of unit cells with different radii along each direction from the initial cell.

     PARAMETERS:
    -----------
        coordinates: numpy.ndarray
            An array containing the coordinates of the atoms. Each row represents an atom and the columns
            contain the x, y, and z coordinates.
        lattice: numpy.ndarray
            An array containing the lattice definition.
        atoms: list
            List containing the atom names for each coordinate entry.
        rx: int
            Number of unit cells to put in the x-direction from the origin
        ry: int
            Number of unit cells to put in the y-direction from the origin
        rz: int
            Number of unit cells to put in the z-direction from the origin

    RETURNS:
    --------
        cell_grid: numpy.ndarray
            An array where each unit cell contains atom coordinates
    """

    #Initialize empty cell_grid where all coordinates will be stored
    cell_grid = initialize_grid(rx, ry, rz)

    #Get the unit cell dimensions from the lattice vector
    unit_cell_length = calculate_lattice_length(lattice)

    #Get index of half of the box
    half_i = (cell_grid.shape[0] - 1) / 2
    half_j = (cell_grid.shape[1] - 1) / 2
    half_k = (cell_grid.shape[2] - 1) / 2

    #Loop through the cell_grid and add coordinates
    for i in range(cell_grid.shape[0]):  # x
        for j in range(cell_grid.shape[1]):  # y
            for k in range(cell_grid.shape[2]):  # z

                #Calculate relative distance from origin in index
                dist_i = i - half_i
                dist_j = j - half_j
                dist_k = k - half_k

                #Copy the origin coordinates
                new_coordinates = unit_coordinates.copy()

                #Calculate coordinate change in each direction from the lattice
                change_x = lattice[0] * dist_i
                change_y = lattice[1] * dist_j
                change_z = lattice[2] * dist_k

                #Apply change to coordinates of origin cell
                new_coordinates += change_x + change_y + change_z
                
                #Add new coordinates to the grid with corresponding atom names
                cell_grid[i][j][k] = (new_coordinates.tolist(), atoms.copy())
                
    return cell_grid


def write_xyz_file(filename, cell_grid, lattice, mask):
    """
    Writes the coordinates of the atoms to an XYZ file.

    PARAMETERS:
    -----------
        filename: str
            The name of the output XYZ file.
        cell_grid: numpy.ndarray
            An array containing tuples where each tuple contains two lists: one for coordinates and one for atom names.
        lattice: numpy.ndarray
            An array containing the lattice definition.
    """
    #Open the file
    with open(filename, 'w') as file:
        #Calculate number of atoms
        num_elements = sum(len(atoms) for ((_,atoms), mask_value) in zip(np.ravel(cell_grid), np.ravel(mask)) if mask_value)

        #Write the number of atoms
        file.write(str(num_elements) + '\n')

        #Write the lattice definition
        file.write('Lattice="' + ' '.join(map(str, lattice.flatten())) + '" Properties=pos:R:3:species:S:1\n')

        #Loop through the cell_grid and write the coordinates and atom names to file
        for (coordinates, atoms), mask_value in zip(np.ravel(cell_grid),np.ravel(mask)):
            if mask_value == 1:
                for atom, coord in zip(atoms, coordinates):
                    # file.write(' ' + atom + ' '.join(map(str, coord)) + '\n')   #Ovido expects xyz like this
                    file.write(f"{atom:>3} {coord[0]:10.5f} {coord[1]:10.5f} {coord[2]:10.5f}\n") #OpenBabel expects xyz like this


def write_gro_file(filename, cell_grid, lattice, mask, morph_type):
    """
    Writes the coordinates of the atoms to a gro file.

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
    mol_index = 1

    #Open the file
    with open(filename, 'w') as file:
        #Calculate number of atoms
        num_elements = sum(len(atoms) for ((_,atoms), mask_value) in zip(np.ravel(cell_grid), np.ravel(mask)) if mask_value)

        #Write title of the title of the file
        file.write(f"{morph_type} glycine crystal\n")

        #Write the number of atoms
        file.write('   ' + str(num_elements) + '\n')

        #Loop through the unit cells in cell_grid and write the coordinates and atom names to file
        for (coordinates, atoms), mask_value in zip(np.ravel(cell_grid),np.ravel(mask)):
            #Only write if mask is active for this cell
            if mask_value == 1:
                #Loop through all the atoms and coord in this unit cell
                for atom, coord in zip(atoms, coordinates):
                    #Use f string to get correct spacing between entries
                    line = f"    {mol_index}GLY{atom:>7}{index:>5}{float(coord[0])*0.1:>8.3f}{float(coord[1]*0.1):>8.3f}{float(coord[2]*0.1):>8.3f}\n"
                    file.write(line)
                    #Increment index
                    index+=1
                    # print(index)
                    # print(index%10)
                    if (index % 11) == 0:
                        mol_index += 1
                        # print("YESSS")
                        # print(mol_index)

        #Calculate lattice definition
        box_size = calculate_lattice_length(lattice) * 0.1
        #Write box size to file
        file.write(f"   {box_size[0]:<9.5f}{box_size[1]:<9.5f}{box_size[2]:<9.5f}")


def collect_gamma_molecules(cell_grid, mask):
    """
    Takes the random ordering of the coordinates in the gamma glycine unit cell and groups atoms by molecules.
    A secondary result of this operation is that stray (incomplete) molecules are removed.

    PARAMETERS:
    -----------
        cell_grid: numpy.ndarray
            Each element is a unit cell that contains a (coordinates,atoms) list tuple.
        mask: numpy.ndarray
            A mask where each element contains an integer (0,1) denoting wether the unit cell is present.
    
    RETURNS:
    --------
        new_grid: numpy.ndarray
            Similar to cell_grid, but with a different ordering of atoms.
    """

    #Get size of the cell
    rx, ry, rz = cell_grid.shape

    #get new grid
    new_grid = cell_grid.copy()
    
    #Add padding to the mask 
    padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
   
    #loop through the grid
    for i in range(rx):
        for j in range(ry):
            for k in range(rz):
                #check if current cell exists
                if mask[i][j][k] == 1:

                    #lists for  new ordering
                    cell_coordinates = []
                    cell_atoms = []

                    #Check if the almost complete molecule can be completed (if the z+ neighbor cells are present)
                    if padded_mask[i+1][j+1][k+2]:
                        
                        #Add coordinates such that they correspond to Gly molecule
                        cell_coordinates.append(cell_grid[i][j][k][0][2]) # N from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][14]) # CA from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][11]) # C from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][5])  # O from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][8])  # O1 from own cell
                        cell_coordinates.append(cell_grid[i][j][k+1][0][17]) # H1 from z+ neighbor
                        cell_coordinates.append(cell_grid[i][j][k][0][20]) # H2 from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][26]) # HA1 from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][29]) # HA2 from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][23]) # H3 from own cell

                        #Add atom names such that they correspond to Gly molecule
                        cell_atoms.append(cell_grid[i][j][k][1][2]) # N from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][14]) # CA from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][11]) # C from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][5])  # O from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][8])  # O1 from own cell
                        cell_atoms.append(cell_grid[i][j][k+1][1][17]) # H1 from z+ neighbor
                        cell_atoms.append(cell_grid[i][j][k][1][20]) # H2 from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][26]) # HA1 from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][29]) # HA2 from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][23]) # H3 from own cell

                    # Check if the first carbon group can be completed (if the z- and y+ neighbor cells are present)
                    if padded_mask[i+1][j+1][k] and padded_mask[i+1][j+2][k+1]:

                        #Add coordinates such that they correspond to Gly molecule
                        cell_coordinates.append(cell_grid[i][j+1][k][0][0]) # N from y+ neighbor
                        cell_coordinates.append(cell_grid[i][j][k][0][12]) # CA from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][9])  # C from own cell 
                        cell_coordinates.append(cell_grid[i][j+1][k][0][3])  # O from y+ neighbor
                        cell_coordinates.append(cell_grid[i][j][k-1][0][6]) # O1 from z-neighbor
                        cell_coordinates.append(cell_grid[i][j+1][k][0][15])    # H1 from y+ neighbor
                        cell_coordinates.append(cell_grid[i][j+1][k][0][18])    # H2 from from y+ neighbor
                        cell_coordinates.append(cell_grid[i][j+1][k][0][24])  # HA1 from y+ neighbor 
                        cell_coordinates.append(cell_grid[i][j][k][0][27])   # HA2 from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][21])   # H3 from own cell
                        
                        #Add atom names such that they correspond to Gly molecule
                        cell_atoms.append(cell_grid[i][j+1][k][1][0]) # N from y+ neighbor
                        cell_atoms.append(cell_grid[i][j][k][1][12]) # CA from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][9])  # C from own cell 
                        cell_atoms.append(cell_grid[i][j][k][1][3])  # O from own cell
                        cell_atoms.append(cell_grid[i][j][k-1][1][6]) # O1 from z-neighbor
                        cell_atoms.append(cell_grid[i][j+1][k][1][15])    # H1 from y+ neighbor
                        cell_atoms.append(cell_grid[i][j+1][k][1][18])    # H2 from from y+ neighbor
                        cell_atoms.append(cell_grid[i][j+1][k][1][24])  # HA1 from y+ neighbor 
                        cell_atoms.append(cell_grid[i][j][k][1][27])   # HA2 from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][21])   # H3 from own cell

                    # Check if the first nitrogen group can be completed (if the y- and (y-,z-) neighbor cells are present)
                    if padded_mask[i+1][j][k+1] and padded_mask[i+1][j][k]:

                        #Add coordinates such that they correspond to Gly molecule
                        cell_coordinates.append(cell_grid[i][j][k][0][0]) # N from own cell
                        cell_coordinates.append(cell_grid[i][j-1][k][0][12]) # CA from y- neighbor
                        cell_coordinates.append(cell_grid[i][j-1][k][0][9])  # C from y- neighbor 
                        cell_coordinates.append(cell_grid[i][j][k][0][3])  # O from own cell
                        cell_coordinates.append(cell_grid[i][j-1][k-1][0][6]) # O1 from (y-,z-) neighbor
                        cell_coordinates.append(cell_grid[i][j][k][0][15])    # H1 from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][18])    # H2 from from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][24])  # HA1 from own cell 
                        cell_coordinates.append(cell_grid[i][j-1][k][0][27])   # HA2 from y- neigbor
                        cell_coordinates.append(cell_grid[i][j-1][k][0][21])   # H3 from y- neighbor
                        
                        #Add atom names such that they correspond to Gly molecule
                        cell_atoms.append(cell_grid[i][j][k][1][0]) # N from own cell
                        cell_atoms.append(cell_grid[i][j-1][k][1][12]) # CA from y- neighbor
                        cell_atoms.append(cell_grid[i][j-1][k][1][9])  # C from y- neighbor 
                        cell_atoms.append(cell_grid[i][j][k][1][3])  # O from own cell
                        cell_atoms.append(cell_grid[i][j-1][k-1][1][6]) # O1 from (y-,z-) neighbor
                        cell_atoms.append(cell_grid[i][j][k][1][15])    # H1 from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][18])    # H2 from from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][24])  # HA1 from own cell 
                        cell_atoms.append(cell_grid[i][j-1][k][1][27])   # HA2 from y- neigbor
                        cell_atoms.append(cell_grid[i][j-1][k][1][21])   # H3 from y- neighbor

                    # Check if the second nitrogen group can be completed (if the z-, x+ and (z-,x+) neighbor cells are present)
                    if padded_mask[i+1][j+1][k] and padded_mask[i+2][j+1][k+1] and padded_mask[i+2][j+1][k]:

                        #Add coordinates such that they correspond to Gly molecule
                        cell_coordinates.append(cell_grid[i][j][k][0][1])  # N from own cell
                        cell_coordinates.append(cell_grid[i+1][j][k][0][13])  # CA from x+ neighbor
                        cell_coordinates.append(cell_grid[i+1][j][k-1][0][10])  # C from (z-, x+) neighbor
                        cell_coordinates.append(cell_grid[i][j][k-1][0][4])  # O from z-neighbor
                        cell_coordinates.append(cell_grid[i+1][j][k-1][0][7])  # O1 from (z-,x+) neighbor
                        cell_coordinates.append(cell_grid[i][j][k][0][16])  # H1 from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][19])  # H2 from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][25])  # HA1 from own cell
                        cell_coordinates.append(cell_grid[i+1][j][k][0][28])  # HA2 from x+ neighbor
                        cell_coordinates.append(cell_grid[i+1][j][k][0][22])  # H3 from x+ neighbor

                        #Add coordinates such that they correspond to Gly molecule
                        cell_atoms.append(cell_grid[i][j][k][1][1])  # N from own cell
                        cell_atoms.append(cell_grid[i+1][j][k][1][13])  # CA from x+ neighbor
                        cell_atoms.append(cell_grid[i+1][j][k-1][1][10])  # C from (z-, x+) neighbor
                        cell_atoms.append(cell_grid[i][j][k-1][1][4])  # O from z-neighbor
                        cell_atoms.append(cell_grid[i+1][j][k-1][1][7])  # O1 from (z-,x+) neighbor
                        cell_atoms.append(cell_grid[i][j][k][1][16])  # H1 from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][19])  # H2 from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][25])  # HA1 from own cell
                        cell_atoms.append(cell_grid[i+1][j][k][1][28])  # HA2 from x+ neighbor
                        cell_atoms.append(cell_grid[i+1][j][k][1][22])  # H3 from x+ neighbor


                    #Check if the second carbon group can be completed (if the z-, x- and (z-,x-) neighbor cells are present)
                    if padded_mask[i+1][j+1][k] and padded_mask[i][j+1][k+1] and padded_mask[i][j+1][k]:

                        # Add coordinates such that they correspond to Gly molecule
                        cell_coordinates.append(cell_grid[i-1][j][k][0][1])  # N from x- neighbor
                        cell_coordinates.append(cell_grid[i][j][k][0][13])  # CA from own cell
                        cell_coordinates.append(cell_grid[i][j][k-1][0][10])  # C from z- neighbor
                        cell_coordinates.append(cell_grid[i-1][j][k-1][0][4])  # O from (x-,z-) neighbor
                        cell_coordinates.append(cell_grid[i][j][k-1][0][7])  # O1 from z- neighbor
                        cell_coordinates.append(cell_grid[i-1][j][k][0][16])  # H1 from x- neighbor
                        cell_coordinates.append(cell_grid[i-1][j][k][0][19])  # H2 from x- neighbor
                        cell_coordinates.append(cell_grid[i-1][j][k][0][25])  # HA1 from x- neighbor
                        cell_coordinates.append(cell_grid[i][j][k][0][28])  # HA2 from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][22])  # H3 from own cell

                        # Add coordinates such that they correspond to Gly molecule
                        cell_atoms.append(cell_grid[i-1][j][k][1][1])  # N from x- neighbor
                        cell_atoms.append(cell_grid[i][j][k][1][13])  # CA from own cell
                        cell_atoms.append(cell_grid[i][j][k-1][1][10])  # C from z- neighbor
                        cell_atoms.append(cell_grid[i-1][j][k-1][1][4])  # O from (x-,z-) neighbor
                        cell_atoms.append(cell_grid[i][j][k-1][1][7])  # O1 from z- neighbor
                        cell_atoms.append(cell_grid[i-1][j][k][1][16])  # H1 from x- neighbor
                        cell_atoms.append(cell_grid[i-1][j][k][1][19])  # H2 from x- neighbor
                        cell_atoms.append(cell_grid[i-1][j][k][1][25])  # HA1 from x- neighbor
                        cell_atoms.append(cell_grid[i][j][k][1][28])  # HA2 from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][22])  # H3 from own cell


                    #Check if the carbon-oxygen group can be completed (if the z+, x- and (z+,x-) neighbor cells are present)
                    if padded_mask[i+1][j+1][k+2] and padded_mask[i][j+1][k+1] and padded_mask[i][j+1][k+2]:

                        # Add coordinates such that they correspond to Gly molecule
                        cell_coordinates.append(cell_grid[i-1][j][k+1][0][1])  # N from (x-,z+) neighbor
                        cell_coordinates.append(cell_grid[i][j][k+1][0][13])  # CA from z+ neighbor
                        cell_coordinates.append(cell_grid[i][j][k][0][10])  # C from own cell
                        cell_coordinates.append(cell_grid[i-1][j][k][0][4])  # O from x- neighbor
                        cell_coordinates.append(cell_grid[i][j][k][0][7])  # O1 from own cell
                        cell_coordinates.append(cell_grid[i-1][j][k+1][0][16])  # H1 from (x-,z+) neighbor
                        cell_coordinates.append(cell_grid[i-1][j][k+1][0][19])  # H2 from (x-,z+) neighbor
                        cell_coordinates.append(cell_grid[i-1][j][k+1][0][25])  # HA1 from (x-,z+) neighbor
                        cell_coordinates.append(cell_grid[i][j][k+1][0][28])  # HA2 from z+ neighbor
                        cell_coordinates.append(cell_grid[i][j][k+1][0][22])  # H3 from z+ neighbor

                        # Add coordinates such that they correspond to Gly molecule
                        cell_atoms.append(cell_grid[i-1][j][k+1][1][1])  # N from (x-,z+) neighbor
                        cell_atoms.append(cell_grid[i][j][k+1][1][13])  # CA from z+ neighbor
                        cell_atoms.append(cell_grid[i][j][k][1][10])  # C from own cell
                        cell_atoms.append(cell_grid[i-1][j][k][1][4])  # O from x- neighbor
                        cell_atoms.append(cell_grid[i][j][k][1][7])  # O1 from own cell
                        cell_atoms.append(cell_grid[i-1][j][k+1][1][16])  # H1 from (x-,z+) neighbor
                        cell_atoms.append(cell_grid[i-1][j][k+1][1][19])  # H2 from (x-,z+) neighbor
                        cell_atoms.append(cell_grid[i-1][j][k+1][1][25])  # HA1 from (x-,z+) neighbor
                        cell_atoms.append(cell_grid[i][j][k+1][1][28])  # HA2 from z+ neighbor
                        cell_atoms.append(cell_grid[i][j][k+1][1][22])  # H3 from z+ neighbor

                    # Check if the single oxygen can be completed (if the z+, x+ and (z+,x+) neighbor cells are present)
                    if padded_mask[i+1][j+1][k+2] and padded_mask[i+2][j+1][k+1] and padded_mask[i+2][j+1][k+2]:

                        # Add coordinates such that they correspond to Gly molecule
                        cell_coordinates.append(cell_grid[i][j][k+1][0][1])  # N from z+ neighbor
                        cell_coordinates.append(cell_grid[i+1][j][k+1][0][13])  # CA from (x+,z+) neighbor
                        cell_coordinates.append(cell_grid[i+1][j][k][0][10])  # C from x+ neighbor
                        cell_coordinates.append(cell_grid[i][j][k][0][4])  # O from own cell
                        cell_coordinates.append(cell_grid[i+1][j][k][0][7])  # O1 from x+ neighbor
                        cell_coordinates.append(cell_grid[i][j][k+1][0][16])  # H1 from z+ neighbor
                        cell_coordinates.append(cell_grid[i][j][k+1][0][19])  # H2 from z+ neighbor
                        cell_coordinates.append(cell_grid[i][j][k+1][0][25])  # HA1 from z+ neighbor
                        cell_coordinates.append(cell_grid[i+1][j][k+1][0][28])  # HA2 from (x+,z+) neighbor
                        cell_coordinates.append(cell_grid[i+1][j][k+1][0][22])  # H3 from (x+,z+) neighbor

                        # Add coordinates such that they correspond to Gly molecule
                        cell_atoms.append(cell_grid[i][j][k+1][1][1])  # N from z+ neighbor
                        cell_atoms.append(cell_grid[i+1][j][k+1][1][13])  # CA from (x+,z+) neighbor
                        cell_atoms.append(cell_grid[i+1][j][k][1][10])  # C from x+ neighbor
                        cell_atoms.append(cell_grid[i][j][k][1][4])  # O from own cell
                        cell_atoms.append(cell_grid[i+1][j][k][1][7])  # O1 from x+ neighbor
                        cell_atoms.append(cell_grid[i][j][k+1][1][16])  # H1 from z+ neighbor
                        cell_atoms.append(cell_grid[i][j][k+1][1][19])  # H2 from z+ neighbor
                        cell_atoms.append(cell_grid[i][j][k+1][1][25])  # HA1 from z+ neighbor
                        cell_atoms.append(cell_grid[i+1][j][k+1][1][28])  # HA2 from (x+,z+) neighbor
                        cell_atoms.append(cell_grid[i+1][j][k+1][1][22])  # H3 from (x+,z+) neighbor

                    #Check if the second single oxygen can be completed (if the z+, y+ and (z+,y+) neighbor cells are present)
                    if padded_mask[i+1][j+1][k+2] and padded_mask[i+1][j+2][k+1] and padded_mask[i][j+2][k+2]:

                        # Add coordinates such that they correspond to Gly molecule
                        cell_coordinates.append(cell_grid[i][j+1][k+1][0][0])   # N from (z+,y+) neighbor
                        cell_coordinates.append(cell_grid[i][j][k+1][0][12])    # CA from z+ neighbor
                        cell_coordinates.append(cell_grid[i][j][k+1][0][9])     # C from z+ neighbor
                        cell_coordinates.append(cell_grid[i][j+1][k+1][0][3])   # O from (z+,y+) neighbor
                        cell_coordinates.append(cell_grid[i][j][k][0][6])       # O1 from own cell
                        cell_coordinates.append(cell_grid[i][j+1][k+1][0][15])  # H1 from (z+,y+) neighbor
                        cell_coordinates.append(cell_grid[i][j+1][k+1][0][18])  # H2 from (z+,y+) neighbor
                        cell_coordinates.append(cell_grid[i][j+1][k+1][0][24])  # HA1 from (z+,y+) neighbor
                        cell_coordinates.append(cell_grid[i][j][k+1][0][27])    # HA2 from z+ neighbor
                        cell_coordinates.append(cell_grid[i][j][k+1][0][21])    # H3 from z+ neighbor

                        # Add coordinates such that they correspond to Gly molecule
                        cell_atoms.append(cell_grid[i][j+1][k+1][1][0])   # N from (z+,y+) neighbor
                        cell_atoms.append(cell_grid[i][j][k+1][1][12])    # CA from z+ neighbor
                        cell_atoms.append(cell_grid[i][j][k+1][1][9])     # C from z+ neighbor
                        cell_atoms.append(cell_grid[i][j+1][k+1][1][3])   # O from (z+,y+) neighbor
                        cell_atoms.append(cell_grid[i][j][k][1][6])       # O1 from own cell
                        cell_atoms.append(cell_grid[i][j+1][k+1][1][15])  # H1 from (z+,y+) neighbor
                        cell_atoms.append(cell_grid[i][j+1][k+1][1][18])  # H2 from (z+,y+) neighbor
                        cell_atoms.append(cell_grid[i][j+1][k+1][1][24])  # HA1 from (z+,y+) neighbor
                        cell_atoms.append(cell_grid[i][j][k+1][1][27])    # HA2 from z+ neighbor
                        cell_atoms.append(cell_grid[i][j][k+1][1][21])    # H3 from z+ neighbor

                    #Check if the second single hydrogen can be completed (if the z- neighbor cells are present)
                    if padded_mask[i+1][j+1][k]:

                        # Add coordinates such that they correspond to Gly molecule
                        cell_coordinates.append(cell_grid[i][j][k-1][0][2])   # N from z- neighbor
                        cell_coordinates.append(cell_grid[i][j][k-1][0][14])    # CA from z- neighbor
                        cell_coordinates.append(cell_grid[i][j][k-1][0][11])     # C from z- neighbor
                        cell_coordinates.append(cell_grid[i][j][k-1][0][5])   # O from z- neighbor
                        cell_coordinates.append(cell_grid[i][j][k-1][0][8])       # O1 from z- neighbor
                        cell_coordinates.append(cell_grid[i][j][k][0][17])  # H1 from own cell
                        cell_coordinates.append(cell_grid[i][j][k-1][0][20])  # H2 from z- neighbor
                        cell_coordinates.append(cell_grid[i][j][k-1][0][26])  # HA1 from z- neighbor
                        cell_coordinates.append(cell_grid[i][j][k-1][0][29])    # HA2 from z- neighbor
                        cell_coordinates.append(cell_grid[i][j][k-1][0][23])    # H3 from z- neighbor

                        # Add atom names such that they correspond to Gly molecule
                        cell_atoms.append(cell_grid[i][j][k-1][1][2])   # N from z- neighbor
                        cell_atoms.append(cell_grid[i][j][k-1][1][14])    # CA from z- neighbor
                        cell_atoms.append(cell_grid[i][j][k-1][1][11])     # C from z- neighbor
                        cell_atoms.append(cell_grid[i][j][k-1][1][5])   # O from z- neighbor
                        cell_atoms.append(cell_grid[i][j][k-1][1][8])       # O1 from z- neighbor
                        cell_atoms.append(cell_grid[i][j][k][1][17])  # H1 from own cell
                        cell_atoms.append(cell_grid[i][j][k-1][1][20])  # H2 from z- neighbor
                        cell_atoms.append(cell_grid[i][j][k-1][1][26])  # HA1 from z- neighbor
                        cell_atoms.append(cell_grid[i][j][k-1][1][29])    # HA2 from z- neighbor
                        cell_atoms.append(cell_grid[i][j][k-1][1][23])    # H3 from z- neighbor

                    #Add the grouped coordinates to the cell
                    new_grid[i][j][k] = (cell_coordinates,cell_atoms)
                    

    return new_grid


def collect_alpha_molecules(cell_grid, mask):
    """
    Takes the random ordering of the coordinates in the alpha glycine unit cell and groups atoms by molecules.
    A secondary result of this operation is that stray (incomplete) molecules are removed.

    PARAMETERS:
    -----------
        cell_grid: numpy.ndarray
            Each element is a unit cell that contains a (coordinates,atoms) list tuple.
        mask: numpy.ndarray
            A mask where each element contains an integer (0,1) denoting wether the unit cell is present.
    
    RETURNS:
    --------
        new_grid: numpy.ndarray
            Similar to cell_grid, but with a different ordering of atoms.
    """

    #Get size of the cell
    rx, ry, rz = cell_grid.shape

    #Indices of complete glycine molecules 1 and 2
    gly1 = [9,17,13,1,5,21,25,33,37,29]
    gly2 = [11,19,15,3,7,23,31,39,35,27]

    #get new grid
    new_grid = cell_grid.copy()
    
    #Add padding to the mask 
    padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
   
    #loop through the grid
    for i in range(rx):
        for j in range(ry):
            for k in range(rz):
                #check if current cell exists
                if mask[i][j][k] == 1:

                    #lists for  new ordering
                    cell_coordinates = []
                    cell_atoms = []
                    
                    #Add the elements
                    for index in gly1 + gly2:
                        cell_coordinates.append(cell_grid[i][j][k][0][index])
                        cell_atoms.append(cell_grid[i][j][k][1][index])

                    #Check if the first nitrogen group can be completed (if the necessary neighbor cells are present)
                    if padded_mask[i+1][j+1][k] and padded_mask[i+2][j+1][k+1] and padded_mask[i+2][j+1][k]:
                        
                        #Add coordinates such that they correspond to Gly molecule
                        cell_coordinates.append(cell_grid[i][j][k][0][10]) # N from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][18]) # CA from own cell
                        cell_coordinates.append(cell_grid[i][j][k-1][0][14]) # C from z- neighbor 
                        cell_coordinates.append(cell_grid[i][j][k-1][0][2])  # O from z- neighbor
                        cell_coordinates.append(cell_grid[i+1][j][k-1][0][6]) # O1 from (z-,x+) neighbor
                        cell_coordinates.append(cell_grid[i][j][k][0][22])    # H1 from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][30])    # H2 from own cell
                        cell_coordinates.append(cell_grid[i+1][j][k][0][38]) # HA1 from x+ neighbor 
                        cell_coordinates.append(cell_grid[i][j][k][0][34])   # HA2 from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][26])   # H3 from own cell

                        #Add atoms such that they correspond to Gly molecule
                        cell_atoms.append(cell_grid[i][j][k][1][10]) # N from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][18]) # CA from own cell
                        cell_atoms.append(cell_grid[i][j][k-1][1][14]) # C from z- neighbor 
                        cell_atoms.append(cell_grid[i][j][k-1][1][2])  # O from z- neighbor
                        cell_atoms.append(cell_grid[i+1][j][k-1][1][6]) # O1 from (z-,x+) neighbor
                        cell_atoms.append(cell_grid[i][j][k][1][22])    # H1 from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][30])    # H2 from own cell
                        cell_atoms.append(cell_grid[i+1][j][k][1][38]) # HA1 from x+ neighbor 
                        cell_atoms.append(cell_grid[i][j][k][1][34])   # HA2 from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][26])   # H3 from own cell

                    # Check if the second nitrogen group can be completed (if the necessary neighbor cells are present)
                    if padded_mask[i+1][j+1][k+2] and padded_mask[i][j+1][k+1] and padded_mask[i][j+1][+2]:

                        #Add coordinates such that they correspond to Gly molecule
                        cell_coordinates.append(cell_grid[i][j][k][0][8]) # N from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][16]) # CA from own cell
                        cell_coordinates.append(cell_grid[i][j][k+1][0][12]) # C from z+ neighbor 
                        cell_coordinates.append(cell_grid[i][j][k+1][0][0])  # O from z+ neighbor
                        cell_coordinates.append(cell_grid[i-1][j][k+1][0][4]) # O1 from (z+,x-) neighbor
                        cell_coordinates.append(cell_grid[i][j][k][0][20])    # H1 from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][24])    # H2 from own cell
                        cell_coordinates.append(cell_grid[i][j][k][0][32])  # HA1 from own cell 
                        cell_coordinates.append(cell_grid[i-1][j][k][0][36])   # HA2 from x- neighbor
                        cell_coordinates.append(cell_grid[i][j][k][0][28])   # H3 from own cell
                        
                        #Add atoms such that they correspond to Gly molecule
                        cell_atoms.append(cell_grid[i][j][k][1][8]) # N from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][16]) # CA from own cell
                        cell_atoms.append(cell_grid[i][j][k+1][1][12]) # C from z+ neighbor 
                        cell_atoms.append(cell_grid[i][j][k+1][1][0])  # O from z+ neighbor
                        cell_atoms.append(cell_grid[i-1][j][k+1][1][4]) # O1 from (z+,x-) neighbor
                        cell_atoms.append(cell_grid[i][j][k][1][20])    # H1 from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][24])    # H2 from own cell
                        cell_atoms.append(cell_grid[i][j][k][1][32])  # HA1 from own cell 
                        cell_atoms.append(cell_grid[i-1][j][k][1][36])   # HA2 from x- neighbor
                        cell_atoms.append(cell_grid[i][j][k][1][28])   # H3 from own cell
                    
                    #Add the grouped coordinates to the cell
                    new_grid[i][j][k] = (cell_coordinates,cell_atoms)

    return new_grid


def count_molecules(file_path):
    """Counts the number of unique molecules in a .gro file.
    Works different from the method defined in 'glycine_topology.py'

    PARAMETERS:
    -----------
        filename: str
            Path to the .gro file.

    RETURNS:
    --------
        num_gly: int
            The number of glycine molecules in the system
        num_sol: int
            The number of solvent molecules in the system
    """

    #Empty lists for storing unique molecules
    unique_gly_molecules = []
    unique_sol_molecules = []

    #Open file and read lines
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Check if the line is not empty

                #Split the line of spaces
                tokens = line.split()
                #Take first split as molecule name
                molecule_name = tokens[0]
            
                #add molecule to set
                if molecule_name.endswith("GLY"):
                    unique_gly_molecules.append(molecule_name)
                elif molecule_name.endswith("SOL"):
                    unique_sol_molecules.append(molecule_name)

    #Calculate number of molecules from number of atoms
    num_gly = int(len(unique_gly_molecules)/10)
    num_sol = int(len(unique_sol_molecules)/3)

    return num_gly, num_sol


def build_crystal(rx, ry, rz, morph_type = "alpha"):
    """Creates the atom coordinates for a glycine crystalline structure of given shape and size.

    PARAMETERS:
    -----------
        rx: int
            Number of times to copy the unit cell in the x direction.
        ry: int
            Number of times to copy the unit cell in the y direction.
        rz: int
            Number of times to copy the unit cell in the z direction.
        morph_type: str 
            The type of polymorph crystal to build. Choices are: alpha, beta, gamma.
    """

    #Load the crystal unit cell from file
    filename = f'Data/Input/System/{morph_type}_glycine_unit_cell'
    coordinates, lattice, atoms = read_xyz_file(filename)

    #Build the crystalline structure by replicating the unit cell
    crystal_coordinates = copy_unit_cell(coordinates, atoms, lattice, rx, ry, rz)

    #Get size of the crystal
    size = np.array(crystal_coordinates.shape)

    #Create a mask for identifying borders
    mask = create_mask(size)

    return crystal_coordinates, mask, lattice


def solvate_crystal(filename, box_size, morph_type, rx, ry, rz, max_sol = np.inf):
    """Use gromacs to fill a simulation box with water molecules.
    
    PARAMETERS:
    -----------
        filename: str
            Path to the .gro crystal file to be solvated.
        box_size: float
            Length of one side of the cubic simulation box in nanometer.
        max_sol: int
            Maximum number of water molecules to add.
    """

    #Redefine box edges
    os.system(f"gmx editconf -f {filename} -o Data/Output/System/Crystal/{morph_type}_glycine_crystal_{rx}_{ry}_{rz}_box_{box_size}.gro -box {box_size} {box_size} {box_size}")

    #Use Gromacs to fill the box with water
    if max_sol == np.inf:
        #Fill completely
        os.system(f"gmx solvate -cp Data/Output/System/Crystal/{morph_type}_glycine_crystal_{rx}_{ry}_{rz}_box_{box_size}.gro -cs spc216.gro -o Data/Output/System/Crystal/{morph_type}_glycine_crystal_{rx}_{ry}_{rz}_box_{box_size}_solv.gro")
    else:
        #Fill with set amount
        os.system(f"gmx solvate -cp Data/Output/System/Crystal/{morph_type}_glycine_crystal_{rx}_{ry}_{rz}_box_{box_size}.gro -o Data/Output/System/Crystal/{morph_type}_glycine_crystal_{rx}_{ry}_{rz}_box_{box_size}_solv.gro -maxsol {max_sol}")


def build_crystal_system(rx,ry,rz, morph_type, box_size, max_sol=np.inf):
    """Builds a molecular system of a rectangular glycine crystalline structure of given shape and size, solvates the system and saves it to file.

    PARAMETERS:
    -----------
        rx: int
            Number of times to copy the unit cell in the x direction.
        ry: int
            Number of times to copy the unit cell in the y direction.
        rz: int
            Number of times to copy the unit cell in the z direction.
        morph_type: str 
            The type of polymorph crystal to build. Choices are: alpha, beta, gamma.
        box_size: float
            Length of one side of the cubic simulation box in nanometer.
        max_sol: int
            Maximum number of water molecules to add.
    """
    
    #Build the unsolvated crystal
    cell_grid, mask, lattice = build_crystal(rx,ry,rz, morph_type)

    #Check for crystal type
    if morph_type == "alpha":
        #Collect atoms by molecule and lose stray atoms for alpha glycine crystal
        cell_grid = collect_alpha_molecules(cell_grid, mask)

    if morph_type == "beta":
        raise NotImplemented
    
    if morph_type == "gamma":
        cell_grid = collect_gamma_molecules(cell_grid, mask)

    #Save the initial crystal
    write_gro_file(f"Data/Output/System/Crystal/{morph_type}_glycine_crystal_{rx}_{ry}_{rz}.gro", cell_grid, lattice, mask, morph_type)

    #Convert the .gro file (Overwrites the input file)
    convert_coordinates(f"Data/Output/System/Crystal/{morph_type}_glycine_crystal_{rx}_{ry}_{rz}.gro")

    #Solvate the crystal with water
    solvate_crystal(f"Data/Output/System/Crystal/{morph_type}_glycine_crystal_{rx}_{ry}_{rz}.gro", box_size, morph_type, rx, ry, rz, max_sol)

    #Convert the .gro file (Overwrites the input file)
    convert_coordinates(f"Data/Output/System/Crystal/{morph_type}_glycine_crystal_{rx}_{ry}_{rz}_box_{box_size}_solv.gro")

    #Count the number of molecules in the solvated system
    num_gly, num_sol = count_molecules(f"Data/Output/System/Crystal/{morph_type}_glycine_crystal_{rx}_{ry}_{rz}_box_{box_size}_solv.gro")

    #Create corresponding topology file for solvated crystal coordinates
    convert_topology(f"Data/Output/System/Crystal/{morph_type}_glycine_crystal_{rx}_{ry}_{rz}_box_{box_size}_solv.top", num_gly, num_sol)

    #Create corresponding topology file for UNsolvated crystal coordinates
    convert_topology(f"Data/Output/System/Crystal/{morph_type}_glycine_crystal_{rx}_{ry}_{rz}_box_{box_size}.top", num_gly, 0)

    #Report on resulting system
    gly_con = calc_concentration(num_gly,num_sol)
    print(f"Created a {morph_type} glycine crystal of {rx}x{ry}x{rz} unit cells consisting of {num_gly} glycine molecules.\nThe crystal is surrounded by a {box_size}x{box_size}x{box_size} nm box with {num_sol} water molecules.\nThe concentration of glycine is {gly_con} mol/L")


#PARAMETERS:
rx, ry, rz = 3, 1, 3
morph_type = "gamma"
mask_type = "square"
box_size = 5.0

#Build a solvated glycine crystal system of monomers
build_crystal_system(rx, ry, rz, morph_type, box_size)

