import os
import re 
import numpy as np

def generate_coordinates(num_gly, box_size):
    """Generates the initial coordinates for a simulation box of glycine molecules in water.
    Needs glycine.top and Glycine_match.pdb to be accessible in the directory.

    PARAMETERS:
    -----------
        num_gly: int
            total number of glycine in water
        box_size: float
            length of one side of simulation cubic (nanometers)
    """
    
    #Use Gromacs to build an empty simulation box
    os.system(f"gmx insert-molecules -ci Data/Input/System/glycine_match.pdb -nmol {num_gly} -box {box_size} {box_size} {box_size} -o Data/Output/System/Monomer/gly_box_{num_gly}_{box_size}.pdb")

    #Use Gromacs to fill the box with water
    os.system(f"gmx solvate -cp Data/Output/System/Monomer/gly_box_{num_gly}_{box_size}.pdb -cs spc216.gro -o Data/Output/System/Monomer/gly_box_{num_gly}_{box_size}_solv.gro")


def water_box(box_size):
    """Use Gromacs to fill an empty box completely with water
    
    PARAMETERS:
    -----------
        box_size: int
            length of one side of simulation cubic (nanometers)
    """

    #Create an empty box
    os.system(f"gmx editconf -f /usr/share/gromacs/top/spc216.gro -bt cubic -box {box_size} {box_size} {box_size} -o empty_box_{box_size}_{box_size}_{box_size}.gro")
    
    #Create topology file for solvate function
    os.system(f"gmx pdb2gmx -f empty_box_{box_size}_{box_size}_{box_size}.gro -o processed_box_{box_size}_{box_size}_{box_size}.gro -water spce")
    
    #Fill the box with water
    os.system(f"gmx solvate -cp processed_box_{box_size}_{box_size}_{box_size}.gro -cs /usr/share/gromacs/top/spc216.gro -o solvated_box_{box_size}_{box_size}_{box_size}.gro -p topol.top")


def count_molecules(file_path):
    """Counts the number of glycine and water molecules from a .gro file.
    Needs access to the .gro file in the directory.

    PARAMETERS:
    -----------
        filename: str
            path to the .gro file to read

    RETURNS:
    --------
        unique_gly: int
            number of unique gly molecules
        unique_sol: int
            number of unique gly molecules
    """

    #Empty sets for storing unique molecules
    unique_gly_molecules = set()
    unique_sol_molecules = set()

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
                    unique_gly_molecules.add(molecule_name)
                elif molecule_name.endswith("SOL"):
                    unique_sol_molecules.add(molecule_name)

    return len(unique_gly_molecules), len(unique_sol_molecules)


def convert_coordinates(file_path):
    """Reads a .gro file and converts it into a format such that it can be used with 
    existing glycine.top. It will overwrite the existing file.

    PARAMETERS:
    -----------
        filename: str
            path to the .gro file to read
    """

    #Read the original file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    #Write a new file line by line
    with open(file_path, 'w') as file:
        for line in lines:
            if line.strip():  # Check if the line is not empty

                #Split string preserving spacing
                tokens = re.split(r'(\s)',line)
                #Remove empty strings from the tokens
                tokens = [x for x in tokens if x != '']
                
                #Get index of molecule name
                index = next(x[0] for x in enumerate(tokens) if len(x[1]) > 1)

                #Rename every atom as 1          
                if tokens[index].endswith("GLY"):
                    tokens = tokens[index:]
                    tokens[0] = "    1GLY"
                elif tokens[index].endswith("SOL"):
                    #Remove spaces before molecule name
                    tokens = tokens[index:]
                    tokens[0] = "    1SOL"

                #Write the updated line
                file.write("".join(tokens))


def convert_topology(output_file, num_gly, num_sol):
    """Reads a .top file and edits it such that it matches the correct number of unique molecules.
    Creates a new .top file.

    PARAMETERS:
    -----------
        output_file: str
            name for the converted .top file
        num_gly: int
            number of glycine molecules to write in .top file
        num_sol: int 
            number of water molecules to write in .top file
    """

    #Read the template topology 
    with open("Data/Input/System/glycine.top", 'r') as file:
        lines = file.readlines()

    #Write new file line by line
    with open(output_file, 'w') as file:
        #Bool for ensuring system name does not get overwritten
        system_name_encountered = False

        #Loop through lines in template
        for line in lines:
            #Change the entry for the system name
            if system_name_encountered == True:
                line = f"{num_gly} glycine molecules in {num_sol} water molecules\n"
                system_name_encountered = False
            elif line.strip().startswith("[ system ]"):
                system_name_encountered = True
            #Change the entry for glycine
            elif line.strip().startswith("glycine"):
                line = f"glycine          {num_gly}\n"
            #Change the entry for Solvent
            elif line.strip().startswith("SOL"):
                line = f"SOL              {num_sol}\n"

            file.write(line)


def calc_concentration(num_gly, num_sol):
    """Given number of glycine and water molecules calculates the concentration in mol/kg
    
    PARAMETERS:
    -----------
        num_gly: int
            number of glycine molecules in solution
        num_sol: int
            number of water molecules in solution
    
    RETURNS:
    --------
        conc_gly: float
            concentration of glycine in solution (mol/kg)
    """

    #Calculate mass of water in grams
    mass_sol = 18.015 * num_sol/1000 

    #Calculate concentration of glycine molecules
    conc_gly = num_gly/mass_sol

    return round(conc_gly,2)


def build_system(num_gly, box_size):
    """Builds a molecular system of Glycine monomers solvated in water.
    It creates a .gro and .top file that can be used directly with Gromacs.

    PARAMETERS:
    -----------
        num_gly: int
            number of glycine molecules to add to the system.
        box_size: float
            the size of one length of the box in nanometer.
    """

    #Generate molceular system of Glycine in water
    generate_coordinates(num_gly, box_size)

    #Read the total number molecules from the file
    num_gly, num_sol = count_molecules(f"Data/Output/System/Monomer/gly_box_{num_gly}_{box_size}_solv.gro")

    #Convert .gro file such that it matches .top
    convert_coordinates(f"Data/Output/System/Monomer/gly_box_{num_gly}_{box_size}_solv.gro")

    #Account for changes in .top file
    convert_topology(f"Data/Output/System/Monomer/gly_box_{num_gly}_{box_size}_solv.top", num_gly, num_sol)

    #Report on resulting system
    gly_con = calc_concentration(num_gly,num_sol)
    print(f"Created a system of {num_gly} glycine molecules solvated in {num_sol} water molecules.\nThe system is surrounded by a {box_size}x{box_size}x{box_size} nm .\nThe concentration of glycine is {gly_con} mol/L")


# print(calc_concentration(510,2600))
# build_system(430, 5.0)






