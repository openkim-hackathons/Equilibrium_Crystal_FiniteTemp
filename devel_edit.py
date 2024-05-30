import os
import random
import subprocess
from kim_python_utils.ase import CrystalGenomeTest
from numpy import multiply
from math import pi
from crystal_genome_util.aflow_util import get_stoich_reduced_list_from_prototype
from copy import deepcopy
from ase.build import bulk
from ase import Atoms
import subprocess
import kim_convergence as cr
from typing import Iterable, List, Optional, Tuple
import numpy.typing as npt
import numpy as np
import re

class EquilibriumCrystalStructureFiniteT(CrystalGenomeTest):
    def _calculate(self, structure_index: int, temperature: float, timestep: float, pressure: float, temperature_offset_fraction: float, number_sampling_timesteps: int, repeat: Tuple[int, int, int]=(3,3,3), seed: Optional[int] = None, loose_triclinic_and_monoclinic=False, **kwargs) -> None:
        """
        structure_index:
            KIM tests can loop over multiple structures (i.e. crystals, molecules, etc.). 
            This indicates which is being used for the current calculation.

        temperature:
            The temperature for which the equilibrium structure is computed.
        """
        #copy original atoms so that their information does not get lost when the new atoms are modified.
        atoms_new = self.atoms.copy()

        #atoms_single_unit_cell = self.atoms[structure_index]
        #print(atoms_single_unit_cell)

        # CALCULATIONS HAPPEN FROM HERE
        #Replicate to create 10*10*10 supercell
        #supercell = atoms_single_unit_cell.repeat(10) # a single r=10 is equivalent to (10,10,10)
        #print(supercell)
        

        #Test a triclinic structure
        atoms_new  = bulk("AlCo", "cesiumchloride", a=2.82)
        # This is how ASE obtains the species that are written to the initial configuration.
        # These species are passed to kim interactions.
        # See https://wiki.fysik.dtu.dk/ase/_modules/ase/io/lammpsdata.html#write_lammps_d
        symbols = atoms_new.get_chemical_symbols()
        species = sorted(set(symbols))

        #Build supercell
        supercell = atoms_new.repeat(repeat)
        #write the supercell in lammpsdata file for simulation in lammps
        supercell.write("atoms.lammpsdata",format='lammps-data')
        
        # Get random 31-bit unsigned integer.
        if seed is None:
            seed = random.getrandbits(31)
        
        # TODO: Move damping factors to argument.
        pdamp = timestep * 100.0
        tdamp = timestep * 1000.0       
        
        # Run NPT simulation for equilibration.
        # TODO: If we notice that this takes too long, maybe use an initial temperature ramp.
        variables = {
            "model_name": self.model_name,
            "temperature": temperature,
            "temperature_seed": seed,
            "temperature_damping": tdamp,
            "pressure": pressure,
            "pressure_damping": pdamp,
            "timestep": timestep,
            "number_sampling_timesteps": number_sampling_timesteps,
            "species": " ".join(species),
            "log_filename": "output/lammps_equilibration.log",
            "average_position_filename": "output/average_position_equilibration.dump.*",
            "write_restart_filename": "output/final_configuration_equilibration.restart" 
        }
        
        # TODO: Maybe use initial temperature ramp. 
        # Call LAMMPS to perform simulation
        command = (
            "lammps " 
            + " ".join(f"-var {key} '{item}'" for key, item in variables.items()) 
            + " -in lammps.in")
        subprocess.run(command, check=True, shell=True)
        #This is another way to run the lammps code here
        #subprocess.run(["lmp","-in","lammps.in","-v","temperature",str(temperature),"-v","model",self.model_name,"-v","species"," ".join(self.stoichiometric_species)])
        #write out the property 
        self._add_property_instance("crystal-structure-npt")
        self._add_common_crystal_genome_keys_to_current_property_instance(structure_index,write_stress=True,write_temp=temperature)

        #.. average over cells in the supercell to get an average conventional unit cell and put that into an atoms object

        #lammps does not create scaled wrapped coordinates system. For this, a new function is created and all dump files were used to find the average atoms positions
        self._compute_average_positions_from_lammps_dump("./output","average_position_equilibration.dump")
  
        #Check symmetry post NPT
        atoms_new.set_cell(self._get_cell_from_lammps_dump("output/average_position_equilibration_over_dump.out"))
        atoms_new.set_scaled_positions(self._get_positions_from_lammps_dump("output/average_position_equilibration_over_dump.out"))

        #Reduce and average
        reduced_atoms = self._reduce_and_avg(atoms_new, repeat)
        
        #Aflow symmetry check
        self._get_crystal_genome_designation_from_atoms_and_verify_unchanged_symmetry(reduced_atoms, loose_triclinic_and_monoclinic=loose_triclinic_and_monoclinic)
        

        #Define all the functions here##############################################################################################################################

    @staticmethod
    def _reduce_and_avg(atoms: Atoms, repeat: Tuple[int, int, int]) -> Atoms:
        '''
        Function to reduce all atoms to the original unit cell position.
        '''
        new_atoms = atoms.copy()

        cell = new_atoms.get_cell()
        
        # Divide each unit vector by its number of repeats.
        # See https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element.
        cell = cell / np.array(repeat)[:, None]

        # Decrease size of cell in the atoms object.
        new_atoms.set_cell(cell)
        new_atoms.set_pbc((True, True, True))

        # Set averaging factor
        M = np.prod(repeat)

        # Wrap back the repeated atoms on top of the reference atoms in the original unit cell.
        positions = new_atoms.get_positions(wrap=True)
        
        number_atoms = len(new_atoms)
        original_number_atoms = number_atoms // M
        assert number_atoms == original_number_atoms * M
        positions_in_prim_cell = np.zeros((original_number_atoms, 3))

        # Start from end of the atoms because we will remove all atoms except the reference ones.
        for i in reversed(range(number_atoms)):
            if i >= original_number_atoms:
                # Get the distance to the reference atom in the original unit cell with the 
                # minimum image convention.
                distance = new_atoms.get_distance(i % original_number_atoms, i,
                                                  mic=True, vector=True)
                # Get the position that has the closest distance to the reference atom in the 
                # original unit cell.
                position_i = positions[i % original_number_atoms] + distance
                # Remove atom from atoms object.
                new_atoms.pop()
            else:
                # Atom was part of the original unit cell.
                position_i = positions[i]
            # Average.
            positions_in_prim_cell[i % original_number_atoms] += position_i / M

        new_atoms.set_positions(positions_in_prim_cell)
    @staticmethod
    def _get_property_from_lammps_log(in_file_path:str="./output/lammps_equilibration.log",property_names:list = ["v_vol_metal", "v_temp_metal"]):
        '''
        The function to get the value of the property with time from ***.log 
        the extracted data are stored as ***.csv and ploted as property_name.png
        data_dir --- the directory contains lammps_equilibration.log 
        property_names --- the list of properties
        '''
        def get_table(in_file):
            if not os.path.isfile(in_file):
                raise FileNotFoundError(in_file + "not found")
            elif not ".log" in in_file:
                raise FileNotFoundError("The file is not a *.log file")
            is_first_header = True
            header_flags  = ["Step", "v_pe_metal", "v_temp_metal", "v_press_metal"]
            eot_flags  = ["Loop", "time", "on", "procs", "for", "steps"]
            table = []
            with open(in_file, "r") as f:
                line = f.readline()
                while line: # not EOF
                    is_header = True
                    for _s in header_flags:
                        is_header = is_header and (_s in line)
                    if is_header:
                        if is_first_header:
                            table.append(line)
                            is_first_header = False
                        content = f.readline()
                        while content:
                            is_eot = True
                            for _s in eot_flags:
                                is_eot = is_eot and (_s in content)
                            if not is_eot:
                                table.append(content)
                            else:
                                break
                            content = f.readline()
                    line = f.readline()
            return table
        def write_table(table,out_file):
            with open(out_file, "w") as f:
                for l in table:
                    f.writelines(l)
        dir_name = os.path.dirname(in_file_path)
        in_file_name = os.path.basename(in_file_path)
        out_file_path = os.path.join(dir_name,in_file_name.replace(".log",".csv"))
    
        table = get_table(in_file_path)
        write_table(table,out_file_path)
        df = np.loadtxt(out_file_path, skiprows=1)
    
        for property_name in property_names:
            with open(out_file_path) as file:
                first_line = file.readline().strip("\n")
            property_index = first_line.split().index(property_name)
            properties = df[:, property_index]
            step = df[:, 0]
            plt.plot(step, properties)
            plt.xlabel("step")
            plt.ylabel(property_name)
            img_file =  os.path.join(dir_name, in_file_name.replace(".log","_")+property_name +".png")
            plt.savefig(img_file, bbox_inches="tight")
            plt.close()
    
    @staticmethod
    def _compute_average_positions_from_lammps_dump(data_dir:str = "./output",file_str = "average_position.dump"):
        '''
        This function compute the average position over *.dump files which contains the file_str (default:average_position.dump) in data_dir and output it
        to data_dir/[file_str]_over_dump.out
 
        input:
        data_dir-- the directory contains all the data e.g average_position.dump.* files
        '''

        def get_id_pos_dict(file_name:str) -> dict:
            '''
            input: 
            file_name--the file_name that contains average postion data
            output:
            the dictionary contains id:position pairs e.g {1:array([x1,y1,z1]),2:array([x2,y2,z2])}
            for the averaged positions over files
            '''
            id_pos_dict = {}
            header4N = ["NUMBER OF ATOMS"]
            header4pos = ["id","f_avePos[1]","f_avePos[2]","f_avePos[3]"]
            is_table_started = False
            is_natom_read = False
            with open(file_name,"r") as f:
                line = f.readline()
                count_content_line = 0
                N = 0
                while line:
                    if not is_natom_read:
                        is_natom_read = np.all([flag in line for flag in header4N])
                        if is_natom_read:
                            line = f.readline()
                            N = int(line)
                    if not is_table_started:
                        contain_flags = np.all([flag in line for flag in header4pos])
                        is_table_started = contain_flags
                    else:
                        count_content_line += 1        
                        words = line.split()
                        id = int(words[0])
                        #pos = np.array([float(words[2]),float(words[3]),float(words[4])])
                        pos = np.array([float(words[1]),float(words[2]),float(words[3])])
                        id_pos_dict[id] = pos 
                    if count_content_line > 0 and count_content_line >= N:
                        break
                    line = f.readline()
            if count_content_line < N:
                print("The file " + file_name +
                      " is not complete, the number of atoms is smaller than " + str(N))
            return id_pos_dict

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(data_dir + " does not exist")
        if not ".dump" in file_str:
            raise ValueError("file_str must be a string containing .dump")

        # extract and store all the data
        pos_list = []
        max_step,last_step_file = -1, ""
        for file_name in os.listdir(data_dir):
            if file_str in file_name:
                file_path = os.path.join(data_dir,file_name)
                id_pos_dict = get_id_pos_dict(file_path)
                id_pos = sorted(id_pos_dict.items())
                id_list = [pair[0] for pair in id_pos]
                pos_list.append([pair[1] for pair in id_pos])
                # check if this is the last step
                step = int(re.findall(r'\d+', file_name)[-1])
                if step > max_step:
                    last_step_file,max_step = os.path.join(data_dir ,file_name),step 
        pos_arr = np.array(pos_list)
        avg_pos = np.mean(pos_arr,axis=0)
        # get the lines above the table from the file of the last step
        with open(last_step_file,"r") as f:
            header4pos = ["id","f_avePos[1]","f_avePos[2]","f_avePos[3]"]
            line = f.readline()
            description_str = ""
            is_table_started = False
            while line:
                description_str += line
                is_table_started = np.all([flag in line for flag in header4pos])
                if is_table_started:
                    break
                else:
                    line = f.readline()
        # write the output to the file
        output_file = os.path.join(data_dir,file_str.replace(".dump","_over_dump.out"))
        with open(output_file,"w") as f:
            f.write(description_str)
            for i in range(len(id_list)):
                f.write(str(id_list[i]))
                f.write("  ")
                for dim in range(3):
                    f.write('{:3.6}'.format(avg_pos[i,dim]))
                    f.write("  ")
                f.write("\n")

    @staticmethod
    def _get_positions_from_lammps_dump(filename: str) -> List[Tuple[float, float, float]]:
        lines = sorted(np.loadtxt(filename, skiprows=9).tolist(), key = lambda x: x[0])     
        return [(line[1], line[2], line[3]) for line in lines]

    @staticmethod
    def _get_cell_from_lammps_dump(filename: str) -> npt.NDArray[float]:
        new_cell = np.loadtxt(filename, skiprows=5, max_rows=3)
        assert new_cell.shape == (3, 2) or new_cell.shape == (3, 3)

        # See https://docs.lammps.org/Howto_triclinic.html.
        xlo_bound = new_cell[0,0]
        xhi_bound = new_cell[0,1]
        ylo_bound = new_cell[1,0]
        yhi_bound = new_cell[1,1]
        zlo_bound = new_cell[2,0]
        zhi_bound = new_cell[2,1]

        # If not cubic add more cell params
        if new_cell.shape[-1] != 2:
            xy = new_cell[0,2]
            xz = new_cell[1,2]
            yz = new_cell[2,2]
        else:
            xy = 0.0
            xz = 0.0
            yz = 0.0
        
        xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
        xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
        ylo = ylo_bound - min(0.0, yz)
        yhi = yhi_bound - max(0.0, yz)
        zlo = zlo_bound
        zhi = zhi_bound
        
        cell = np.empty(shape=(3, 3))
        cell[0, :] = np.array([xhi - xlo, 0.0, 0.0])
        cell[1, :] = np.array([xy, yhi - ylo, 0.0])
        cell[2, :] = np.array([xz, yz, zhi - zlo])
        return cell

        return new_atoms

        #.. if prototype changed, report error, otherwise continue to following code that writes the property
        # TO HERE
        #END OF function DEFINITION #########################################################################################################################

# This queries for equilibrium structures in this prototype and builds atoms
# test = BindingEnergyVsWignerSeitzRadius(model_name="MEAM_LAMMPS_KoJimLee_2012_FeP__MO_179420363944_002", stoichiometric_species=['Fe','P'], prototype_label='AB_oP8_62_c_c')
                
# Alternatively, for debugging, give it atoms object or a list of atoms objects   
if __name__ == "__main__":
    atoms = bulk("AlCo", "cesiumchloride", a=2.82)
    model_name = "EAM_Dynamo_PunYamakovMishin_2013_AlCo__MO_678952612413_000"
    subprocess.run(f"kimitems install {model_name}", shell=True, check=True)
    test = EquilibriumCrystalStructureFiniteT(model_name, atoms=atoms)
    test(temperature=300.0, pressure=1.0, temperature_offset_fraction=0.01, timestep=0.001, number_sampling_timesteps=100, loose_triclinic_and_monoclinic=False)

