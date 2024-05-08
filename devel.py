from kim_python_utils.ase import CrystalGenomeTest
from numpy import multiply
from math import pi
from crystal_genome_util.aflow_util import get_stoich_reduced_list_from_prototype
from copy import deepcopy
from ase.build import bulk
import subprocess
import kim_convergence as cr
 
class EquilibriumCrystalStructureFiniteT(CrystalGenomeTest):
    def _calculate(self, structure_index: int, temperature: float):
        """
        structure_index:
            KIM tests can loop over multiple structures (i.e. crystals, molecules, etc.). 
            This indicates which is being used for the current calculation.

        temperature:
            The temperature for which the equilibrium structure is computed.
        """
        atoms_single_unit_cell = self.atoms[structure_index]
        print(atoms_single_unit_cell)

        # CALCULATIONS HAPPEN FROM HERE
        #Replicate to create 10*10*10 supercell
        supercell = atoms_single_unit_cell.repeat(10) # a single r=10 is equivalent to (10,10,10)
        print(supercell)

        #write the supercell in lammpsdata file for simulation in lammps
        supercell.write("atoms.lammpsdata",format='lammps-data')

        #.. call to LAMMPS to perfom simulation to get the time average a,b,c,alpha,beta,gamma and internal atoms at the specified temperature
        subprocess.run(["lmp","-in","lammps.in","-v","temperature",str(temperature),"-v","model",self.model_name,"-v","elements"," ".join(self.stoichiometric_species)])

        #The kim convergence package is directly used in the lammps script.
        
 
        #.. average over cells in the supercell to get an average conventional unit cell and put that into an atoms object
        #.. test whether its the same prototype that came in (call Ilia's routine with average atoms object)
        #.. if prototype changed, report error, otherwise continue to following code that writes the property
        # TO HERE

        # Write out the the property
        self._add_property_instance("crystal-structure-npt")
        self._add_common_crystal_genome_keys_to_current_property_instance(structure_index,write_stress=False,write_temp=temperature)

# This queries for equilibrium structures in this prototype and builds atoms
# test = BindingEnergyVsWignerSeitzRadius(model_name="MEAM_LAMMPS_KoJimLee_2012_FeP__MO_179420363944_002", stoichiometric_species=['Fe','P'], prototype_label='AB_oP8_62_c_c')
                
# Alternatively, for debugging, give it atoms object or a list of atoms objects
atoms = bulk('AlCo','cesiumchloride',a=2.82)   
test = EquilibriumCrystalStructureFiniteT(model_name="EAM_Dynamo_PunYamakovMishin_2013_AlCo__MO_678952612413_000", atoms=atoms)
test(temperature=300)

