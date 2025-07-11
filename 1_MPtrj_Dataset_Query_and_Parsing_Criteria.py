#!/usr/bin/env python3

# AUTHOR :: Asif IQBAL BHATTI
#
# USAGE:: New version of https://github.com/CederGroupHub/chgnet/blob/main/examples/QueryMPtrj.md
#
#
#

from __future__ import annotations
from mp_api.client import MPRester
from tqdm import tqdm
from pymatgen.analysis.structure_matcher import StructureMatcher


MP_API_KEY = "NAN"
mpr = MPRester(MP_API_KEY)

material_ids = mpr.materials.summary.search(
    elements = ['Co', 'S'],
    #material_ids = ['mp-90'],
    energy_above_hull = [0, 100],
    fields = ['material_id']
)

def calc_type_equal(task_doc, main_entry,) -> bool:
    # Check the LDAU of task
    try:
        is_hubbard = task_doc.calcs_reversed[0].input.parameters['LDAU']
    except:
        is_hubbard = task_doc.calcs_reversed[0].input.incar['LDAU']

    # Make sure we don't include both GGA and GGA+U for the same mp_id
    #print(is_hubbard)
    if main_entry.parameters['is_hubbard'] != is_hubbard:
        print(f'{main_entry.entry_id}, {task_doc.task_id} is_hubbard= {is_hubbard}')
        return False
    elif is_hubbard == True:
        # If the task is calculated with GGA+U
        # Make sure the +U values are the same for each element
        composition = task_doc.output.structure.composition
        #print(task_doc.calcs_reversed[0].input.parameters['LDAUU'])
        hubbards = {element.symbol: U for element, U in
                    zip(composition.elements,task_doc.calcs_reversed[0].input.incar['LDAUU'])
                    }
        #print(hubbards)            
        if main_entry.parameters['hubbards'] != hubbards:
            thermo_hubbards = main_entry.parameters['hubbards']
            return False
        else:
            # Check the energy convergence of the task wrt. the main entry
            return check_energy_convergence(
                task_doc,
                main_entry.uncorrected_energy_per_atom,
            )
    else:
        # Check energy convergence for pure GGA tasks
        check_energy_convergence(
            task_doc,
            main_entry.uncorrected_energy_per_atom,
        )

def check_energy_convergence(task_doc,relaxed_entry_uncorrected_energy_per_atom,) -> bool:
    task_energy = task_doc.calcs_reversed[0].output.ionic_steps[-1].e_fr_energy
    n_atom = task_doc.calcs_reversed[0].output.ionic_steps[-1].structure.composition.num_atoms
    e_per_atom = task_energy / n_atom
    # This is the energy difference of the last frame of the task vs main_entry energy
    e_diff = abs(e_per_atom - relaxed_entry_uncorrected_energy_per_atom)

    if e_diff < 0.02:
        # The task is properly relaxed if the final frame has less than 20 meV/atom difference
        return True
    else:
        # The task is falsely relaxed, we will discard the whole task
        # This step will filter out tasks that relaxed into different spin states
        # that caused large energy discrepancies
        f'e_diff is too large, '
        f'task last step energy_per_atom = {e_per_atom}, '
        f'relaxed_entry_uncorrected_e_per_atom = {relaxed_entry_uncorrected_energy_per_atom}'
        return False



class UniquenessCheck:
    """Check whether a frame in trajectory is valid and unique."""

    def __init__(
        self,
        main_entry_uncorrected_energy_per_atom,
        ltol=0.002,
        stol=0.001,
        angle_tol=0.05,
    ):
        self.uniq_struct_list = []
        self.relaxed_entry_uncorrected_energy_per_atom = (
            main_entry_uncorrected_energy_per_atom
        )
        self.added_relaxed = False
        self.matcher = StructureMatcher(
            ltol=ltol, stol=stol, angle_tol=angle_tol, scale=False
        )
        self.energy_threshold = 0.002

    def is_unique(self, step, struct_id, NELM, mag=None):
        self.adjust_matcher(mag=mag)
        struct = step.structure
        energy = step.e_fr_energy / struct.composition.num_atoms

        # Check whether a frame is valid
        # Discard frame with no energy
        if energy is None:
            return False

        # Always accept the relaxed frame on Materials Project website
        if (
            abs(energy - self.relaxed_entry_uncorrected_energy_per_atom) < 1e-5
            and self.added_relaxed is False
        ):
            # we prioritize to add the relaxed entry
            self.uniq_struct_list.append((energy, struct))
            self.added_relaxed = True
            return True

        # Discard frame with electronic step not converged
        if len(step.electronic_steps) == NELM:
            return False

        e_diff = energy - self.relaxed_entry_uncorrected_energy_per_atom
        if e_diff < -0.01:
            # Discard frame that is more stable than the Materials Project website frame
            return False
        if e_diff > 1:
            # Discard frame that is too unstable than the Materials Project website frame
            return False

        # Now we're in uniqueness check
        # Accept the frame if we still have no frame parsed
        if len(self.uniq_struct_list) == 0:
            self.uniq_struct_list.append((energy, struct))
            return True

        min_e_diff = min(
            [abs(energy - tmp[0]) for tmp in self.uniq_struct_list]
        )

        if min_e_diff > self.energy_threshold:
            # Accept the frame if its energy is different from all parsed frames
            self.uniq_struct_list.append((energy, struct))
            return True

        # Discard the frame if it's structural similar to another frame in the uniq_struc_list
        for uniq_energy, uniq_struct in self.uniq_struct_list:
            if self.matcher.fit(struct, uniq_struct):
                return False

        # Accept the frame
        self.uniq_struct_list.append((energy, struct))
        return True

    def adjust_matcher(self, mag):
        if mag is not None:
            # prioritize frame with mag
            self.matcher = StructureMatcher(
                ltol=0.002, stol=0.001, angle_tol=0.05, scale=False
            )
            self.energy_threshold = 0.002

        if len(self.uniq_struct_list) > 50:
            self.matcher = StructureMatcher(
                ltol=0.3, stol=0.2, angle_tol=3, scale=False
            )
            self.energy_threshold = 0.03
        elif len(self.uniq_struct_list) > 30:
            self.matcher = StructureMatcher(
                ltol=0.1, stol=0.1, angle_tol=1, scale=False
            )
            self.energy_threshold = 0.01
        elif len(self.uniq_struct_list) > 10:
            self.matcher = StructureMatcher(
                ltol=0.01, stol=0.01, angle_tol=0.1, scale=False
            )
            self.energy_threshold = 0.005
        else:
            self.matcher = StructureMatcher(
                ltol=0.002, stol=0.001, angle_tol=0.05, scale=False
            )
            self.energy_threshold = 0.002



opt_task_types = [
    'GGA Static', 'GGA Structure Optimization',
    'GGA+U Static', 'GGA+U Structure Optimization'
]

optimization_task_ids = {}
for doc in material_ids:
    material_id = str(doc.material_id)
    mp_doc = mpr.materials.search(material_id, fields = ["material_id", "calc_types"])
    #print(mp_doc)
    
    for doc in (mp_doc):
        for task_id, task_type in doc.calc_types.items():
            if task_type in opt_task_types:
                optimization_task_ids[material_id] = task_id
            
                # ThermoDoc: Query MP main entries
                main_entry = mpr.get_entry_by_material_id(material_id=material_id)[0]
                # Query one relaxation task
                task_doc = mpr.materials.tasks.search(task_id, fields=["input", "output", "calcs_reversed", 'task_id', "run_type"])
                #print(  (task_doc[0].calcs_reversed[0].output.ionic_steps[-1])  )
                
                eualCT = calc_type_equal(    task_doc[0],     main_entry,)
                print(eualCT)
                uniq = UniquenessCheck(main_entry.uncorrected_energy_per_atom)
                
                NELM = task_doc[0].calcs_reversed[0].input.incar['NELM']
                #print(NELM)
                for step in task_doc[0].calcs_reversed[0].output.ionic_steps:
                    aa = uniq.is_unique(step, material_id, NELM, mag=None)
                    print(f"CHECK MP ENTRY is MATCHED WITH corresponding TASKIDs - > {aa}")















        
