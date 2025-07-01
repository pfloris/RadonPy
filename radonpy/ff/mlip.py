#  Copyright (c) 2023. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# ff.gaff module
# ******************************************************************************

import numpy as np
import os
import json
from itertools import permutations
from rdkit import Chem
from ..core import calc, utils
from . import ff_class, gaff2_mod

__version__ = '0.2.9'


class MLIP(gaff2_mod.GAFF2_mod):
    """
    mlip.MLIP() class

    Forcefield object with typing rules for MLIP models.
    By default reads data file in forcefields subdirectory.

    Attributes:
        ff_name: mlip_style ('mace', 'pace')
        pair_style: mlip_style ('mace', 'pace')
        bond_style: zero
        angle_style: zero
        dihedral_style: zero
        improper_style: zero
        ff_class: 1
    """
    def __init__(self, mlip_style, mlip_file, db_file=None):
        super().__init__(db_file)
        self.name = mlip_style
        self.mlip_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ff_dat', mlip_file)
        self.mpair_style = mlip_style
        self.mbond_style = 'zero'
        self.mangle_style = 'zero'
        self.mdihedral_style = 'zero'
        self.mimproper_style = 'zero'
        self.ff_class = '1'


    def ff_assign(self, mol, charge=None, retryMDL=True, useMDL=True):
        """
        MLIP.ff_assign

        MLIP force field assignment for RDkit Mol object

        Args:
            mol: rdkit mol object

        Optional args:
            charge: Method of charge assignment. If None, charge assignment is skipped.
            retryMDL: Retry assignment using MDL aromaticity model if default aromaticity model is failure (boolean)
            useMDL: Assignment using MDL aromaticity model (boolean)

        Returns: (boolean)
            True: Success assignment
            False: Failure assignment
        """

        if useMDL:
            Chem.rdmolops.Kekulize(mol, clearAromaticFlags=True)
            Chem.rdmolops.SetAromaticity(mol, model=Chem.rdmolops.AromaticityModel.AROMATICITY_MDL)

        mol.SetProp('ff_name', str(self.name))
        mol.SetProp('ff_class', str(self.ff_class))
        result = self.assign_ptypes(mol)
        if result: result = self.assign_btypes(mol)
        if result: result = self.assign_atypes(mol)
        if result: result = self.assign_dtypes(mol)
        if result: result = self.assign_itypes(mol)
        if result and charge is not None: result = calc.assign_charges(mol, charge=charge)

        if not result and retryMDL and not useMDL:
            utils.radon_print('Retry to assign with MDL aromaticity model', level=1)
            Chem.rdmolops.Kekulize(mol, clearAromaticFlags=True)
            Chem.rdmolops.SetAromaticity(mol, model=Chem.rdmolops.AromaticityModel.AROMATICITY_MDL)

            result = self.assign_ptypes(mol)
            if result: result = self.assign_btypes(mol)
            if result: result = self.assign_atypes(mol)
            if result: result = self.assign_dtypes(mol)
            if result: result = self.assign_itypes(mol)
            if result and charge is not None: result = calc.assign_charges(mol, charge=charge)
            if result: utils.radon_print('Success to assign with MDL aromaticity model', level=1)

        mol.SetProp('mlip_file', str(self.mlip_file))
        mresult = self.massign_ptypes(mol)
        if mresult: mresult = self.massign_btypes(mol)
        if mresult: mresult = self.massign_atypes(mol)
        if mresult: mresult = self.massign_dtypes(mol)
        if mresult: mresult = self.massign_itypes(mol)

        mlip_ptypes = []
        atomic_number = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}
        for p in mol.GetAtoms():
            ptype = '%s' % (p.GetProp('mlip_type'))
            if ptype not in mlip_ptypes:
                mlip_ptypes.append(ptype)
        mlip_ptypes = ' '.join(sorted(mlip_ptypes, key=lambda x: atomic_number[x]))
        mol.SetProp('mlip_ptypes', mlip_ptypes)

        return (result and mresult)


    def massign_ptypes(self, mol):
        """
        MLIP.massign_ptypes

        MLIP particle typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('mpair_style', self.mpair_style)

        for p in mol.GetAtoms():
            pt = p.GetSymbol()
            self.mset_ptype(p, pt)

        return result_flag

    def mset_ptype(self, p, pt):
        p.SetProp('mlip_type', pt)
        p.SetDoubleProp('mlip_epsilon', 0.0)
        p.SetDoubleProp('mlip_sigma', 0.0)

        return p


    def massign_btypes(self, mol):
        """
        MLIP.massign_btypes

        MLIP bond typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('mbond_style', self.mbond_style)

        for b in mol.GetBonds():
            ba = b.GetBeginAtom().GetProp('ff_type')
            bb = b.GetEndAtom().GetProp('ff_type')
            bt = '%s,%s' % (ba, bb)

            self.mset_btype(b, bt)

        return result_flag


    def mset_btype(self, b, bt):
        b.SetProp('mlip_type', bt)
        b.SetDoubleProp('mlip_k', 0.0)
        b.SetDoubleProp('mlip_r0', 0.0)

        return True


    def massign_atypes(self, mol):
        """
        MLIP.massign_atypes

        MLIP angle typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('mangle_style', self.mangle_style)
        setattr(mol, 'mangles', [])

        for p in mol.GetAtoms():
            for p1 in p.GetNeighbors():
                for p2 in p.GetNeighbors():
                    if p1.GetIdx() == p2.GetIdx(): continue
                    unique = True
                    atoms = [p1, p, p2]
                    for ang in mol.angles:
                        if ((ang.a == p1.GetIdx() and ang.b == p.GetIdx() and ang.c == p2.GetIdx()) or
                                (ang.c == p1.GetIdx() and ang.b == p.GetIdx() and ang.a == p2.GetIdx())):
                            unique = False
                    if unique:
                        pt1 = p1.GetProp('ff_type')
                        pt = p.GetProp('ff_type')
                        pt2 = p2.GetProp('ff_type')
                        at = '%s,%s,%s' % (pt1, pt, pt2)

                        self.mset_atype(mol, a=p1.GetIdx(), b=p.GetIdx(), c=p2.GetIdx(), at=at)

        return result_flag


    def mset_atype(self, mol, a, b, c, at):

        angle = utils.Angle(
            a=a, b=b, c=c,
            ff=ff_class.Angle_harmonic(
                ff_type=at,
                k=0.0,
                theta0=0.0
            )
        )

        mol.angles.append(angle)

        return True


    def massign_dtypes(self, mol):
        """
        MLIP.massign_dtypes

        MLIP specific dihedral typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('mdihedral_style', self.mdihedral_style)
        setattr(mol, 'mdihedrals', [])

        for b in mol.GetBonds():
            p1 = b.GetBeginAtom()
            p2 = b.GetEndAtom()
            for p1b in p1.GetNeighbors():
                for p2b in p2.GetNeighbors():
                    if p1.GetIdx() == p2b.GetIdx() or p2.GetIdx() == p1b.GetIdx() or p1b.GetIdx() == p2b.GetIdx(): continue
                    unique = True
                    atoms = [p1b, p1, p2, p2b]
                    for dih in mol.dihedrals:
                        if ((dih.a == p1b.GetIdx() and dih.b == p1.GetIdx() and
                             dih.c == p2.GetIdx() and dih.d == p2b.GetIdx()) or
                                (dih.d == p1b.GetIdx() and dih.c == p1.GetIdx() and
                                 dih.b == p2.GetIdx() and dih.a == p2b.GetIdx())):
                            unique = False
                    if unique:
                        p1bt = p1b.GetProp('ff_type')
                        p1t = p1.GetProp('ff_type')
                        p2t = p2.GetProp('ff_type')
                        p2bt = p2b.GetProp('ff_type')
                        dt = '%s,%s,%s,%s' % (p1bt, p1t, p2t, p2bt)

                        self.mset_dtype(mol, a=p1b.GetIdx(), b=p1.GetIdx(), c=p2.GetIdx(), d=p2b.GetIdx(),
                                                dt=dt)

        return result_flag


    def mset_dtype(self, mol, a, b, c, d, dt):

        dihedral = utils.Dihedral(
            a=a, b=b, c=c, d=d,
            ff=ff_class.Dihedral_fourier(
                ff_type=dt,
                k=[0.0],
                d0=[0.0],
                m=1,
                n=[0]
            )
        )

        mol.dihedrals.append(dihedral)

        return True

    def massign_itypes(self, mol):
        """
        MLIP.massign_itypes

        MLIP specific improper typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        mol.SetProp('mimproper_style', self.mimproper_style)
        setattr(mol, 'mimpropers', [])

        for p in mol.GetAtoms():
            if len(p.GetNeighbors()) == 3:
                for perm in permutations(p.GetNeighbors(), 3):
                    pt = p.GetProp('ff_type')
                    p1t = perm[0].GetProp('ff_type')
                    p2t = perm[1].GetProp('ff_type')
                    p3t = perm[2].GetProp('ff_type')
                    it = '%s,%s,%s,%s' % (pt, p1t, p2t, p3t)

                    self.mset_itype(mol, a=p.GetIdx(), b=perm[0].GetIdx(), c=perm[1].GetIdx(),
                                   d=perm[2].GetIdx(), it=it)


        return True

    def mset_itype(self, mol, a, b, c, d, it):

        improper = utils.Improper(
            a=a, b=b, c=c, d=d,
            ff=ff_class.Improper_zero(
                ff_type=it,
                k=0.0,
                d0=1,
                n=0
            )
        )

        mol.impropers.append(improper)

        return True