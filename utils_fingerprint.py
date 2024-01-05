import numpy as np
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, DataStructs, MACCSkeys


def mol_to_Avalon(mol_list, fpszie):
    if isinstance(mol_list, list):
        print("mol_list is list")
        fp_bit_list = []
        fp_list = []
        for mol in mol_list:
            try:
                fp = pyAvalonTools.GetAvalonFP(mol, nBits=fpszie)
                fp_bit = np.zeros(len(fp))
                DataStructs.ConvertToNumpyArray(fp, fp_bit)
            except ValueError as e:
                print(e)
                fp = [np.nan]
                fp_bit = [np.nan]
            fp_bit_list.append(fp_bit)
            fp_list.append(fp)
    else:
        try:
            fp = pyAvalonTools.GetAvalonFP(mol_list, nBits=fpszie)
            fp_bit_list = np.zeros(len(fp))
            DataStructs.ConvertToNumpyArray(fp, fp_bit_list)
            fp_list = fp
        except ValueError as e:
            print(e)
            fp_list = [np.nan]
            fp_bit_list = [np.nan]
    return fp_list, fp_bit_list


def mol_to_ecfp4(mol_list, fpszie):
    if isinstance(mol_list, list):
        print("mol_list is list")
        fp_bit_list = []
        fp_list = []
        for mol in mol_list:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fpszie)
                fp_bit = np.zeros(len(fp))
                DataStructs.ConvertToNumpyArray(fp, fp_bit)
            except ValueError as e:
                print(e)
                fp = [np.nan]
                fp_bit = [np.nan]
            fp_bit_list.append(fp_bit)
            fp_list.append(fp)
    else:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol_list, 2, nBits=fpszie)
            fp_bit_list = np.zeros(len(fp))
            DataStructs.ConvertToNumpyArray(fp, fp_bit_list)
            fp_list = fp
        except ValueError as e:
            print(e)
            fp_list = [np.nan]
            fp_bit_list = [np.nan]
    return fp_list, fp_bit_list


def mol_to_fcfp4(mol_list, fpszie):
    if isinstance(mol_list, list):
        print("mol_list is list")
        fp_bit_list = []
        fp_list = []
        for mol in mol_list:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fpszie, useFeatures=True)
                fp_bit = np.zeros(len(fp))
                DataStructs.ConvertToNumpyArray(fp, fp_bit)
            except ValueError as e:
                print(e)
                fp = [np.nan]
                fp_bit = [np.nan]
            fp_bit_list.append(fp_bit)
            fp_list.append(fp)
    else:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol_list, 2, nBits=fpszie, useFeatures=True)
            fp_bit_list = np.zeros(len(fp))
            DataStructs.ConvertToNumpyArray(fp, fp_bit_list)
            fp_list = fp
        except ValueError as e:
            print(e)
            fp_list = [np.nan]
            fp_bit_list = [np.nan]
    return fp_list, fp_bit_list


def mol_to_maccs(mol_list):
    if isinstance(mol_list, list):
        print("mol_list is list")
        fp_bit_list = []
        fp_list = []
        for mol in mol_list:
            try:
                fp = MACCSkeys.GenMACCSKeys(mol)
                fp_bit = np.zeros(len(fp))
                DataStructs.ConvertToNumpyArray(fp, fp_bit)
            except ValueError as e:
                print(e)
                fp = [np.nan]
                fp_bit = [np.nan]
            fp_list.append(fp)
            fp_bit_list.append(fp_bit)
    else:
        try:
            fp = MACCSkeys.GenMACCSKeys(mol_list)
            fingerprint = np.zeros(len(fp))
            DataStructs.ConvertToNumpyArray(fp, fingerprint)
            fp_list = fp
            fp_bit_list = fingerprint
        except ValueError as e:
            print(e)
            fp_list = [np.nan]
            fp_bit_list = [np.nan]
    return fp_list, fp_bit_list


def mol_to_fp(mol_list, fpszie):  # 与rdkit分子描述符相同
    if isinstance(mol_list, list):
        print("mol_list is list")
        fp_bit_list = []
        fp_list = []
        for mol in mol_list:
            try:
                fp = Chem.RDKFingerprint(mol, fpSize=fpszie)
                fp_bit = np.zeros(len(fp))
                DataStructs.ConvertToNumpyArray(fp, fp_bit)
            except ValueError as e:
                print(e)
                fp = [np.nan]
                fp_bit = [np.nan]
            fp_list.append(fp)
            fp_bit_list.append(fp_bit)

    else:
        try:
            fp = Chem.RDKFingerprint(mol_list, fpSize=fpszie)
            fp_bit_list = np.zeros(len(fp))
            DataStructs.ConvertToNumpyArray(fp, fp_bit_list)
            fp_list = fp
        except ValueError as e:
            print(e)
            fp_list = [np.nan]
            fp_bit_list = [np.nan]
    return fp_list, fp_bit_list
