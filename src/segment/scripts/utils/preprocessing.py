# Utility functions for preprocessing

import SimpleITK as sitk

# File format conversions #

def dicom_to_nifti(dicom_folder_path: str, nifti_path: str) -> None:
    """
    Converts the DICOM slices in `dicom_folder_path` to a single NIfTI file.
    Writes the NIfTI to `nifti_path`.
    """

    assert nifti_path.endswith(".nii.gz"), "Expects a NIfTI file format."

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, nifti_path)
