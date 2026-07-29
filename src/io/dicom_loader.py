"""
============================================================
DICOM Loader
------------------------------------------------------------
Loads a patient's DICOM study and groups images into series.

Responsibilities
----------------
1. Search recursively for DICOM files
2. Read DICOM metadata only
3. Group files by SeriesInstanceUID
4. Identify ADC and DWI series
5. Validate required modalities
============================================================
"""

from pathlib import Path
from collections import defaultdict
import pydicom
from pydicom.errors import InvalidDicomError


class PatientStudy:

    def __init__(self):

        self.patient_id = None
        self.study_uid = None

        self.series = {}

        self.adc = None
        self.dwi = None


# ------------------------------------------------------------
# Read metadata only
# ------------------------------------------------------------

def read_metadata(dicom_path):

    return pydicom.dcmread(
        dicom_path,
        stop_before_pixels=True,
        force=True
    )


# ------------------------------------------------------------
# Find all DICOM files
# ------------------------------------------------------------

def find_dicom_files(patient_folder):

    patient_folder = Path(patient_folder)

    dicom_files = []

    for file in patient_folder.rglob("*"):

        if file.is_file():

            try:
                ds = pydicom.dcmread(
                    file,
                    stop_before_pixels=True,
                    force=True
                )

                if hasattr(ds, "SOPInstanceUID"):
                    dicom_files.append(file)

            except Exception:
                pass

    return sorted(dicom_files)


# ------------------------------------------------------------
# Detect MRI modality
# ------------------------------------------------------------

def detect_modality(ds):

    series_description = str(
        getattr(ds, "SeriesDescription", "")
    ).upper()

    protocol_name = str(
        getattr(ds, "ProtocolName", "")
    ).upper()

    sequence_name = str(
        getattr(ds, "SequenceName", "")
    ).upper()

    image_type = " ".join(
        [str(x).upper() for x in getattr(ds, "ImageType", [])]
    )

    text = " ".join([
        series_description,
        protocol_name,
        sequence_name,
        image_type
    ])

    # ---------------- ADC ----------------

    if "ADC" in text:
        return "ADC"

    # ---------------- DWI ----------------

    if any(keyword in text for keyword in [
        "DWI",
        "DIFF",
        "TRACE",
        "DW",
        "B1000"
    ]):
        return "DWI"

    return "UNKNOWN"

# ------------------------------------------------------------
# Load patient
# ------------------------------------------------------------

def load_patient(patient_folder):

    patient = PatientStudy()

    dicom_files = find_dicom_files(patient_folder)

    if len(dicom_files) == 0:
        raise RuntimeError("No DICOM files found.")

    grouped = defaultdict(list)

    # ---------------------------------------
    # Read metadata
    # ---------------------------------------

    for file in dicom_files:

        ds = read_metadata(file)

        uid = ds.SeriesInstanceUID

        grouped[uid].append({

            "path": file,

            "dataset": ds

        })

    patient.series = grouped

    # ---------------------------------------
    # Patient information
    # ---------------------------------------

    first = grouped[next(iter(grouped))][0]["dataset"]

    patient.patient_id = getattr(first, "PatientID", "UNKNOWN")

    patient.study_uid = getattr(first, "StudyInstanceUID", "")

    # ---------------------------------------
    # Identify series
    # ---------------------------------------

    # A study can contain more than one diffusion/ADC series (for example,
    # localizers or repeated acquisitions).  Prefer the complete series rather
    # than whichever happens to be encountered first in the file-system walk.
    candidates = {"ADC": [], "DWI": []}
    for uid, series in grouped.items():
        modality = detect_modality(series[0]["dataset"])
        if modality in candidates:
            candidates[modality].append(series)

    for modality, series_list in candidates.items():
        if series_list:
            # More slices is the strongest useful signal for the patient-level
            # 3D volume used by the local notebook workflow.
            selected = max(series_list, key=len)
            if modality == "ADC":
                patient.adc = selected
            else:
                patient.dwi = selected

    # ---------------------------------------
    # Validation
    # ---------------------------------------

    if patient.adc is None:
        raise RuntimeError("ADC series not found.")

    if patient.dwi is None:
        raise RuntimeError("DWI series not found.")

    return patient


# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

def print_summary(patient):

    print("=" * 60)

    print("PATIENT SUMMARY")

    print("=" * 60)

    print(f"Patient ID : {patient.patient_id}")

    print(f"Study UID  : {patient.study_uid}")

    print()

    print(f"Total Series : {len(patient.series)}")

    print()

    print(f"ADC Slices : {len(patient.adc)}")

    print(f"DWI Slices : {len(patient.dwi)}")

    print()

    print("✓ Required MRI sequences found.")

    print("=" * 60)
