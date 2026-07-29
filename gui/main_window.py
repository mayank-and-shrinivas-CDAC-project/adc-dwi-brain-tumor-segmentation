from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QProgressBar,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMessageBox,
)
from PySide6.QtWidgets import QComboBox
import torch
import numpy as np
from src.preprocessing.preprocess import preprocess
from src.preprocessing.brain_extraction import extract_brain_mask
from src.io.dicom_loader import load_patient
from PySide6.QtWidgets import QApplication

from src.reconstruction.volume_builder import build_volume

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSlider
from src.registration.register import register_volumes
from src.inference.predict import predict
from src.models import available_models, checkpoint_path, load_model
from src.analysis.report import (analyze_prediction,create_report, save_report,)
from pathlib import Path

from src.analysis.report import (
    analyze_prediction,
    create_report,
    save_report,
)

from src.analysis.save_prediction import save_prediction
from src.analysis.save_overlay import save_overlay
from src.analysis.save_images import save_middle_slice
from src.analysis.render3d import render_prediction
from src.analysis.pdf_report import create_pdf


MODEL_OPTIONS = available_models()

class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Brain Tumor Analysis System")

        self.resize(1400, 850)

        self.patient_folder = ""
        self.patient = None
        self.adc_volume = None
        self.dwi_volume = None
        self.registered_adc = None
        self.registered_dwi = None
        self.brain_mask = None

        self.preprocessed_adc = None
        self.preprocessed_dwi = None
        self.preprocessed_brain_mask = None
        self.input_tensor = None

        self.prediction = None
        self.model = None
        self.build_ui()

    # --------------------------------------------------------

    def build_ui(self):

        central = QWidget()

        self.setCentralWidget(central)

        root = QHBoxLayout(central)

        # =====================================================
        # LEFT SIDEBAR
        # =====================================================

        self.sidebar = QListWidget()
        

        self.sidebar.setFixedWidth(220)

        steps = [

            "⬜ Load Patient",
            "⬜ Reconstruction",
            "⬜ Registration",
            "⬜ HD-BET Brain Extraction",
            "⬜ Preprocessing",
            "⬜ Segmentation",
            "⬜ Report"

        ]

        for s in steps:

            self.sidebar.addItem(QListWidgetItem(s))

        root.addWidget(self.sidebar)

        # =====================================================
        # CENTER
        # =====================================================

        center = QVBoxLayout()

        title = QLabel("Brain Tumor Analysis Using ADC and DWI MRI")

        title.setAlignment(Qt.AlignCenter)

        title.setStyleSheet("""
            font-size:24px;
            font-weight:bold;
        """)

        center.addWidget(title)

        # -----------------------------

        row = QHBoxLayout()

        self.folder = QLineEdit()

        self.folder.setPlaceholderText(
            "Select Patient Folder..."
        )

        browse = QPushButton("Browse")

        browse.clicked.connect(self.browse)

        row.addWidget(self.folder)

        row.addWidget(browse)

        center.addLayout(row)

        # -----------------------------

        # ---------------------------------------------------------
        # Buttons
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # Pipeline Buttons
        # ---------------------------------------------------------

        self.load_button = QPushButton("Load Patient")
        self.load_button.setMinimumHeight(45)
        self.load_button.clicked.connect(self.load_patient)

        self.reconstruct_button = QPushButton("Reconstruct")
        self.reconstruct_button.setMinimumHeight(45)
        self.reconstruct_button.setEnabled(False)
        self.reconstruct_button.clicked.connect(self.reconstruct)

        self.register_button = QPushButton("Register")
        self.register_button.setMinimumHeight(45)
        self.register_button.setEnabled(False)
        self.register_button.clicked.connect(self.register_patient)

        self.preprocess_button = QPushButton("Preprocess")
        self.preprocess_button.setMinimumHeight(45)
        self.preprocess_button.setEnabled(False)
        self.preprocess_button.clicked.connect(self.preprocess_patient)

        self.hd_bet_button = QPushButton("Run HD-BET Brain Extraction")
        self.hd_bet_button.setMinimumHeight(45)
        self.hd_bet_button.setEnabled(False)
        self.hd_bet_button.clicked.connect(self.extract_brain)

        # Model Selection
        self.model_box = QComboBox()
        self.model_box.addItems(MODEL_OPTIONS)
        self.model_box.setEnabled(True)

        # Prediction
        self.predict_button = QPushButton("Predict Tumor")
        self.predict_button.setMinimumHeight(45)
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(self.predict_patient)

        # Report
        self.report_button = QPushButton("Generate Report")
        self.report_button.setMinimumHeight(45)
        self.report_button.setEnabled(False)
        self.report_button.clicked.connect(self.generate_report)

        # ---------------------------------------------------------
        # Add widgets in pipeline order
        # ---------------------------------------------------------

        center.addWidget(self.load_button)
        center.addWidget(self.reconstruct_button)
        center.addWidget(self.register_button)
        center.addWidget(self.hd_bet_button)
        center.addWidget(self.preprocess_button)
        center.addWidget(self.model_box)
        center.addWidget(self.predict_button)
        center.addWidget(self.report_button)
        # ---------------------------------------------------------
        # Image Viewer
        # ---------------------------------------------------------

        viewer = QFrame()
        viewer.setFrameShape(QFrame.Box)

        viewer_layout = QVBoxLayout(viewer)

        self.figure = Figure(figsize=(8,4))

        self.canvas = FigureCanvas(self.figure)

        viewer_layout.addWidget(self.canvas)

        # Slider

        self.slider = QSlider(Qt.Horizontal)

        self.slider.setEnabled(False)

        self.slider.valueChanged.connect(
            self.update_slice
        )

        viewer_layout.addWidget(self.slider)

        center.addWidget(viewer)

        root.addLayout(center, 3)

        # =====================================================
        # RIGHT PANEL
        # =====================================================

        right = QVBoxLayout()

        patient = QLabel("Patient Information")

        patient.setStyleSheet("""
            font-size:18px;
            font-weight:bold;
        """)

        right.addWidget(patient)

        self.info = QTextEdit()

        self.info.setReadOnly(True)

        right.addWidget(self.info)

        status = QLabel("Status")

        status.setStyleSheet("""
            font-size:18px;
            font-weight:bold;
        """)

        right.addWidget(status)

        self.log = QTextEdit()

        self.log.setReadOnly(True)

        right.addWidget(self.log)

        self.progress = QProgressBar()

        self.progress.setValue(0)

        right.addWidget(self.progress)

        root.addLayout(right, 1)

        self.write_log("Application Started")

    def complete_step(self, index):

        item = self.sidebar.item(index)

        item.setText(
            item.text().replace("⬜", "✅")
        )
    # --------------------------------------------------------

    def write_log(self, text):

        self.log.append(text)

    # --------------------------------------------------------

    def browse(self):

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Patient Folder"
        )

        if folder:

            self.patient_folder = folder

            self.folder.setText(folder)

            self.write_log("Patient folder selected.")

    # --------------------------------------------------------

    def load_patient(self):

        if not self.patient_folder:

            QMessageBox.warning(
                self,
                "Warning",
                "Please select a patient folder."
            )

            return

        try:

            self.progress.setValue(10)

            self.write_log("Searching DICOM files...")

            patient = load_patient(self.patient_folder)
            QApplication.processEvents()
            self.patient = patient

            self.progress.setValue(80)

            self.info.clear()

            self.info.append(
                f"Patient ID : {patient.patient_id}"
            )

            self.info.append(
                f"Study UID : {patient.study_uid}"
            )

            self.info.append("")

            self.info.append(
                f"Series Found : {len(patient.series)}"
            )

            self.info.append("")

            self.info.append(
                f"ADC : {len(patient.adc)} slices"
            )

            self.info.append(
                f"DWI : {len(patient.dwi)} slices"
            )

            self.progress.setValue(100)

            self.complete_step(0)
            self.reconstruct_button.setEnabled(True)
            self.sidebar.setCurrentRow(1)

            self.write_log("Patient loaded successfully.")

        except Exception as e:

            QMessageBox.critical(
                self,
                "Error",
                str(e)
            )

            self.progress.setValue(0)
    
    def reconstruct(self):

        self.write_log("Reconstructing ADC volume...")

        self.progress.setValue(20)

        self.adc_volume = build_volume(self.patient.adc)

        self.write_log("Reconstructing DWI volume...")

        self.progress.setValue(60)

        self.dwi_volume = build_volume(self.patient.dwi)

        self.progress.setValue(100)

        self.complete_step(1)

        self.info.append("")
        self.info.append("Volumes")
        self.info.append("----------------")

        self.info.append(f"ADC : {self.adc_volume.shape}")
        self.info.append(f"DWI : {self.dwi_volume.shape}")

        self.write_log("Reconstruction complete.")

        # Enable viewer
        self.slider.setEnabled(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.adc_volume.shape[0] - 1)

        middle = self.adc_volume.shape[0] // 2
        self.slider.setValue(middle)

        # Enable registration
        self.register_button.setEnabled(True)

        self.write_log("Registration is now enabled.")

        # Display middle slice
        self.update_slice()

    def update_slice(self):

        if self.adc_volume is None:
            return

        import cv2

        # -------------------------------------------------------
        # Use registered volumes if available
        # -------------------------------------------------------

        if self.registered_adc is not None:

            adc_volume = self.registered_adc
            dwi_volume = self.registered_dwi
            dwi_title = "Registered DWI"

        else:

            adc_volume = self.adc_volume
            dwi_volume = self.dwi_volume
            dwi_title = "DWI"

        # -------------------------------------------------------
        # Current slice
        # -------------------------------------------------------

        index = self.slider.value()

        adc = adc_volume[index]

        if dwi_volume.shape[0] == adc_volume.shape[0]:

            dwi = dwi_volume[index]

        else:

            scale = dwi_volume.shape[0] / adc_volume.shape[0]

            dwi = dwi_volume[int(index * scale)]

        # DWI source series can contain isolated zero-valued pixels.  After
        # registration these become especially conspicuous as black marks in
        # the viewer.  Repair only small zero islands for display; the stored
        # registered DWI remains unchanged for preprocessing and inference.
        dwi_display = dwi
        if self.registered_dwi is not None:
            zero_mask = (dwi <= 0).astype("uint8")
            _, labels, stats, _ = cv2.connectedComponentsWithStats(
                zero_mask, connectivity=8
            )
            small_zero_islands = np.zeros_like(zero_mask)
            for label in range(1, len(stats)):
                area = stats[label, cv2.CC_STAT_AREA]
                if area <= 128:
                    small_zero_islands[labels == label] = 255

            if np.any(small_zero_islands):
                dwi_display = cv2.inpaint(
                    dwi.astype(np.float32),
                    small_zero_islands,
                    3,
                    cv2.INPAINT_TELEA,
                )

        # -------------------------------------------------------
        # Display
        # -------------------------------------------------------

        self.figure.clear()

        if self.prediction is None:

            ax1 = self.figure.add_subplot(121)
            ax2 = self.figure.add_subplot(122)

        else:

            ax1 = self.figure.add_subplot(131)
            ax2 = self.figure.add_subplot(132)
            ax3 = self.figure.add_subplot(133)

        # -------------------------------------------------------
        # ADC
        # -------------------------------------------------------

        ax1.imshow(adc, cmap="gray")
        ax1.set_title("ADC")
        ax1.axis("off")

        # -------------------------------------------------------
        # DWI
        # -------------------------------------------------------

        ax2.imshow(dwi_display, cmap="gray")
        ax2.set_title(dwi_title)
        ax2.axis("off")

        # -------------------------------------------------------
        # Prediction Overlay
        # -------------------------------------------------------

        if self.prediction is not None:

            adc_depth = adc_volume.shape[0]
            pred_depth = self.prediction.shape[2]

            pred_index = int(index * pred_depth / adc_depth)
            pred_index = min(pred_index, pred_depth - 1)

            # Prediction is (H,W,D)
            mask = self.prediction[:, :, pred_index]

            # Resize mask to match displayed ADC slice
            mask = cv2.resize(
                mask.astype("uint8"),
                (adc.shape[1], adc.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            ax3.imshow(adc, cmap="gray")

            from matplotlib.colors import ListedColormap

            colors = ListedColormap(["#00000000", "#00c8ff", "#ff6b00", "#e53935"])
            ax3.imshow(
                mask,
                cmap=colors,
                alpha=(mask > 0) * 0.7,
                vmin=0,
                vmax=3,
            )

            ax3.set_title("Segmentation")
            ax3.axis("off")

            from matplotlib.patches import Patch

            self.figure.legend(
                handles=[
                    Patch(color="#00c8ff", label="Edema"),
                    Patch(color="#ff6b00", label="Core"),
                    Patch(color="#e53935", label="Enhancing"),
                ],
                loc="lower center",
                ncol=3,
                frameon=False,
                fontsize=8,
            )

        self.figure.subplots_adjust(wspace=0.04, bottom=0.10, top=0.93)
        self.canvas.draw()
    
    def register_patient(self):

        if self.adc_volume is None or self.dwi_volume is None:

            QMessageBox.warning(
                self,
                "Warning",
                "Please reconstruct the volumes first."
            )

            return

        self.write_log("Starting registration...")

        self.progress.setValue(10)

        try:

            self.registered_adc, self.registered_dwi = register_volumes(
                self.adc_volume,
                self.dwi_volume
            )
            self.brain_mask = None

            self.progress.setValue(100)

            self.complete_step(2)

            self.write_log("Registration completed successfully.")

            self.info.append("")
            self.info.append("Registration")
            self.info.append("----------------------")
            self.info.append(
                f"Registered ADC : {self.registered_adc.shape}"
            )
            self.info.append(
                f"Registered DWI : {self.registered_dwi.shape}"
            )

            self.hd_bet_button.setEnabled(True)
            self.write_log("HD-BET brain extraction is now enabled.")

            self.update_slice()

        except Exception as e:

            QMessageBox.critical(
                self,
                "Registration Error",
                str(e)
            )

            self.progress.setValue(0)

            self.write_log("Registration failed.")

    def extract_brain(self):

        if self.registered_adc is None:

            QMessageBox.warning(
                self,
                "Warning",
                "Please perform registration first."
            )

            return

        self.write_log("Running HD-BET brain extraction...")
        self.progress.setValue(20)
        QApplication.processEvents()

        try:

            self.brain_mask, method = extract_brain_mask(self.registered_adc)

            if not self.brain_mask.any():
                raise RuntimeError("Brain extraction returned an empty mask.")

            self.progress.setValue(100)
            self.complete_step(3)

            self.info.append("")
            self.info.append("Brain Extraction")
            self.info.append("----------------------")
            self.info.append(f"Method : {method}")
            self.info.append(f"Mask shape : {self.brain_mask.shape}")

            self.preprocess_button.setEnabled(True)
            self.write_log(f"Brain extraction completed using {method}.")
            self.write_log("Preprocessing is now enabled.")

        except Exception as e:

            QMessageBox.critical(
                self,
                "Brain Extraction Error",
                str(e)
            )

            self.progress.setValue(0)
            self.write_log("Brain extraction failed.")
    
    def preprocess_patient(self):

        if self.brain_mask is None:

            QMessageBox.warning(
                self,
                "Warning",
                "Please run HD-BET brain extraction first."
            )

            return

        self.write_log("Starting preprocessing...")

        self.progress.setValue(10)

        self.preprocessed_adc, \
        self.preprocessed_dwi, \
        self.input_tensor, \
        self.preprocessed_brain_mask = preprocess(
            self.registered_adc,
            self.registered_dwi,
            self.brain_mask,
        )

        self.progress.setValue(100)

        self.complete_step(4)

        self.info.append("")
        self.info.append("Preprocessing")
        self.info.append("----------------")

        self.info.append(
            f"ADC : {self.preprocessed_adc.shape}"
        )

        self.info.append(
            f"DWI : {self.preprocessed_dwi.shape}"
        )

        self.info.append(
            f"Tensor : {tuple(self.input_tensor.shape)}"
        )

        self.write_log(
            "Preprocessing completed."
        )
        self.predict_button.setEnabled(True)
        self.write_log(
            "Prediction is now enabled."
        )
    
    def predict_patient(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = self.model_box.currentText()
        try:
            weight_path = checkpoint_path(model_name)
            self.model = load_model(model_name, device)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            QMessageBox.critical(self, "Model Loading Error", str(error))
            self.write_log(f"Model loading failed: {error}")
            return
        self.write_log(f"Loaded {model_name} from {weight_path.name}.")
        
        self.prediction = predict(
            self.model,
            self.input_tensor,
            device,
            self.preprocessed_brain_mask,
        )

        patient = self.patient.patient_id

        result_dir = Path("results") / patient
        result_dir.mkdir(parents=True, exist_ok=True)

        # Save prediction
        save_prediction(
            self.prediction,
            patient,
        )

        # Save middle slices
        save_middle_slice(
            self.preprocessed_adc,
            result_dir / "adc_middle.png",
        )

        save_middle_slice(
            self.preprocessed_dwi,
            result_dir / "dwi_middle.png",
        )

        # Save overlay
        save_overlay(
            self.preprocessed_adc,
            self.prediction,
            patient,
        )

        # Generate statistics
        stats = analyze_prediction(
            self.prediction,
            self.preprocessed_brain_mask,
        )

        # Create report text
        report_text = create_report(
            patient,
            self.model_box.currentText(),
            stats,
        )

        # Save txt report
        report_path = save_report(
            report_text,
            patient,
        )

        # Generate PDF
        create_pdf(
            patient,
            report_text,
        )

        # Render 3D
        render_prediction(
            self.prediction,
            patient,
        )

        self.write_log("Prediction completed.")
        self.complete_step(5)
        self.sidebar.setCurrentRow(6)   # Move selection to Report
        self.report_button.setEnabled(True)
        self.write_log(f"Results saved to {result_dir}")

        # Show prediction
        self.update_slice()

        self.report_button.setEnabled(True)
    
    def generate_report(self):

        if self.prediction is None:

            QMessageBox.warning(
                self,
                "Warning",
                "Run prediction first."
            )

            return

        patient = self.patient.patient_id

        model = self.model_box.currentText()

        stats = analyze_prediction(
            self.prediction,
            self.preprocessed_brain_mask,
        )

        text = create_report(
            patient,
            model,
            stats
        )

        filename = save_report(
            text,
            patient
        )

        self.info.append("")
        self.info.append(text)

        QMessageBox.information(
            self,
            "Report",
            f"Saved to\n{filename}"
        )
        self.complete_step(6)
