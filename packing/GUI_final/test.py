import warnings
warnings.filterwarnings('ignore')

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from multiprocessing import Process, Queue, freeze_support
import subprocess
import os
import Hyundai_rc_rc
import Lab_rc_black_rc
import inference
import inference_CAD

from pathlib import Path
import sys
from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QMessageBox

from PyQt5 import QtWidgets, QtGui, QtCore
import os

class LoadingDialog(QtWidgets.QDialog):
    def __init__(self, message="예측 중입니다...", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Running...")
        self.setFixedSize(300, 150)
        self.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint)
        self.setModal(True)
        self.setStyleSheet("background-color: white;")  # 🔸 배경색 명시적으로 설정

        self.cancelled = False  # ✅ 중단 여부 플래그

        layout = QtWidgets.QVBoxLayout(self)

        self.label = QtWidgets.QLabel(message)
        font = QtGui.QFont("Arial", 11)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        self.spinner = QtWidgets.QLabel()
        self.spinner.setAlignment(QtCore.Qt.AlignCenter)
        self.spinner.setStyleSheet("background-color: transparent;")  # 🔸 투명 처리

        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(__file__)
        gif_path = os.path.join(base_path, "spinner.gif")

        self.movie = QtGui.QMovie(gif_path)  # 반드시 self.movie로 유지
        if self.movie.isValid():
            self.movie.setScaledSize(QtCore.QSize(64, 64))
            self.movie.setParent(self)  # 명시적으로 부모 지정
            self.spinner.setMovie(self.movie)
            self.movie.start()
        else:
            # print("❌ 유효하지 않은 GIF입니다:", gif_path)
            self.spinner.setText("GIF 로딩 실패")

        layout.addWidget(self.spinner)
        layout.addWidget(self.label)

    def closeEvent(self, event):
        if self.movie.state() == QtGui.QMovie.Running:
            self.movie.stop()
        self.cancelled = True 
        event.accept()


class Worker(Process):
    def __init__(self, func, *args):
        super(Worker, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)

class InfoDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Information")
        self.setFixedSize(450, 150)
        self.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.WindowCloseButtonHint)
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        info_label = QtWidgets.QLabel()
        info_label.setText("""
        <b>Implemented  by</b> Computational Mechanics and Design Optimization Lab<br>
        <br>
        <b>Supervised   by</b> Gunwoo Noh (<i>gunwoonoh@korea.ac.kr</i>)<br>
        <br>
        <b>Contributed  by</b> Sanghyun Cho, Chanju Lee, Cheolgi Lyoo, Dongwoo Suh<br>
        <br>
        <b>Supported    by</b> Jihun Kim<br>
        """)
        info_label.setAlignment(QtCore.Qt.AlignLeft)
        info_label.setWordWrap(True)
        font = QtGui.QFont("Arial", 10)
        info_label.setFont(font)
        layout.addWidget(info_label)

class ExtrapolationInputDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input Extrapolation Range")
        self.setFixedSize(400, 250)

        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        layout = QtWidgets.QVBoxLayout()

        self.inputs = []
        for i in range(5):
            h_layout = QtWidgets.QHBoxLayout()
            if i in [0, 1, 2]:
                label = QtWidgets.QLabel(f"d{i+1} (mm):    ")
            else:
                label = QtWidgets.QLabel(f"a{i-2} (degree):")
            line_edit = QtWidgets.QLineEdit()
            h_layout.addWidget(label)
            h_layout.addWidget(line_edit)
            layout.addLayout(h_layout)
            self.inputs.append(line_edit)

        self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def get_values(self):
        return [inp.text() for inp in self.inputs]


class BasePredictionThread(QtCore.QThread):
    error_occurred = pyqtSignal(str)  # 에러 메시지 전달용

    def __init__(self):
        super().__init__()
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True

    def should_stop(self):
        return self._is_stopped

class ParameterPredictionThread(BasePredictionThread):
    def __init__(self, csv_path, linear_scaling, extrapolation_values=None):
        super().__init__()
        self.csv_path = csv_path
        self.linear_scaling = linear_scaling
        self.extrapolation_values = extrapolation_values

    def run(self):
        try:
            inference.main(self.csv_path, linear_scaling=self.linear_scaling, extrapolation_values=self.extrapolation_values)
        except Exception as e:
            self.error_occurred.emit(str(e))  # 에러 발생 시 시그널 송출

class CADPredictionThread(BasePredictionThread):
    def __init__(self, csv_path, step_path, linear_scaling, extrapolation_values=None):
        super().__init__()
        self.csv_path = csv_path
        self.step_path = step_path
        self.linear_scaling = linear_scaling
        self.extrapolation_values = extrapolation_values

    def run(self):
        try:
            inference_CAD.main(self.csv_path, self.step_path, linear_scaling=self.linear_scaling, extrapolation_values=self.extrapolation_values)
        except Exception as e:
            self.error_occurred.emit(str(e))  # 에러 발생 시 시그널 송출


class CATPartConversionThread(BasePredictionThread):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True

    def run(self):
        try:
            import CATPart2step_total_RUBBER as catpart_converter
            catpart_converter.convert_catpart_to_step(self.folder_path)
        except Exception as e:
            self.error_occurred.emit(str(e))  # 에러 발생 시 시그널 송출


class Ui_MainWindow(object):
    csv_folder_path = ''
    cad_folder_path = ''

    def show_success_message(self, title="완료", message="작업이 완료되었습니다."):
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle(title)
        dialog.setFixedSize(300, 150)  # 로딩 창과 동일한 크기

        dialog.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint)
        dialog.setModal(True)
        dialog.setStyleSheet("background-color: white;")

        layout = QtWidgets.QVBoxLayout(dialog)

        # 체크 이미지 경로 (아이콘은 .ico 또는 .png 형식)
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(__file__)
        icon_path = os.path.join(base_path, "check.png")  # 예: check_icon.png 파일이 루트에 있어야 함
        
        icon_label = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(icon_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon_label.setPixmap(pixmap)
            icon_label.setAlignment(QtCore.Qt.AlignCenter)
        else:
            icon_label.setText("✔️")  # fallback 텍스트

        label = QtWidgets.QLabel(message)
        font = QtGui.QFont("Arial", 11)
        label.setFont(font)
        label.setAlignment(QtCore.Qt.AlignCenter)

        ok_button = QtWidgets.QPushButton("확인")
        ok_button.setFixedWidth(100)
        ok_button.clicked.connect(dialog.accept)

        layout.addWidget(icon_label)
        layout.addWidget(label)
        layout.addWidget(ok_button, alignment=QtCore.Qt.AlignCenter)

        dialog.exec_()

    def show_error_message_box(self, message):
        msg_box = QtWidgets.QMessageBox()
        msg_box.setIcon(QtWidgets.QMessageBox.Critical)
        msg_box.setWindowTitle("오류 발생")
        msg_box.setText("예측 실행 중 오류가 발생했습니다:\n\n" + message)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg_box.exec_()  # ❗ 유저가 끌 때까지 유지
        
    def show_error_message_box_module3(self, message):
        msg_box = QtWidgets.QMessageBox()
        msg_box.setIcon(QtWidgets.QMessageBox.Critical)
        msg_box.setWindowTitle("오류 발생")
        msg_box.setText("변환 실행 중 오류가 발생했습니다:\n\n" + message)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg_box.exec_()  # ❗ 유저가 끌 때까지 유지

    def handle_extrapolation_checkbox(self, checked):
        if checked:
            dialog = ExtrapolationInputDialog()
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                self.extrapolation_values = dialog.get_values()
                # print("입력된 기하최대범위 값:", self.extrapolation_values)
            else:
                self.checkBox_extrapolation.setChecked(False)  # 취소 시 상태 원복
        else:
            self.extrapolation_values = None
            # print("Extrapolation 옵션 해제됨")
    

    def _handle_dialog_closed(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            if hasattr(self.thread, 'stop'):
                self.thread.stop()
            self.thread.terminate()
            # print("🛑 스레드 강제 종료됨")

    def run_program3(self):
        current_tab_index = self.tabs.currentIndex()
        self.dialog = LoadingDialog("2차원 강성을 예측 중입니다...")
        self.dialog.finished.connect(self._handle_dialog_closed)

        # --------------------------- 0. Parameter Input AI Model --------------------------
        if current_tab_index == 0:  # Parameter AI
            csv_path = self.lineEdit_1.text().strip()
            if not csv_path:
                print("CSV 경로가 비어 있습니다.")
                return

            linear_scaling = self.checkBox_linear.isChecked()
            extrapolation_values = getattr(self, 'extrapolation_values', None)
            self.thread = ParameterPredictionThread(csv_path, linear_scaling, extrapolation_values)
            self.thread.error_occurred.connect(self.show_error_message_box)
            success_message = "예측이 성공적으로 완료되었습니다."

        # --------------------------- 1. CAD Input AI Model --------------------------
        elif current_tab_index == 1:  # CAD Input
            csv_path = self.lineEdit_1.text().strip()
            step_path = self.lineEdit_2.text().strip()
            if not csv_path or not step_path:
                print("CSV 또는 STEP 경로 누락")
                return

            linear_scaling = self.checkBox_linear.isChecked()
            extrapolation_values = getattr(self, 'extrapolation_values', None)
            self.thread = CADPredictionThread(csv_path, step_path, linear_scaling, extrapolation_values)
            self.thread.error_occurred.connect(self.show_error_message_box)
            success_message = "예측이 성공적으로 완료되었습니다."

        # --------------------------- 2. CATPart to Step Converter --------------------------
        elif current_tab_index == 2:
            folder_path = self.lineEdit_3.text().strip()
            if not folder_path:
                print("CATPart 폴더 경로가 비어 있습니다.")
                return

            self.dialog = LoadingDialog("CATPart를 step으로 변환 중입니다...")
            self.thread = CATPartConversionThread(folder_path)
            self.thread.error_occurred.connect(self.show_error_message_box_module3)
            success_message = "변환이 성공적으로 완료되었습니다."

        self.thread.finished.connect(self.dialog.close)
        self.dialog.finished.connect(self._handle_dialog_closed)
        self.thread.error_occurred.connect(lambda _: setattr(self, "_had_error", True))
        self._had_error = False

        def maybe_show_success():
            if not self._had_error:
                self.show_success_message("Complete!", success_message)

        self.thread.finished.connect(maybe_show_success)

        self.thread.start()
        self.dialog.exec_()


    def onTabChanged(self, index):
        """
        탭에 따라 입력 필드 표시를 조절합니다.
        - Parameter Input AI Model 탭: CSV만 표시, CAD 숨김
        - CAD Input AI Model 탭: CSV와 CAD 모두 표시
        """
        # 일괄적으로 모두 숨김 처리
        self.label_8.setVisible(False)
        self.lineEdit_1.setVisible(False)
        self.pushButton_3.setVisible(False)

        self.label_9.setVisible(False)
        self.lineEdit_2.setVisible(False)
        self.pushButton_4.setVisible(False)

        self.label_10.setVisible(False)
        self.lineEdit_3.setVisible(False)
        self.pushButton_5.setVisible(False)

        self.checkBox_linear.setVisible(False)
        self.checkBox_extrapolation.setVisible(False)

        if index == 0:  # Parameter Input AI Model
            self.label_8.setVisible(True)
            self.lineEdit_1.setVisible(True)
            self.pushButton_3.setVisible(True)

            self.checkBox_linear.setGeometry(QtCore.QRect(80, 195, 300, 25))
            self.checkBox_linear.setVisible(True)

            self.checkBox_extrapolation.setGeometry(QtCore.QRect(80, 220, 300, 25))
            self.checkBox_extrapolation.setVisible(True)


        elif index == 1:  # CAD Input AI Model
            self.label_8.setVisible(True)
            self.lineEdit_1.setVisible(True)
            self.pushButton_3.setVisible(True)

            self.label_9.setVisible(True)
            self.lineEdit_2.setVisible(True)
            self.pushButton_4.setVisible(True)

            self.checkBox_linear.setGeometry(QtCore.QRect(80, 255, 300, 25))  # CAD 경로 아래
            self.checkBox_linear.setVisible(True)

            self.checkBox_extrapolation.setGeometry(QtCore.QRect(80, 280, 300, 25))
            self.checkBox_extrapolation.setVisible(True)

        elif index == 2:  # CATPart to Step Converter
            self.label_10.setVisible(True)
            self.lineEdit_3.setVisible(True)
            self.pushButton_5.setVisible(True)


    def show_info_dialog(self):
        dialog = InfoDialog()
        dialog.exec_()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(669, 419)

        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(7)
        MainWindow.setFont(font)
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("background-color: rgb(243, 243, 243);\n"
"gridline-color: rgb(34, 34, 34);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")


        # 🔵 탭 추가 위치: CSV Folder Path 위
        self.tabs = QtWidgets.QTabWidget(self.centralwidget)
        self.tabs.setGeometry(QtCore.QRect(40, 110, 591, 200))
        # self.tabs.setGeometry(QtCore.QRect(20, 90, 631, 41))

        self.tabs.setStyleSheet("QTabWidget::pane { border: 2px solid #444; border-radius: 6px; padding: 2px; }"
                                 "QTabBar::tab { background: #dcdcdc; border: 1px solid #888; padding: 6px; border-top-left-radius: 4px; border-top-right-radius: 4px; }"
                                 "QTabBar::tab:selected { background: white; font-weight: bold; border-color: #555; }")
        self.tabs.setObjectName("tabs")

        self.tab_param_ai = QtWidgets.QWidget()
        self.tab_param_ai.setObjectName("tab_param_ai")
        self.tabs.addTab(self.tab_param_ai, "   Parameter Input AI    ")

        self.tab_cad_ai = QtWidgets.QWidget()
        self.tab_cad_ai.setObjectName("tab_cad_ai")
        self.tabs.addTab(self.tab_cad_ai, "   CAD Input AI    ")

        self.tab_converter = QtWidgets.QWidget()
        self.tab_converter.setObjectName("tab_converter")
        self.tabs.addTab(self.tab_converter, "   CATPart to step Converter   ")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(510, 340, 101, 51))
        self.pushButton.setStyleSheet("QPushButton {\n"
"    font: 87 15pt \"Arial Black\";                     \n"
"    color: white;                                    \n"
"    border: 2px solid #A0138E;                        \n"
"    border-radius: 10px;                            \n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                      stop: 0 rgb(0, 0, 103), stop: 1 rgb(0, 0, 71));   \n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                      stop: 0 rgb(40, 40, 153), stop: 1 rgb(40, 40, 131));\n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                      stop: 0 rgb(70, 70,255), stop: 1 rgb(70, 70, 237)); \n"
"}")
        self.pushButton.setObjectName("pushButton")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(66, 145, 151, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(1)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("QLabel {\n"
"    font-size: 12px;\n"
"    font-family: \"Arial\";\n"
"    font-weight: bold; \n"
"    color: white; \n"
"    background-color: #505050; \n"
"    border: 1px solid gray;\n"
"    border-radius: 3px;\n"
"    padding: 5px;\n"
"    text-align: center;\n"
"    margin: 5px;\n"
"}")
        self.label_8.setObjectName("label_8")
        self.lineEdit_1 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_1.setGeometry(QtCore.QRect(202, 150, 311, 41))
        self.lineEdit_1.setStyleSheet("QLineEdit {\n"
"    font-size: 13px; \n"
"    border: 2px solid gray;\n"
"    padding: 0 8px;\n"
"    selection-background-color: darkgray;\n"
"    background-color: white; \n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QLineEdit:not(:empty) {\n"
"    background-color: gray;\n"
"}\n"
"\n"
"QLineEdit:disabled {\n"
"    background-color: lightgray;\n"
"    color: gray;\n"
"}\n"
"\n"
"QLineEdit {\n"
"    border-bottom: 2px solid gray;\n"
"    border-top: none;\n"
"    border-left: none;\n"
"    border-right: none;\n"
"}")
        self.lineEdit_1.setObjectName("lineEdit_1")


        self.checkBox_linear = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_linear.setGeometry(QtCore.QRect(80, 195, 300, 25))
        self.checkBox_linear.setText("Option: Scaling Linear Stiffness")
        self.checkBox_linear.setChecked(False)
        self.checkBox_linear.setObjectName("checkBox_linear")

        # Input Extrapolation Range 옵션 (바로 아래 위치)
        self.checkBox_extrapolation = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_extrapolation.setGeometry(QtCore.QRect(80, 225, 300, 25))  # 195 + 30
        self.checkBox_extrapolation.setText("Option: Input Extrapolation Range")
        self.checkBox_extrapolation.setChecked(False)
        self.checkBox_extrapolation.setObjectName("checkBox_extrapolation")
        self.checkBox_extrapolation.toggled.connect(self.handle_extrapolation_checkbox)

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(519, 150, 91, 41))
        self.pushButton_3.setStyleSheet("QPushButton {\n"
"    font-size: 14px;\n"
"    font-family: \"Arial\";\n"
"    font-weight: bold;          \n"
"    background-color: #5B6E91;  \n"
"    border: none;\n"
"    color: white;              \n"
"    padding: 5px 10px;         \n"
"    border-radius: 5px;         \n"
"    text-align: center;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #826AA7; \n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #767676; \n"
"}")
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 310, 391, 81))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("")
        self.label_3.setObjectName("label_3")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, -2, 951, 101))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(423, 13, 241, 81))
        self.label_2.setStyleSheet("background-color: rgb(38, 38, 38);")
        self.label_2.setObjectName("label_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(202, 211, 311, 41))
        self.lineEdit_2.setStyleSheet("QLineEdit {\n"
"    font-size: 13px; \n"
"    border: 2px solid gray;\n"
"    padding: 0 8px;\n"
"    selection-background-color: darkgray;\n"
"    background-color: white; \n"
"    border-radius: 10px;\n"
"}\n"
"\n"
"QLineEdit:not(:empty) {\n"
"    background-color: gray;\n"
"}\n"
"\n"
"QLineEdit:disabled {\n"
"    background-color: lightgray;\n"
"    color: gray;\n"
"}\n"
"\n"
"QLineEdit {\n"
"    border-bottom: 2px solid gray;\n"
"    border-top: none;\n"
"    border-left: none;\n"
"    border-right: none;\n"
"}")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(66, 206, 151, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(1)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setStyleSheet("QLabel {\n"
"    font-size: 12px;\n"
"    font-family: \"Arial\";\n"
"    font-weight: bold; \n"
"    color: white; \n"
"    background-color: #505050; \n"
"    border: 1px solid gray;\n"
"    border-radius: 3px;\n"
"    padding: 5px;\n"
"    text-align: center;\n"
"    margin: 5px;\n"
"}")
        self.label_9.setObjectName("label_9")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(519, 211, 91, 41))
        self.pushButton_4.setStyleSheet("QPushButton {\n"
"    font-size: 14px;\n"
"    font-family: \"Arial\";\n"
"    font-weight: bold;          \n"
"    background-color: #5B6E91;  \n"
"    border: none;\n"
"    color: white;              \n"
"    padding: 5px 10px;         \n"
"    border-radius: 5px;         \n"
"    text-align: center;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: #826AA7; \n"
"}\n"
"\n"
"QPushButton:pressed {\n"
"    background-color: #767676; \n"
"}")
        self.pushButton_4.setObjectName("pushButton_4")
        self.label.raise_()
        self.pushButton.raise_()
        self.lineEdit_1.raise_()
        self.pushButton_3.raise_()
        self.label_3.raise_()
        self.label_8.raise_()
        self.label_2.raise_()
        self.lineEdit_2.raise_()
        self.label_9.raise_()
        self.pushButton_4.raise_()

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(self.run_program3)
        self.pushButton_3.clicked.connect(self.browseFolder)
        self.pushButton_4.clicked.connect(self.browseFolder)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.lineEdit_1, self.pushButton_3)
        MainWindow.setTabOrder(self.pushButton_3, self.lineEdit_2)
        MainWindow.setTabOrder(self.lineEdit_2, self.pushButton_4)

        # ➕ CATPart 탭: Label
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(66, 145, 180, 51))  # 동일 위치
        self.label_10.setFont(font)
        self.label_10.setStyleSheet(self.label_8.styleSheet())  # 동일 스타일
        self.label_10.setObjectName("label_10")
        self.label_10.setText("<html><head/><body><p align=\"center\"><span style=\" font-size:10.5pt;\">CATPart Folder Path</span></p></body></html>")

        # ➕ CATPart 탭: LineEdit
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(230, 150, 283, 41))  # 동일 위치
        self.lineEdit_3.setStyleSheet(self.lineEdit_1.styleSheet())
        self.lineEdit_3.setPlaceholderText(" (Ex)  C:/Input_CATPart")  # 동일 문구
        self.lineEdit_3.setObjectName("lineEdit_3")

        # ➕ CATPart 탭: Browse 버튼
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(519, 150, 91, 41))  # 동일 위치
        self.pushButton_5.setStyleSheet(self.pushButton_3.styleSheet())
        self.pushButton_5.setText("Browser")  # 동일 문구
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(self.browseFolder)  

        self.lineEdit_3.raise_()
        self.label_10.raise_()
        self.pushButton_5.raise_()


        self.infoButton = QtWidgets.QPushButton(self.centralwidget)
        self.infoButton.setGeometry(QtCore.QRect(40, 340, 90, 30))  # 위치 조정 가능
        self.infoButton.setText("Info")
        self.infoButton.setStyleSheet("QPushButton { font-size: 11pt; background-color: #6C7A89; color: white; border-radius: 6px; }"
                                    "QPushButton:hover { background-color: #95A5A6; }")
        self.infoButton.clicked.connect(self.show_info_dialog)


        self.tabs.currentChanged.connect(self.onTabChanged)  # 탭 변경 이벤트 연결
        self.onTabChanged(self.tabs.currentIndex())  # 시작 시 초기 탭 상태에 따라 UI 조절


    def browseFolder(self):
        sender = QtWidgets.QApplication.instance().sender()

        if sender == self.pushButton_3:
            filePath, _ = QFileDialog.getOpenFileName(
                self.centralwidget,
                "Select CSV file",
                "",
                "CSV Files (*.csv)"
            )
            if filePath:
                self.lineEdit_1.setText(filePath)
                self.csv_folder_path = filePath

        elif sender == self.pushButton_4:
            filePath, _ = QFileDialog.getOpenFileName(
                self.centralwidget,
                "Select CAD file",
                "",
                "CAD Files (*.step *.stp)"
            )
            if filePath:
                self.lineEdit_2.setText(filePath)
                self.cad_folder_path = filePath

        elif sender == self.pushButton_5:
            folder_path = QFileDialog.getExistingDirectory(
                self.centralwidget,
                "Select Folder",
                ""
            )
            if folder_path:
                self.lineEdit_3.setText(folder_path)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        # MainWindow.setWindowTitle(_translate("MainWindow", "Program: Bushing 2D Stiffness Predict Model"))
        MainWindow.setWindowTitle(_translate("MainWindow", "BusPredictor"))
        self.pushButton.setText(_translate("MainWindow", "Run"))
        self.label_8.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10.5pt;\">CSV File Path</span></p></body></html>"))
        self.lineEdit_1.setPlaceholderText(_translate("MainWindow", " (Ex)  C:/Input_AI/bushing.csv"))
        self.pushButton_3.setText(_translate("MainWindow", "Browser"))
        # self.label_3.setText(_translate("MainWindow", "<html><head/><body><p>Implemented by <span style=\" font-weight:600;\">Computational Mechanics and Design Optimization Lab</span></p><p>Supervised by <span style=\" font-weight:600;\">Gunwoo Noh </span><span style=\" font-style:italic;\">(gunwoonoh@korea.ac.kr)</span></p><p>Supported by <span style=\" font-weight:600;\">Jihun Kim</span></p></body></html>"))
        

        # 🔽 KOREA Univ 마크 크기 조절
        lab_logo_pixmap = QtGui.QPixmap(":/newPrefix/LabLogo_InputStiffness_pixelRevise.png")
        scaled_lab_logo = lab_logo_pixmap.scaled(
            int(lab_logo_pixmap.width() * 0.9),
            int(lab_logo_pixmap.height() * 0.9),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.label.setPixmap(scaled_lab_logo)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # 왼쪽 정렬 유지

        # 🔽 HYUNDAI 마크 크기 조절
        hyundai_logo_pixmap = QtGui.QPixmap(":/newPrefix/Hyundai2.png")
        scaled_hyundai_logo = hyundai_logo_pixmap.scaled(
            int(hyundai_logo_pixmap.width() * 0.9),
            int(hyundai_logo_pixmap.height() * 0.8),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.label_2.setPixmap(scaled_hyundai_logo)
        self.label_2.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # 오른쪽 정렬 유지


        self.lineEdit_2.setPlaceholderText(_translate("MainWindow", " (Ex)  C:/Input_AI/bushing_RUBBER.step"))
        self.label_9.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10.5pt;\">CAD File Path</span></p></body></html>"))
        self.pushButton_4.setText(_translate("MainWindow", "Browser"))

if __name__ == "__main__":
    freeze_support()
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

