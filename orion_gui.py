import math
import random
import sys
import os
import time
import queue
from PySide6.QtCore import QThread, Signal, QTimer, QEasingCurve, QPropertyAnimation, QObject, QUrl, Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout,
    QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QLabel, QButtonGroup, QLineEdit,
    QMessageBox, QGraphicsOpacityEffect, QPlainTextEdit, QScrollArea, QFrame, QComboBox, QRadioButton
)
from PySide6.QtGui import QFontDatabase, QFont, Qt
from PySide6.QtMultimedia import QSoundEffect
import json
import soundfile as sf
import sounddevice as sd
import numpy as np
import keyboard
from datetime import datetime
from functools import wraps

class HomeAnimation(QThread):
    angle_changed = Signal(list)

    def __init__(self):
        super().__init__()
        self._running = True
        self.frequency = 0.1
        self.time_phase = [random.randint(1, 100) / 10, random.randint(1, 100) / 10]

        # speed up
        self.speed_up = 0
        self.speed_up_start_value = 0
        self.speed_up_to = 0
        self.speed_up_time = 0 # seconds
        self.speed_up_difference = 0
        self.speed_up_start_time = 0
        self.init_speed_up = False
        self.speed_over_time(1, 2)

        # rms aniamtion
        self.rms_values = []
        self.chunk_time = 1
        self.rms_start_time = 0
        self.run_rms_animation = False
        self.init_rms_run = True
        self.rms_add_radius = 0.4
        self.max_rms_time = 0


    def run(self):
        angle = 0.0

        # time based variables
        start_time = time.time()
        last_time = start_time
        local_clock = 0.0

        while self._running:
            # speed up over time
            if self.speed_up != self.speed_up_to:
                if self.init_speed_up:
                    self.speed_up_start_time = time.time()
                    self.speed_up_start_value = self.speed_up
                    self.init_speed_up = False

                speed_d_time = time.time() - self.speed_up_start_time
                speed_percentage = min(1, speed_d_time / self.speed_up_time)
                self.speed_up = self.speed_up_start_value + (speed_percentage * self.speed_up_difference)

            # update clock
            add_time = time.time() - last_time
            local_clock += (add_time * self.speed_up)
            last_time = time.time()

            # start angle
            acceleration_angle = 0.2 * math.sin(2 * math.pi * local_clock * self.frequency) * self.speed_up + 0.1
            angle = (angle + abs(acceleration_angle)) % 360

            # layer 1 conical stops
            stop1 = 0.25 / 2 + (0.1 * math.sin(2 * math.pi * local_clock * self.frequency))
            stop2 = 0.5 + (0.1 * math.sin(2 * math.pi * (local_clock - self.time_phase[0]) * self.frequency))
            stop3 = 0.75 + (0.1 * math.sin(2 * math.pi * (local_clock - self.time_phase[1]) * self.frequency))

            # rms aniamtion
            if self.run_rms_animation:
                if self.init_rms_run:
                    self.rms_start_time = time.time()
                    self.init_rms_run = False

                rms_d_time = time.time() - self.rms_start_time

                if rms_d_time >= self.max_rms_time:
                    self.run_rms_animation = False
                    current_rms = 0
                else:
                    rms_index = int(rms_d_time // self.chunk_time)
                    current_rms = self.rms_values[rms_index]

            else:
                current_rms = 0

            # circle
            circle_stop_1 = 0.1 + (current_rms * self.rms_add_radius)
            circle_stop_2 = 0.2 + (current_rms * self.rms_add_radius)
            circle_stop_3 = 0.6 + (current_rms * self.rms_add_radius)

            self.angle_changed.emit([angle, stop1, stop2, stop3, circle_stop_1, circle_stop_2, circle_stop_3])

            time.sleep(1/60)

    def stop(self):
        self._running = False

    def rms_over_time(self, rms_values, chunk_time, total_length):
        self.rms_values = rms_values
        self.chunk_time = chunk_time
        self.init_rms_run = True
        self.run_rms_animation = True
        self.max_rms_time = total_length * chunk_time

    def speed_over_time(self, speed_up_to, speed_up_time):
        self.speed_up_to = speed_up_to
        self.speed_up_time = speed_up_time
        self.speed_up_difference = self.speed_up_to - self.speed_up
        self.init_speed_up = True

class OrionWorkflow(QThread):
    current_status = Signal(int)
    final_response = Signal(dict, object, int)
    play_sound = Signal(str)

    def __init__(self, get_ai_response, text_to_speach, multi_message_ai_response):
        super().__init__()
        self._running = True
        self.prompt_input: list = []
        self.ai_response = ""
        self.tts_response = None
        self.send_multi_message = False

        # function
        self.get_ai_response = get_ai_response
        self.tts = text_to_speach
        self.multi_message_ai_response = multi_message_ai_response

    def run(self):
        while self._running:

            if self.prompt_input or self.send_multi_message:
                # ai response
                self.play_sound.emit("orion_thinking")
                if self.send_multi_message:
                    self.current_status.emit(5)
                    self.ai_response = self.multi_message_ai_response(None, None, is_final = True)
                else:
                    self.current_status.emit(1)
                    self.ai_response = self.get_ai_response(self.prompt_input[0], self.prompt_input[1])


                # tts
                self.current_status.emit(2)
                self.play_sound.emit("orion_finished")
                self.tts_response = self.tts(self.ai_response["content"])

                # send data back
                self.final_response.emit(self.ai_response, self.tts_response[0], self.tts_response[1])

                # reset
                self.prompt_input = ""
                self.send_multi_message = False

            time.sleep(0.01)

    def stop(self):
        self._running = False

    def feed_data(self, new_data: str, prompt_type:str):
        self.prompt_input = [new_data, prompt_type]

    def feed_multiple_data(self, all_data):
        for item in all_data:
            self.multi_message_ai_response(item[1], item[0], is_final = False)
        self.send_multi_message = True

class OrionExecution(QThread):
    results = Signal(object)
    current_status = Signal(int)
    request_popup = Signal(str, str)

    def __init__(self, settings, py_execution_func, cmd_execution_func, online_serach_func, save_chat_history_func, save_memory_func):
        super().__init__()
        self._running = True

        self.setting = settings

        self.py_execution_func = py_execution_func
        self.cmd_execution_func = cmd_execution_func
        self.online_serach_func = online_serach_func
        self.save_chat_history_func = save_chat_history_func
        self.save_memory_func = save_memory_func

        self.data = {}
        self.check_executions = False
        self.execution_detected = False
        self.all_results = []

        self.users_response = None

    def run(self):

        while self._running:
            if self.check_executions:

                all_keys = list(self.data.keys())
                self.all_results = []

                # chat history
                if "error" in all_keys:
                    if self.data["error"] == "RESOURCE_EXHAUSTED":
                        self.current_status.emit(7.1)
                    elif self.data["error"] == "UNAVAILABLE":
                        self.current_status.emit(7.2)
                    else:
                        self.current_status.emit(7)
                    time.sleep(2)
                    self.current_status.emit(7)

                if "content" in all_keys:
                    self.save_chat_history_func(self.data["content"], "ai")
                    self.current_status.emit(4)

                if "memory" in all_keys:
                    self.save_memory_func(self.data["memory"])
                    self.current_status.emit(4)

                if "pythonCode" in all_keys:
                    if not self.check_if_empty(self.data["pythonCode"]):
                        if not self.setting["allow_python_execution"]:
                            self.all_results.append(["pythonCode", "Die Python Ausführung ist abgeschaltet und somit derzeit nicht verfügbar."])
                        else:
                            if not self.setting["auto_python_execution"]:
                                # max out at 10 lines (only for message box)
                                lines = self.data['pythonCode'].splitlines()
                                max_lines = 10
                                display_text = self.data['pythonCode']
                                if len(lines) > max_lines:
                                    visible_lines = lines[:max_lines]
                                    remaining = len(lines) - max_lines
                                    display_text = "\n".join(
                                        visible_lines) + f"\n\n... <und {remaining} weitere Zeilen>"

                                allow_py = self.ask_user_permission("Python Code", f"Möchtest du folgenden Python code ausführen:\n{display_text}")
                            else:
                                allow_py = True

                            if allow_py:
                                res = self.py_execution_func(self.data["pythonCode"])
                                self.current_status.emit(4.1)
                                self.all_results.append(["pythonCode", res])
                            else:
                                self.all_results.append(["pythonCode", "Der Nutzer hat die Python Ausführung verweigert."])

                if "online_search" in all_keys:
                    if not self.check_if_empty(self.data["online_search"]):
                        if not self.setting["allow_tavily_search"]:
                            self.all_results.append(["search_result", "Die Online-Suche ist abgeschaltet und somit derzeit nicht verfügbar."])
                        else:
                            res = self.online_serach_func(self.data["online_search"])
                            self.current_status.emit(4.3)
                            self.all_results.append(["search_result", [self.data["online_search"], res]])

                if "cmd_execution" in all_keys:
                    if not self.check_if_empty(self.data["cmd_execution"]):
                        if not self.setting["allow_cmd_execution"]:
                            self.all_results.append(["cmd", "Die CMD Ausführung ist asugeschaltet und somit nicht verfügbar."])
                        else:
                            if not self.setting["auto_cmd_execution"]:
                                allow_cmd = self.ask_user_permission("CMD ausfhrung", f"Möchtest du folgenden Befehl asuführen:\n{self.data['cmd_execution']}")
                            else:
                                allow_cmd = True

                            if allow_cmd:
                                res = self.cmd_execution_func(self.data["cmd_execution"])
                                self.current_status.emit(4.2)
                                self.all_results.append(["cmd", res])
                            else:
                                self.all_results.append(["cmd", "Der Nutzer hat die CMD ausführung verweigert."])

                if "end_of_conversation" in all_keys:
                    pass


                # send results
                self.check_executions = False
                self.results.emit(self.all_results)
            time.sleep(0.1)

    def stop(self):
        self._running = False

    def feed_data(self, new_data):
        self.data = new_data
        self.check_executions = True
        self.execution_detected = False

    def ask_user_permission(self, title, text):
        self.users_response = None
        self.request_popup.emit(title, text)

        while self.users_response is None:
            time.sleep(0.01)

        return self.users_response

    def set_users_resonse(self, up):
        self.users_response = up

    def check_if_empty(self, item: str):
        return item.lower() in ["", "null", "none"]

class SpeachToText(QThread):
    current_status = Signal(float)
    final_input = Signal(str)
    show_toast = Signal(str)
    play_sound = Signal(str)

    def __init__(self, settings, transcript_audio_func, predict_activation_word_func):
        super().__init__()
        self._is_running = True
        self.listening_mode = None # smart, voice, manually

        self.transcript_audio_func = transcript_audio_func
        self.predict_activation_word_func = predict_activation_word_func

        self.settings = settings
        self.stt_audio_queue = queue.Queue()
        self.start_record = False
        self.last_record_end_time = 0
        self.ui_current_status = 0
        self.has_already_happened = False
    
    def run(self):
        
        with sd.InputStream(samplerate=16000, channels=1, blocksize=1280, dtype='int16', callback=self.audio_callback):

            while self._is_running:

                if self.ui_current_status != 0:
                    time.sleep(0.01)
                    with self.stt_audio_queue.mutex:
                        self.stt_audio_queue.queue.clear()
                    self.last_record_end_time = time.time()
                    continue

                if not self.settings["use_stt"]:
                    with self.stt_audio_queue.mutex:
                        self.stt_audio_queue.queue.clear()
                    time.sleep(0.01)
                    continue

                if self.start_record:
                    self.play_sound.emit("orion_listening")

                    res = self.record_audio()

                    if res is not False:
                        self.current_status.emit(6.3)
                        transcripted = self.transcript_audio_func(audio_data=res)
                        if transcripted.strip() != "":
                            self.final_input.emit(transcripted)
                        else:
                            self.show_toast.emit("Keine Stimme erkannt.")
                            self.current_status.emit(0)
                    else:
                        self.show_toast.emit("Keine Stimme erkannt.")
                        self.current_status.emit(0)

                    self.start_record = False

                    with self.stt_audio_queue.mutex:
                        self.stt_audio_queue.queue.clear()
                    self.last_record_end_time = time.time()

                else:
                    if self.listening_mode == "smart":
                        pass

                    elif self.listening_mode == "voice":
                        try:
                            audio_chunk = self.stt_audio_queue.get(timeout=0.1)
                            prediction = self.predict_activation_word_func(audio_chunk)

                            if time.time() - self.last_record_end_time < 2:
                                with self.stt_audio_queue.mutex:
                                    self.stt_audio_queue.queue.clear()
                                continue

                            if prediction:
                                self.start_record = True

                            with self.stt_audio_queue.mutex:
                                self.stt_audio_queue.queue.clear()

                        except queue.Empty:
                            pass

                    elif self.listening_mode == "manually":
                        time.sleep(0.01)

    def stop(self):
        self._is_running = False
        
    def start_recording(self):
        self.start_record = True
        
    def set_mode(self, listening_mode):
        self.listening_mode = listening_mode
    
    def audio_callback(self,indata, frames, time_info, status):
        if status:
            print(f"Status from OWW: {status}")
        self.stt_audio_queue.put(np.frombuffer(indata, dtype=np.int16).copy())
    
    def record_audio(self):
        recorded_chunks = []
        speech_started = False
        silence_chunks_count = 0

        chunk_duration = 1280 / 16_000  # ca. 0.08 sec @ 16khz
        max_silence_chunks = int(self.settings["oww_settings"]["SilenceTimeout"] / chunk_duration)

        with self.stt_audio_queue.mutex:
            self.stt_audio_queue.queue.clear()

        self.current_status.emit(6.1)

        start_time = time.time()

        while True:
            if not speech_started and (time.time() - start_time > self.settings["oww_settings"]["InitialSilenceTimeout"]):
                print("Timeout: No speech detected")
                return False

            try:
                chunk = self.stt_audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # get how loud
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

            # check if loud enough
            is_speech = rms > self.settings["oww_settings"]["threshold"]

            if not speech_started:
                if is_speech:
                    speech_started = True
                    self.current_status.emit(6.2)
                    recorded_chunks.append(chunk)
            else:
                recorded_chunks.append(chunk)

                if not is_speech:
                    silence_chunks_count += 1

                    if silence_chunks_count >= max_silence_chunks:
                        break
                else:
                    silence_chunks_count = 0

        # analyse record
        if not recorded_chunks:
            return False

        audio_flatten = np.concatenate(recorded_chunks).astype(np.float32) / 32768.0

        return audio_flatten

class GlobalHotkeyListener(QObject):
    triggered = Signal(str)

    def start_listening(self):
        keyboard.add_hotkey("F7", lambda: self.triggered.emit("F7"))

class TextBubble(QWidget):
    def __init__(self, text_data, parent=None):
        super(TextBubble, self).__init__(parent)

        self.setProperty("text_type", text_data["role"])
        self.setObjectName("TextBubble")
        self.setContentsMargins(0, 30, 0, 0)

        row_layout = QHBoxLayout(self)

        self.bubble_widget = QFrame(self)
        self.bubble_widget.setObjectName("bubble_frame")
        layout = QGridLayout(self.bubble_widget)
        layout.setContentsMargins(10, 10, 10, 10)

        # style based on role
        if text_data["role"] == "user" or text_data["role"] == "transcript":
            row_layout.addWidget(self.bubble_widget)
            row_layout.addStretch()
        elif text_data["role"] == "ai":
            row_layout.addStretch()
            row_layout.addWidget(self.bubble_widget)
        else:
            row_layout.addStretch()
            row_layout.addWidget(self.bubble_widget)
            row_layout.addStretch()

        # add role text
        text_by_role: dict = {
            "user": "NUTZER",
            "ai": "ORION",
            "cmd": "TERMINAL-AUSGABE",
            "cmd_script": "BEFEHL",
            "search_result": "ONLINE-SUCHE",
            "pythonCode": "CODE-AUSGABE",
            "python_code_script": "SCRIPT",
            "system": "SYSTEM",
            "transcript": "TRANSKRIPIERT",
        }
        self.type_lable = QLabel(text_by_role.get(text_data["role"], "UNBEKANNT"))
        self.type_lable.setAlignment(Qt.AlignCenter)
        self.type_lable.setObjectName("text-buble-role")

        # content
        if text_data["role"] == "search_result":
            # get data
            all_data = text_data["content"][1]

            # format results
            all_res = []
            for res in all_data["results"]:
                url = res["url"].split("/")

                all_res.append(
f"""                Website: {url[2]}
                Titel: {res['title']}
                Score: {res['score']:.2f}
""")

            # format text
            text_to_show = (f"Suche: {all_data['query']}\n"
                            f"Ergebnisse:")
            for res in all_res:
                text_to_show += f"\n{res}"

        else:
            text_to_show = text_data["content"]

        self.content_lable = QLabel(str(text_to_show))
        self.content_lable.setObjectName("text-buble-content")

        # time
        time_text: str = text_data["time"]
        time_text = time_text.replace("-", ".")
        time_text = time_text[0: len(time_text) - 3]
        self.time_lable = QLabel(time_text)
        self.time_lable.setObjectName("text-buble-time")

        layout.addWidget(self.type_lable)
        layout.addWidget(self.content_lable)
        layout.addWidget(self.time_lable)

class MemoryBubble(QWidget):
    content_chnage = Signal(str, int)

    def __init__(self, text_data, index, parent=None):
        super(MemoryBubble, self).__init__(parent)

        self.setContentsMargins(0, 30, 0, 0)

        self.frame_widget = QWidget(self)
        self.frame_widget.setProperty("index", index % 4)
        self.frame_widget.setObjectName("MemoryBubble")

        v_layout = QVBoxLayout(self.frame_widget)

        self.bubble_widget = QWidget(self)
        self.bubble_widget.setObjectName("memory_bubble_frame")

        h_layout = QHBoxLayout(self.bubble_widget)
        h_layout.setContentsMargins(10, 10, 10, 10)

        # add horizontal widget
        v_layout.addWidget(self.bubble_widget)

        # time lable
        time_text: str = text_data["time"]
        time_text = time_text.replace("-", ".")
        time_text = time_text[0: len(time_text) - 3]
        self.time_lable = QLabel(time_text)
        self.time_lable.setObjectName("memory-buble-time")
        v_layout.addWidget(self.time_lable)

        # add content
        self.content_input = QLineEdit(self)
        self.content_input.setText(text_data["content"])
        self.content_input.editingFinished.connect(lambda : self.content_chnage.emit(self.content_input.text(), index))
        h_layout.addWidget(self.content_input)

        # add delete button
        delete_button = QPushButton(" Löschen ")
        delete_button.clicked.connect(lambda : self.content_chnage.emit("delete", index))
        h_layout.addWidget(delete_button)

        # add to main
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.frame_widget)

class SettingsCategorie(QWidget):
    def __init__(self, title, max_colums = 2, parent=None):
        super(SettingsCategorie, self).__init__(parent)

        self.layout = QGridLayout(self)

        title_lable = QLabel(title)
        title_lable.setObjectName("settings-title")
        title_lable.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title_lable, 0, 0, 1, max_colums)

        line = QFrame(self)
        line.setObjectName("settings-line")
        self.layout.addWidget(line, 1, 0, 1, max_colums)


    def add_new_widget(self, widget, row, column, rowSpan = 1, columnSpan = 1):
        self.layout.addWidget(widget, row + 2, column, rowSpan, columnSpan)

class SettingsDropDown(QWidget):
    item_selected = Signal(str)
    def __init__(self, title, options, parent=None):
        super(SettingsDropDown, self).__init__(parent)

        self.main_layout = QHBoxLayout(self)

        title_lable = QLabel(title)
        title_lable.setObjectName("drop-down-title")
        self.main_layout.addWidget(title_lable)

        self.drop_down_menu = QComboBox(self)
        self.drop_down_menu.addItems(options)
        self.drop_down_menu.currentTextChanged.connect(self.item_selected)
        self.main_layout.addWidget(self.drop_down_menu)

    def select_option(self, option):
        self.drop_down_menu.setCurrentText(option)

def has_something_changed(func):
    @wraps(func)
    def inner(self, *args, **kwargs):
        self.change_detected = True

        res =  func(self, *args, **kwargs)

        self.update_settings()

        return res

    return inner

class MainWindow(QMainWindow):

    orion_input = Signal(str, str)
    ui_current_status = Signal(int)

    def __init__(self, tts_func, ai_response_func, py_execution_func, cmd_execution_func, online_serach_func,
                 save_chat_history_func, save_memory_func, send_multiple_messages_func, record_audio_func,
                 transcript_audio_func, predict_activation_word_func, get_current_chat_history, get_current_memory,
                 save_edited_memory_func):
        # <editor-fold desc="GENEREL">
        super().__init__()
        self.toast_fade_in = None
        self._active_toast = None
        self.setWindowTitle("ORION GemAI")
        self.resize(1200, 800)

        # style and path
        self.selected_style = "dark_blue"
        self.selected_font = ["Montserrat", "Montserrat-Regular"]
        self.working_dir: str = os.path.abspath("./")
        self.apply_style()

        # orion and gui settings
        self.orion_setting_file: str = rf"{self.working_dir}/assets/settings.json"
        self.gui_settings_file: str = rf"{self.working_dir}/assets/gui/gui_settings.json"

        with open(self.orion_setting_file, "r") as file:
            self.orion_setting = json.load(file)
        with open(self.gui_settings_file, "r") as file:
            self.gui_setting = json.load(file)

        self.ui_sound_volume = self.gui_setting["ui_sounds"]
        # </editor-fold>

        # <editor-fold desc="ALL IMPORTED FUNCTIONS">
        self.tts_func = tts_func
        self.ai_response_func = ai_response_func
        self.send_multiple_messages_func = send_multiple_messages_func

        self.py_execution_func = py_execution_func
        self.cmd_execution_func = cmd_execution_func
        self.online_serach_func = online_serach_func
        self.save_chat_history_func = save_chat_history_func
        self.save_memory_func = save_memory_func

        self.record_audio_func = record_audio_func
        self.transcript_audio_func = transcript_audio_func
        self.predict_activation_word_func = predict_activation_word_func

        self.get_current_chat_history = get_current_chat_history
        self.get_current_memory = get_current_memory
        self.save_edited_memory_func = save_edited_memory_func
        # </editor-fold>

        # <editor-fold desc="UI SOUND FX">
        # UI Sounds
        self.ui_sounds = {}
        for sfx in os.listdir(rf"{self.working_dir}/assets/gui/sounds"):
            sfx_name = sfx.split(".")[0]
            self.ui_sounds[sfx_name] = QSoundEffect(self)
            self.ui_sounds[sfx_name].setVolume(self.ui_sound_volume)
            self.ui_sounds[sfx_name].setSource(QUrl.fromLocalFile(rf"{self.working_dir}/assets/gui/sounds/{sfx}"))

        # </editor-fold>

        # <editor-fold desc="PAGES">
        # home page
        self.orion_info_label: QLabel = QLabel(self)
        self.orion_info_label.setText("Lade...")
        self.orion_info_label.setAlignment(Qt.AlignCenter)

        self.orion_control_element = None

        # history page
        self.all_text_bubbles = []

        # memory page
        self.all_memory_bubbles = []
        self.all_memorys: list = []
        # </editor-fold>

        # <editor-fold desc="SETTINGS (AND PAGE)">
        # all related to settings and settings page
        self.change_detected = False
        # </editor-fold>

        # <editor-fold desc="LAYOUT AND TABS">
        # layout
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName('mainCentralWidget')
        self.setCentralWidget(self.centralwidget)
        self.main_layout = QVBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        # stack widget
        self.stack = QStackedWidget(self)

        # create all widgets
        self.page_home, self.page_home_conical_layer, self.page_home_radial_layer = self.create_page_home()
        self.page_history, self.page_history_widget, self.page_history_layout = self.create_page_history()
        self.page_memory, self.page_memory_layout = self.create_page_memory()
        self.page_settings = self.create_page_settings()


        # add all widgets
        self.stack.addWidget(self.page_home)
        self.stack.addWidget(self.page_history)
        self.stack.addWidget(self.page_memory)
        self.stack.addWidget(self.page_settings)

        self.stack.setCurrentIndex(3)
        # </editor-fold>

        # <editor-fold desc="TAB BAR">
        # upper bar
        self.top_bar_widget = QWidget(self)
        self.top_bar_widget.setObjectName("top_bar_widget")
        self.top_bar_layout = QHBoxLayout(self.top_bar_widget)
        self.top_bar_layout.setSpacing(5)

        # to center upper bar
        h_container_layout = QHBoxLayout()
        h_container_layout.addStretch()
        h_container_layout.addWidget(self.top_bar_widget)
        h_container_layout.addStretch()
        self.main_layout.addLayout(h_container_layout)

        # tab buttons
        self.tab_button_group = QButtonGroup(self)
        self.tab_button_group.setExclusive(True)

        self.all_buttons =  [
            QPushButton("Home"),
            QPushButton("Historie"),
            QPushButton("Erinnerungen"),
            QPushButton("Einstellungen"),
        ]

        for index, button in enumerate(self.all_buttons):
            button.setObjectName("TabButton")
            button.setCheckable(True)
            self.top_bar_layout.addWidget(button)
            self.tab_button_group.addButton(button, index)

        self.all_buttons[0].setChecked(True)
        self.tab_button_group.idClicked.connect(self.stack.setCurrentIndex)

        # add stacked widget
        self.main_layout.addWidget(self.stack)
        # </editor-fold>

        # <editor-fold desc="THREADS">
        # QThread
        self.home_animation_worker = HomeAnimation()
        self.home_animation_worker.angle_changed.connect(self.home_page_animations)
        self.home_animation_worker.start()

        self.orion_worklfow = OrionWorkflow(get_ai_response=self.ai_response_func, text_to_speach=self.tts_func, multi_message_ai_response=self.send_multiple_messages_func)
        self.orion_worklfow.current_status.connect(self.on_status_changed)
        self.orion_worklfow.final_response.connect(self.process_orion_output)
        self.orion_worklfow.play_sound.connect(self.play_sound)
        self.orion_worklfow.start()

        self.orion_execution = OrionExecution(
            settings=self.orion_setting,
            py_execution_func=self.py_execution_func,
            cmd_execution_func=self.cmd_execution_func,
            online_serach_func=self.online_serach_func,
            save_chat_history_func=self.save_chat_history_func,
            save_memory_func=self.save_memory_func,
        )
        self.orion_execution.current_status.connect(self.on_status_changed)
        self.orion_execution.request_popup.connect(self.request_user)
        self.orion_execution.results.connect(self.process_execution_results)
        self.orion_execution.start()

        self.stt_thread = SpeachToText(
            settings=self.orion_setting,
            transcript_audio_func=self.transcript_audio_func,
            predict_activation_word_func = self.predict_activation_word_func
        )
        self.stt_thread.current_status.connect(self.on_status_changed)
        self.stt_thread.show_toast.connect(self.show_toast)
        self.stt_thread.final_input.connect(lambda text: self.orion_input.emit(text, "transcript"))
        self.stt_thread.play_sound.connect(self.play_sound)
        self.stt_thread.set_mode(self.orion_setting["speaking_recognition_mode"])
        self.stt_thread.start()

        self.hotkey_listener = GlobalHotkeyListener()
        self.hotkey_listener.triggered.connect(self.hotkey_pressed)
        self.hotkey_listener.start_listening()
        # </editor-fold>

        # <editor-fold desc="WORKFLOW AND SIGNALS">
        # workflow
        self.current_status = 0
        # 0 = inactive
        # 1 = processing input

        # connect signals
        self.orion_input.connect(self.process_orion_input)
        self.ui_current_status.connect(self.on_status_changed)
        # </editor-fold>

        # reload so no bugs happen
        self.update_memory_entrys(init=False)
        self.update_history_bubbles(init=False)

        # say helo
        self.active_on_status(-1)
        self.orion_input.emit("Begrüße den Nutzer basierend auf deinen Informationen.", "system")

    """GENRELL"""

    def get_current_timestamp(self):
        return datetime.now().strftime("%d-%m-%y %H:%M:%S")

    def apply_style(self):
        with open(rf"{self.working_dir}/assets/gui/theme/{self.selected_style}.qss", "r") as f:
            self.setStyleSheet(f.read())

        font_path = rf"{self.working_dir}/assets/gui/fonts/{self.selected_font[0]}/static/{self.selected_font[1]}.ttf"
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id != 1:
            font_families = QFontDatabase.applicationFontFamilies(font_id)
            if font_families:
                custom_font_family = font_families[0]

                app_font = QFont(custom_font_family, 20)
                self.setFont(app_font)

                current_style = self.styleSheet()
                global_font_style = f"* {{ font-family: '{custom_font_family}'; }}"
                self.setStyleSheet(global_font_style + current_style)
        else:
            print(f"Couldn't load font: {font_path}")

    """PAGES"""

    def create_page_home(self):
        main_widget = QWidget(self)
        main_widget.setObjectName("home_page_background")
        main_widget_layout = QHBoxLayout(main_widget)

        # all layers for animation
        layer1_conical = QWidget()
        layer1_conical.setObjectName("conicalLayer")
        layer1_layout = QVBoxLayout(layer1_conical)
        layer1_layout.setContentsMargins(0, 0, 0, 0)
        main_widget_layout.addWidget(layer1_conical)

        layer2_radial = QWidget()
        layer2_radial.setObjectName("radialOverlayLayer")
        layer1_layout.addWidget(layer2_radial)

        # gui layer
        layer3_grid = QGridLayout(layer2_radial)
        layer3_grid.setContentsMargins(0 ,0 ,0 ,0)
        layer3_grid.setSpacing(0)

        # element based on settings
        if self.orion_setting["use_stt"]:
            if self.orion_setting["speaking_recognition_mode"] == "manually":
                self.orion_info_label.setText("Push to Talk aktiv")

                self.orion_control_element = QPushButton(f"Push-to-Talk aktiv")
                self.orion_control_element.clicked.connect(self.audio_manuel_input)
                self.orion_control_element.setObjectName("input_button")

                layer3_grid.addWidget(self.orion_control_element, 2, 0, 1, 2)

            elif self.orion_setting["speaking_recognition_mode"] == "voice":
                self.orion_info_label.setText("Sprach aktivierung aktiv")

                self.orion_control_element = QLabel(f"Sprachaktivierung aktiv")
                self.orion_control_element.setObjectName("input_lable")
                self.orion_control_element.setAlignment(Qt.AlignCenter)

                layer3_grid.addWidget(self.orion_control_element, 2, 0, 1, 2)

            else:
                self.orion_info_label.setText("Adaptives zuhören aktiv")
        else:
            self.orion_info_label.setText("Text-Eingabe aktiv")

            self.orion_control_element = [QPlainTextEdit(self), QPushButton("Senden", self)]
            self.orion_control_element[0].setPlaceholderText("Nachricht eingeben...")
            self.orion_control_element[0].setObjectName("text_input_line_edit")
            self.orion_control_element[1].clicked.connect(self.text_input)
            self.orion_control_element[1].setObjectName("input_button")

            layer3_grid.addWidget(self.orion_control_element[0], 2, 0)
            layer3_grid.addWidget(self.orion_control_element[1], 2, 1)

        layer3_grid.setRowStretch(1, 2)
        layer3_grid.addWidget(self.orion_info_label, 0, 0, 1, 2)


        return main_widget, layer1_conical, layer2_radial

    def create_page_history(self):
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        page = QWidget()
        page.setObjectName("history_page")
        layout = QVBoxLayout(page)

        # add all bubbles
        self.update_history_bubbles(init=True)

        for chat_buble in self.all_text_bubbles:
            layout.addWidget(chat_buble)

        # scroll area add and scroll down
        scroll_area.setWidget(page)
        scroll_area.verticalScrollBar().setValue(
            scroll_area.verticalScrollBar().maximum()
        )

        return scroll_area, page, layout

    def update_history_bubbles(self, init=False):
        if not init:
            for item in self.all_text_bubbles:
                self.page_history_layout.removeWidget(item)
                item.deleteLater()

        self.all_text_bubbles = []

        for chat_item in self.get_current_chat_history():
            new_item = TextBubble(chat_item, self)
            self.all_text_bubbles.append(new_item)
            if not init:
                self.page_history_layout.addWidget(new_item)

    def create_page_memory(self):

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        page = QWidget()
        page.setObjectName("memory_page")
        layout = QVBoxLayout(page)

        # add all memory bubble
        self.update_memory_entrys(init=True)
        for item in self.all_memory_bubbles:
            layout.addWidget(item)

        # set widget and scroll
        scroll_area.setWidget(page)
        scroll_area.verticalScrollBar().setValue(scroll_area.verticalScrollBar().maximum())

        return scroll_area, layout

    def update_memory_entrys(self, init=False):
        if not init:
            for item in self.all_memory_bubbles:
                self.page_memory_layout.removeWidget(item)
                item.deleteLater()

        self.all_memory_bubbles = []
        self.all_memorys = self.get_current_memory()

        for index, memory_item in enumerate(self.all_memorys):
            new_item = MemoryBubble(memory_item, index, self)
            new_item.content_chnage.connect(self.edit_memorys)
            self.all_memory_bubbles.append(new_item)


            if not init:
                self.page_memory_layout.addWidget(new_item)

        # add new memory button
        new_item = QPushButton("Neue Erinnerung erstellen", self)
        new_item.clicked.connect(lambda : self.edit_memorys("add_new", -1))
        new_item.setObjectName("new_memory_button")

        self.all_memory_bubbles.append(new_item)
        if not init:
            self.page_memory_layout.addWidget(new_item, alignment=Qt.AlignHCenter)

    def create_page_settings(self):
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        page = QWidget()
        page.setObjectName("settings_page")
        layout = QVBoxLayout(page)

        # <editor-fold desc="AI PROVIDER">
        # AI PROVIDER
        ai_type_categorie = SettingsCategorie("KI Anbieter und Prompt", max_colums=4)

        ai_provider = SettingsDropDown("KI Anbieter:", ["Gemini (Cloud)", "Ollama (lokal)"])
        ollama_version = SettingsDropDown("Ollama Version:", ["llama3.1:latest", "llama3.2:latest"])
        gemini_version = SettingsDropDown("Gemini Version:", ["gemini-3.7-flash", "gemini-3.6-flash",
                                                              "gemini-3.5-flash", "gemini-3.5-flash-lite"])

        # set default
        if self.orion_setting["llm_provider"] == "gemini":
            ai_provider.select_option("Gemini (Cloud)")
            gemini_version.select_option(self.orion_setting["llm_version"])
        else:
            ai_provider.select_option("Ollama (lokal)")
            ollama_version.select_option(self.orion_setting["llm_version"])

        # connect to function
        ai_provider.item_selected.connect(self.change_ai_provider)
        ollama_version.item_selected.connect(self.change_llm_modell)
        gemini_version.item_selected.connect(self.change_llm_modell)

        ai_type_categorie.add_new_widget(ai_provider, 0, 0, 2, 2)
        ai_type_categorie.add_new_widget(ollama_version, 0, 2, 1, 2)
        ai_type_categorie.add_new_widget(gemini_version, 1, 2, 1, 2)

        layout.addWidget(ai_type_categorie)
        # </editor-fold>

        layout.addStretch()
        scroll_area.setWidget(page)

        return scroll_area

    """ANIMATION SINGNALS"""

    def home_page_animations(self, new_angles: list):

        new_style = f"""
                    #conicalLayer {{
                        background: qconicalgradient(
                            cx: 0.5, cy: 0.5, 
                            angle: {new_angles[0]},
                            stop: 0.00 rgba(0,51,153, 220),
                            stop: {new_angles[1]} rgba( 72, 61, 139, 255),
                            stop: {new_angles[2]} rgba(0,51,153, 255),
                            stop: {new_angles[3]} rgba(54, 100, 139, 255),
                            stop: 1.00 rgba(0, 51, 153, 220)
                        );
                        border-radius: {self.page_home.width() // 2}px;
                    }}
                    #radialOverlayLayer {{
                        background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                                                     fx:0.5, fy:0.5,
                                                     stop:0 rgba(2, 6, 23, 255),
                                                     stop: {new_angles[4]} rgba(2, 6, 23, 255),
                                                     stop: {new_angles[5]} rgba(2, 6, 23, 0),
                                                     stop: {new_angles[6]} rgba(2, 6, 23, 245),
                                                     stop:1 rgba(2, 6, 23, 255));
                        border-radius: 30px;
                        border: 10px solid rgba(2, 6, 23, 100%)
                    }}
                """
        self.page_home.setStyleSheet(new_style)

    """EVENTS"""

    def closeEvent(self, event):
        self.home_animation_worker.stop()
        self.home_animation_worker.wait()
        event.accept()

    def resizeEvent(self, event, /):
        super().resizeEvent(event)
        w, h = self.width(), self.height()
        # resize main page circle
        min_side = min(self.width(), self.height()) * 0.8
        self.page_home_conical_layer.setFixedSize(int(min_side), int(min_side))
        self.page_home_radial_layer.setFixedSize(int(min_side), int(min_side))

        # history page
        self.page_history_widget.setMaximumWidth(int(w - 52))

        for element in self.all_text_bubbles:
            element.bubble_widget.setMaximumWidth(int(w - 52 - 300))
            element.bubble_widget.setMinimumWidth(500)

    """Inputs"""

    def text_input(self):
        text = self.orion_control_element[0].toPlainText()

        if text.strip():
            self.orion_input.emit(text, "user")
        else:
            self.show_toast("Kein Text eingeben.", duration_ms=2_000)

    def audio_manuel_input(self):
        if self.current_status == 0:
            self.stt_thread.start_recording()
        else:
            self.show_toast("Bitte warte, bis die vorherige Eingabe bearbeitet wurde.", duration_ms=5_000)

    """ON SIGNALS"""

    def process_orion_input(self, text: str, prompt_type: str):
        if self.current_status not in [0, 6.3]:
            self.show_toast("Bitte warte, bis die vorherige Eingabe bearbeitet wurde.", duration_ms=5_000)
            return

        sd.stop(True) # stop every current sd.play-output
        self.home_animation_worker.rms_over_time([0], 0.5, 1)
        self.current_status = 1
        self.orion_worklfow.feed_data(text, prompt_type)

    def process_orion_output(self, output: dict, wav_data: list, fs: int):

        # set rms animatiom and play scaled sound
        rms_values, chunk_time, total_length = self.wav_to_rms(wav_data, fs)
        scaled_audio = wav_data * self.gui_setting["orion_volume"]

        self.home_animation_worker.rms_over_time(rms_values, chunk_time, total_length)
        sd.play(scaled_audio, fs)

        # execute
        self.orion_execution.feed_data(output)

    def on_status_changed(self, status: int):
        # stop speach
        if self.current_status == 0 and status != 0:
            sd.stop(True)  # stop every current sd.play-output
            self.home_animation_worker.rms_over_time([0], 0.5, 1)

        self.active_on_status(status)

        self.current_status = status

        # input active
        if status == 0:
            self.home_animation_worker.speed_over_time(1, 0.5)
            self.orion_info_label.setText("Bereit")

        # processing input
        elif status == 1:
            self.orion_info_label.setText("Eingabe wird verarbeitet...")
            self.home_animation_worker.speed_over_time(5, 1)
        elif status == 2:
            self.orion_info_label.setText("Sprache wird generiert...")

        # output
        elif status == 3:
            self.orion_info_label.setText("Spreche...")
            self.home_animation_worker.speed_over_time(3, 1)

        # executions
        elif status == 4:
            self.orion_info_label.setText("Sepichere Daten...")
            self.home_animation_worker.speed_over_time(5, 1)

        elif status == 4.1:
            self.orion_info_label.setText("Python Code wird ausgeführt...")
            self.home_animation_worker.speed_over_time(5, 1)

        elif status == 4.2:
            self.orion_info_label.setText("CMD wird asugeführt...")
            self.home_animation_worker.speed_over_time(5, 1)

        elif status == 4.3:
            self.orion_info_label.setText("Online-Suche wird durchgeführt...")
            self.home_animation_worker.speed_over_time(5, 1)

        # execution results
        elif status == 5:
            self.orion_info_label.setText("Outputs werden verarbeitet...")
            self.home_animation_worker.speed_over_time(5, 1)

            # execution results

        # STT
        elif status == 6.1:
            self.orion_info_label.setText("Warte aufs sprechen...")
            self.home_animation_worker.speed_over_time(5, 1)
        elif status == 6.2:
            self.orion_info_label.setText("Audio wird aufgenommen...")
            self.home_animation_worker.speed_over_time(5, 1)
        elif status == 6.3:
            self.orion_info_label.setText("Wird transkripiert...")
            self.home_animation_worker.speed_over_time(5, 1)
        elif status == 6.4:
            self.orion_info_label.setText("Keine Stimme erkannt.")
            self.home_animation_worker.speed_over_time(5, 1)

        # error
        elif status == 7:
            self.orion_info_label.setText("Es ist ein Fehler aufgetreten")
            self.home_animation_worker.speed_over_time(3, 1)
        elif status == 7.1:
            self.orion_info_label.setText(f"Limit der API erreicht (im Model {self.orion_setting['gemini_version']})")
            self.home_animation_worker.speed_over_time(3, 1)
        elif status == 7.2:
            self.orion_info_label.setText("Google Server überlastet")
            self.home_animation_worker.speed_over_time(3, 1)

    def request_user(self, title, text):
        reply = QMessageBox.question(
            self, title, text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        result = (reply == QMessageBox.Yes)
        self.orion_execution.set_users_resonse(result)

    def process_execution_results(self, exec_results: list):

        if len(exec_results) > 0:
            self.orion_worklfow.feed_multiple_data(exec_results)
        else:
            self.ui_current_status.emit(0)

    def hotkey_pressed(self, key):
        # manages all hot keys
        if key == "F7":
            if self.orion_setting["use_stt"]:
                self.stt_thread.start_recording()

    # deactivate / activate ui elements

    def active_on_status(self, status):
        # threads
        self.stt_thread.ui_current_status = status

        # ui elements
        is_active_str = "True" if status == 0 else "False"
        if self.orion_setting["use_stt"]:
            self.orion_control_element.setProperty("is_active", is_active_str)
            self.orion_control_element.style().unpolish(self.orion_control_element)
            self.orion_control_element.style().polish(self.orion_control_element)
            if self.orion_setting["speaking_recognition_mode"] == "manually":
                pass
            elif self.orion_setting["speaking_recognition_mode"] == "voice":
                if status == 0:
                    self.orion_control_element.setText(f"Drücke -{self.gui_setting['push-to-talk']}- oder sage 'Orion'")
                elif status == 6.1:
                    self.orion_control_element.setText(f"Fange an zu sprechen...")
                elif status == 6.2:
                    self.orion_control_element.setText(f"")
                else:
                    self.orion_control_element.setText("Bitte warten...")
            else:
                pass
        else:
            self.orion_control_element[0].setReadOnly(status != 0)
            for i in range(2):
                self.orion_control_element[i].setProperty("is_active", is_active_str)
                self.orion_control_element[i].style().unpolish(self.orion_control_element[i])
                self.orion_control_element[i].style().polish(self.orion_control_element[i])
            if status == 0 and int(self.current_status):
                self.orion_control_element[0].setPlainText("")

    def play_sound(self, sound_name):
        self.ui_sounds[sound_name].play()

    # edit memory

    def edit_memorys(self, new_data, index):
        if new_data == "delete":
            self.all_memorys.pop(index)
        elif new_data == "add_new" and index == -1:
            self.all_memorys.append({
                "time":self.get_current_timestamp(),
                "content": ""
            })
        else:
            self.all_memorys[index]["content"] = new_data

        self.save_edited_memory_func(self.all_memorys)
        self.update_memory_entrys(init=False)


    """STATIC FUNCTIONS"""

    def wav_to_rms(self, wav_data, fs):

        # convert 2 dimensional to 1 dimensional
        if wav_data.ndim > 1:
            wav_data = np.mean(wav_data, axis=1)

        # get chunk_size
        chunk_size = int(fs / 30) # x per seconds

        # split into chunks
        num_chunks = len(wav_data) // chunk_size
        truncated_data = wav_data[:num_chunks * chunk_size]

        chunks = truncated_data.reshape(num_chunks, chunk_size)

        # calculate rms
        rms_values = np.sqrt(np.mean(chunks.astype(np.float32) ** 2, axis=1))

        # chunk length (time)
        chunk_time = chunk_size / fs

        return [rms_values, chunk_time, len(rms_values)]

    """ELEMENTS"""

    def show_toast(self, message: str, duration_ms: int = 2000):

        if hasattr(self, "_active_toast") and self._active_toast:
            self._active_toast.close()

        toast = QLabel(message, self)
        toast.setObjectName("toast")

        toast.adjustSize()

        parent_width = self.width()
        parent_height = self.height()

        toast_width = toast.width()
        toast_height = toast.height()

        x = (parent_width - toast_width) // 2
        y = parent_height - toast_height - 60
        toast.move(x, y)

        toast.show()
        self._active_toast = toast

        # set 0 opacity
        opacity_effect = QGraphicsOpacityEffect(toast)
        toast.setGraphicsEffect(opacity_effect)
        opacity_effect.setOpacity(0.0)

        # animation: FADE IN
        self.toast_fade_in = QPropertyAnimation(opacity_effect, b"opacity")
        self.toast_fade_in.setDuration(250)
        self.toast_fade_in.setStartValue(0)
        self.toast_fade_in.setEndValue(1)
        self.toast_fade_in.setEasingCurve(QEasingCurve.InOutQuad)

        def fade_out_animation():
            if not toast.isVisible():
                return

            self.fade_out_anim = QPropertyAnimation(opacity_effect, b"opacity")
            self.fade_out_anim.setDuration(500)  # Etwas langsamer beim Ausblenden
            self.fade_out_anim.setStartValue(1.0)
            self.fade_out_anim.setEndValue(0.0)
            self.fade_out_anim.setEasingCurve(QEasingCurve.InOutQuad)

            self.fade_out_anim.finished.connect(toast.close)
            self.fade_out_anim.start()

        # start
        self.toast_fade_in.start()
        fade_out_delay = max(100, duration_ms - 500)
        QTimer.singleShot(fade_out_delay, fade_out_animation)

    """SETTING CHANGE (EVENT ON SIGNAL)"""

    def update_settings(self):
        pass

    @has_something_changed
    def change_ai_provider(self, new_provider):
        print(new_provider)

    @has_something_changed
    def change_llm_modell(self, new_modell):
        print(new_modell)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # sample functions
    def ai_res(text: str, prompt_type: str):
        time.sleep(.5)
        test_comp = {"content":"Hallo, dies ist kein KI generierter Inhalt, sonder lediglich ein Test. Bitte starte die Datei: Mein punkt p y um das programm korrekt zu starten."}
        return test_comp

    def tts_res(text: str):
        time.sleep(.5)
        data, fs = sf.read(rf"./assets/gui/test.wav")
        return data, fs

    def get_history():
        with open("./assets/json_files/chat_history.json", "r", encoding="utf-8") as history_file:
            data = json.load(history_file)
        return data

    def get_memory():
        with open("./assets/json_files/memory.json", "r", encoding="utf-8") as mem_file:
            data = json.load(mem_file)
        return data

    def save_edited_memory(edited_memory):
        print("Saving edited memory")

        with open("./assets/json_files/memory.json", "w", encoding="utf-8") as mem_file:
            json.dump(edited_memory, mem_file, ensure_ascii=False, indent=4)

    window = MainWindow(
        tts_func=tts_res,
        ai_response_func=ai_res,
        py_execution_func=lambda *args: "No Func",
        cmd_execution_func=lambda *args: "No Func",
        online_serach_func=lambda *args: "No Func",
        save_chat_history_func=lambda *args: "No Func",
        save_memory_func=lambda *args: "No Func",
        send_multiple_messages_func=lambda *args, is_final: "No Func",
        record_audio_func=lambda *args: "No Func",
        transcript_audio_func=lambda *args, audio_data: "No func",
        predict_activation_word_func=lambda *args: False,
        get_current_chat_history=get_history,
        get_current_memory=get_memory,
        save_edited_memory_func=save_edited_memory
    )
    window.showMaximized()

    sys.exit(app.exec())