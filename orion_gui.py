import math
import random
import sys
import os
import time
import queue
from PySide6.QtCore import QThread, Signal, QTimer, QEasingCurve, QPropertyAnimation, QObject, QUrl, Qt, QProcess
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout,
    QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QLabel, QButtonGroup, QLineEdit,
    QMessageBox, QGraphicsOpacityEffect, QPlainTextEdit, QScrollArea, QFrame, QComboBox, QCheckBox,
    QSlider, QDialog, QProgressBar, QSizePolicy
)
from PySide6.QtGui import QFontDatabase, QFont, Qt, QFontMetrics, QKeySequence
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
                self.tts_response = self.tts(self.ai_response["content"], specific_voice_style="")

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
    has_conversation_ended = Signal(bool)

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
                    if not "no_safe" in all_keys:
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
                    self.has_conversation_ended.emit(self.data["end_of_conversation"])
                else:
                    self.has_conversation_ended.emit(False)


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
        self.keep_conversation = False

        self.send_rms_values = False
        self.rms_bus = lambda x: x
        self.send_score_values = False
        self.score_bus = lambda x: x
    
    def run(self):
        
        with sd.InputStream(samplerate=16000, channels=1, blocksize=1280, dtype='int16', callback=self.audio_callback):

            while self._is_running:


                if self.send_rms_values:
                    try:
                        chunk = self.stt_audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                    self.rms_bus(rms)
                    continue

                elif self.send_score_values:
                    try:
                        audio_chunk = self.stt_audio_queue.get(timeout=0.1)
                        _, score = self.predict_activation_word_func(audio_chunk)
                    except queue.Empty:
                        continue

                    self.score_bus(score)

                    with self.stt_audio_queue.mutex:
                        self.stt_audio_queue.queue.clear()

                    continue

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
                        self.start_record = self.keep_conversation

                        if self.keep_conversation:
                            self.keep_conversation = False
                        else:
                            try:
                                audio_chunk = self.stt_audio_queue.get(timeout=0.1)
                                prediction, score = self.predict_activation_word_func(audio_chunk)

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


                    elif self.listening_mode == "voice":
                        try:
                            audio_chunk = self.stt_audio_queue.get(timeout=0.1)
                            prediction, score = self.predict_activation_word_func(audio_chunk)

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

        InitialSilenceTimeout = self.settings["oww_settings"]["InitialSilenceTimeout"]
        if self.settings["speaking_recognition_mode"] == "smart":
            InitialSilenceTimeout *= 10

        while True:
            if not speech_started and (time.time() - start_time > InitialSilenceTimeout):
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

    def send_rms_to_volume_test(self, is_true, func = lambda x: x):
        self.send_rms_values = is_true
        self.rms_bus = func

    def send_score_to_test(self, is_true, func = lambda x: x):
        self.send_score_values = is_true
        self.score_bus = func

class GlobalHotkeyListener(QObject):
    triggered = Signal(str)

    def start_listening(self, hotkey):
        keyboard.add_hotkey(hotkey, lambda: self.triggered.emit("push-to-talk"))

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
        self.type_lable.setAlignment(Qt.AlignmentFlag.AlignCenter)
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

        self.content_lable = QLabel("")
        self.content_lable.setObjectName("text-buble-content")

        metrics = QFontMetrics(self.content_lable.font())
        required_width = metrics.horizontalAdvance(str(text_to_show)) + 8
        if text_data["role"] != "python_code_script" and required_width > 900:
            self.content_lable.setWordWrap(True)
            self.content_lable.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.content_lable.setText(f"<p style='line-height: 150%;'>{str(text_to_show)}</p>")

        # time
        time_text: str = text_data["time"]
        time_text = time_text.replace("-", ".")
        time_text = time_text[0: len(time_text) - 3]
        self.time_lable = QLabel(time_text)
        self.time_lable.setObjectName("text-buble-time")

        layout.addWidget(self.type_lable)
        layout.addWidget(self.content_lable)
        layout.addWidget(self.time_lable)

        self.style().unpolish(self)
        self.style().polish(self)

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
    def __init__(self, title: str, max_colums: int = 2, parent=None):
        super(SettingsCategorie, self).__init__(parent)

        self.layout = QGridLayout(self)

        title_lable = QLabel(title.upper())
        title_lable.setObjectName("settings-title")
        title_lable.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(title_lable, 0, 0, 1, max_colums + 2)

        line = QFrame(self)
        line.setObjectName("settings-line")
        self.layout.addWidget(line, 1, 0, 1, max_colums + 2)


    def add_new_widget(self, widget, row, column, rowSpan = 1, columnSpan = 1):
        self.layout.addWidget(widget, row + 2, column + 1, rowSpan, columnSpan)

class SettingsDropDown(QWidget):
    item_selected = Signal(str)
    index_selected = Signal(int)
    def __init__(self, title, options, parent=None):
        super(SettingsDropDown, self).__init__(parent)

        self.main_layout = QHBoxLayout(self)

        title_lable = QLabel(title)
        title_lable.setObjectName("drop-down-title")
        self.main_layout.addWidget(title_lable)

        self.drop_down_menu = QComboBox(self)
        self.drop_down_menu.addItems(options)
        self.drop_down_menu.currentTextChanged.connect(self.item_selected)
        self.drop_down_menu.currentIndexChanged.connect(self.index_selected)
        self.main_layout.addWidget(self.drop_down_menu)

    def select_option(self, option):
        self.drop_down_menu.setCurrentText(option)

class SettingsSlider(QWidget):
    slider_value_changed = Signal(int)

    def __init__(self, title, min_value, max_value, unit="", fixed_title_with: int = 220, toolTip: str = "", parent=None,
                 step_size: int = 1):
        super(SettingsSlider, self).__init__(parent)
        self.unit = unit
        self.step_size = step_size
        self.min_value = min_value

        self.setToolTip(toolTip)
        layout = QHBoxLayout(self)

        # title
        title_lable = QLabel(title)
        title_lable.setFixedWidth(fixed_title_with)
        layout.addWidget(title_lable)

        # slider
        self.slider = QSlider(Qt.Horizontal, self)

        self.slider.setMaximum(max_value)
        self.slider.setMinimum(min_value)
        self.slider.setMinimumWidth(50)
        self.slider.setTickInterval(100)

        self.slider.sliderReleased.connect(lambda : self.slider_value_changed.emit(self.slider.value()))
        self.slider.valueChanged.connect(self.on_value_chnage_slider)

        layout.addWidget(self.slider)

        # current value lable
        max_text = f"{max_value} {self.unit}"
        min_text = f"{min_value} {self.unit}"
        longest_text = max_text if len(max_text) >= len(min_text) else min_text

        self.slider_lable = QLabel(self)
        self.slider_lable.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        metrics = QFontMetrics(self.slider_lable.font())
        required_width = metrics.horizontalAdvance(longest_text) + 8
        self.slider_lable.setFixedWidth(required_width)

        layout.addWidget(self.slider_lable)

        self.set_curent_value(min_value)

    def set_curent_value(self, value):
        self.slider.setValue(value)
        self.slider_lable.setText(str(int(value)) + self.unit)

    def on_value_chnage_slider(self, value):
        snapped_value = round(value / self.step_size) * self.step_size
        snapped_value = max(snapped_value, self.min_value)

        self.set_curent_value(snapped_value)

class VolumeTest(QDialog):

    apply_new_volume = Signal(int)

    def __init__(self, parent):
        super(VolumeTest, self).__init__(parent)

        self.setWindowTitle("Lautstärke Test")
        self.setFixedSize(300, 180)
        self.setObjectName("volume-test")

        self.max_value = 0

        layout = QGridLayout(self)

        # left side: max value, set value, exit

        self.max_value_lable = QLabel("---")
        self.max_value_lable.setAlignment(Qt.AlignmentFlag.AlignCenter)


        self.recommended_value = QLabel("---")
        self.recommended_value.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(QLabel("Maximaler Wert"), 0, 0, 1, 1)
        layout.addWidget(self.max_value_lable, 0, 1, 1, 1)


        layout.addWidget(QLabel("Empfohlen:"), 1, 0, 1, 1)
        layout.addWidget(self.recommended_value, 1, 1, 1, 1)

        apply_button = QPushButton("Wert übernehmen")
        apply_button.setObjectName("apply-button")
        apply_button.clicked.connect(lambda : self.apply_new_volume.emit(int(self.max_value * 0.8)))
        self.apply_new_volume.connect(self.close)

        exit_button = QPushButton("Abbrechen")
        exit_button.setObjectName("exit-button")
        exit_button.clicked.connect(self.close)

        layout.addWidget(apply_button, 2, 0, 1, 2)
        layout.addWidget(exit_button, 3, 0, 1, 2)

        # right side: current volume
        self.volume_lable = QLabel(str(self.max_value))
        self.volume_lable.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.volume_lable, 3, 2, 1, 1)

        self.volume_bar = QProgressBar(self)
        self.volume_bar.setObjectName("volume-bar")
        self.volume_bar.setOrientation(Qt.Orientation.Vertical)
        self.volume_bar.setRange(0, self.max_value)
        self.volume_bar.setTextVisible(False)
        self.volume_bar.setFixedWidth(50)

        layout.addWidget(self.volume_bar, 0, 2, 3, 1)

    def recv_rms(self, rms_value):
        rms_value_int = int(rms_value)

        self.volume_bar.setValue(rms_value_int)
        self.volume_lable.setText(str(int(rms_value_int)))

        self.max_value = max(self.max_value, rms_value_int)
        self.volume_bar.setRange(0, self.max_value)

        self.max_value_lable.setText(f"{self.max_value}")
        self.recommended_value.setText(f"{int(self.max_value * 0.8)}")

class ScoreTest(QDialog):
    apply_new_score = Signal(int)
    def __init__(self, parent):
        super(ScoreTest, self).__init__(parent)

        self.setWindowTitle("Score Test")
        self.setFixedSize(300, 180)
        self.setObjectName("score-test")

        self.max_value = 0

        layout = QGridLayout(self)

        # left side
        self.max_score_lable = QLabel("---")
        self.max_score_lable.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(QLabel("Maximaler Wert"), 0, 0, 1, 1)
        layout.addWidget(self.max_score_lable, 0, 1, 1, 1)

        self.recommended_score_lable = QLabel("---")
        self.recommended_score_lable.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(QLabel("Empfohlen"), 1, 0, 1, 1)
        layout.addWidget(self.recommended_score_lable, 1, 1, 1, 1)

        apply_button = QPushButton("Wert übernehmen")
        apply_button.setObjectName("apply-button")
        apply_button.clicked.connect(lambda: self.apply_new_score.emit(int(self.max_value * 0.8)))
        self.apply_new_score.connect(self.close)

        exit_button = QPushButton("Abbrechen")
        exit_button.setObjectName("exit-button")
        exit_button.clicked.connect(self.close)

        layout.addWidget(apply_button, 2, 0, 2, 1)
        layout.addWidget(exit_button, 3, 0, 2, 1)

        # right side
        self.score_bar = QProgressBar(self)
        self.score_bar.setObjectName("score-bar")
        self.score_bar.setOrientation(Qt.Orientation.Vertical)
        self.score_bar.setRange(0, 100)
        self.score_bar.setTextVisible(False)
        self.score_bar.setFixedWidth(50)

        self.score_bar_lable = QLabel(self)
        self.score_bar_lable.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.score_bar, 0, 2, 3, 1)
        layout.addWidget(self.score_bar_lable, 3, 2, 1, 1)


    def recv_score(self, score_value):
        score_value_int = int(score_value * 100)

        self.score_bar.setValue(score_value_int)
        self.score_bar_lable.setText(str(score_value_int) + "%")

        self.max_value = max(self.max_value, score_value_int)
        self.max_score_lable.setText(str(self.max_value) + "%")
        self.recommended_score_lable.setText(str(int(self.max_value * 0.8)) + "%")

class HotKeyButton(QPushButton):
    hotkey_changed = Signal(str)

    def __init__(self, parent, current, added_text: str = ""):
        super(HotKeyButton, self).__init__(parent)

        self.setText(f"Hotkey {added_text}-{current}-")

        self.added_text = added_text
        self.current_hotkey = current
        self.is_recording = False

        self.clicked.connect(self.start_recording)

    def start_recording(self):
        self.is_recording = True
        self.setText("Taste drücken...")
        self.setFocus()

    def keyPressEvent(self, event):
        if not self.is_recording:
            super().keyPressEvent(event)
            return

        key = event.key()

        if key == Qt.Key.Key_Escape:
            self.stop_recording(self.current_hotkey)
            return

        if key in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            return

        modifiers = event.modifiers()

        key_sequence = QKeySequence(event.keyCombination())
        new_hotkey = key_sequence.toString(QKeySequence.NativeText)

        self.stop_recording(new_hotkey)

    def stop_recording(self, hotkey_str):
        self.is_recording = False
        self.current_hotkey = hotkey_str

        self.setText(f"Hotkey {self.added_text}-{hotkey_str}-")
        self.hotkey_changed.emit(self.current_hotkey)

    def focusOutEvent(self, event):
        # Falls der Nutzer woanders hinklickt, Aufzeichnung abbrechen
        if self.is_recording:
            self.stop_recording(self.current_hotkey)
        super().focusOutEvent(event)

    def set_hotkey(self, hotkey_str):
        self.current_hotkey = hotkey_str
        self.setText(f"Hotkey {self.added_text}-{hotkey_str}-")

def has_something_changed(func):
    @wraps(func)
    def inner(self, *args, **kwargs):
        self.has_changes_made.emit()

        res =  func(self, *args, **kwargs)

        self.update_settings()

        return res

    return inner

class MainWindow(QMainWindow):

    orion_input = Signal(str, str)
    ui_current_status = Signal(int)
    has_changes_made = Signal()

    def __init__(self, tts_func, ai_response_func, py_execution_func, cmd_execution_func, online_serach_func,
                 save_chat_history_func, save_memory_func, send_multiple_messages_func, record_audio_func,
                 transcript_audio_func, predict_activation_word_func, get_current_chat_history, get_current_memory,
                 save_edited_memory_func, save_new_settings_func, save_new_character_func, get_current_character_func,
                 restart_app_func):
        # <editor-fold desc="GENEREL">
        super().__init__()
        self.toast_fade_in = None
        self._active_toast = None
        self.all_microphones = self.get_input_devices()

        self.setWindowTitle("ORION GemAI")

        # style and path
        self.selected_style = "dark_blue"
        self.selected_font = ["Montserrat", "Montserrat-Regular"]
        self.working_dir: str = os.path.abspath("./")
        self.apply_style()

        # orion and gui settings
        self.orion_setting_file: str = rf"{self.working_dir}/assets/settings.json"
        self.gui_settings_file: str = rf"{self.working_dir}/assets/gui/gui_settings.json"

        with open(self.orion_setting_file, "r", encoding="utf-8") as file:
            self.orion_setting = json.load(file)
        with open(self.gui_settings_file, "r", encoding="utf-8") as file:
            self.gui_setting = json.load(file)

        self.ui_sound_volume = self.gui_setting["ui_sounds"]

        self.all_character_templates = os.listdir(rf"{self.working_dir}{self.gui_setting['character_templates']}")
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
        self.save_new_settings_func = save_new_settings_func

        self.save_new_character_func = save_new_character_func
        self.get_current_character_func = get_current_character_func

        self.restart_app_func = restart_app_func
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
        self.orion_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.orion_control_element = None

        # history page
        self.all_text_bubbles = []

        # memory page
        self.all_memory_bubbles = []
        self.all_memorys: list = []
        # </editor-fold>

        # <editor-fold desc="SETTINGS (AND PAGE)">
        # all related to settings and settings page
        # ui
        self.all_change_hotkey_buttons = []

        self.restart_app_button = None

        self.ollama_version_drop_down = None
        self.gemini_version_drop_down = None

        self.orion_input_settings_stack = None

        self.manuel_silence_time_out_slider = None
        self.manuel_volume_threashold = None

        self.voice_silence_time_out_slider = None
        self.voice_volume_threashold = None
        self.voice_score_threashold = None

        self.adaptive_silence_time_out_slider = None
        self.adaptive_volume_threashold = None
        self.adaptive_score_threashold = None

        self.all_character_slider: dict = {}

        # settings
        self.change_detected = False
        self.current_character = self.get_current_character_func()
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

        self.stack.setCurrentIndex(1)
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
        self.orion_execution.has_conversation_ended.connect(self.update_end_of_conv)
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
        self.hotkey_listener.start_listening(hotkey=self.gui_setting["push-to-talk"])
        # </editor-fold>

        # <editor-fold desc="WORKFLOW AND SIGNALS">
        # workflow
        self.current_status = 0
        # 0 = inactive
        # 1 = processing input

        # connect signals
        self.orion_input.connect(self.process_orion_input)
        self.ui_current_status.connect(self.on_status_changed)
        self.has_changes_made.connect(self.changes_has_made)
        # </editor-fold>

        # reload so no bugs happen
        self.update_memory_entrys(init=False)
        self.update_history_bubbles(init=False)

        # say helo
        self.active_on_status(-1)
        self.orion_input.emit("Begrüße den Nutzer basierend auf deinen Informationen.", "system")

    """GENRELL"""

    def get_input_devices(self):
        input_devices = []
        for index, device in enumerate(sd.query_devices()):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'id': index,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'default_samplerate': device['default_samplerate']
                })
        return input_devices

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
                self.orion_control_element.setAlignment(Qt.AlignmentFlag.AlignCenter)

                layer3_grid.addWidget(self.orion_control_element, 2, 0, 1, 2)

            elif self.orion_setting["speaking_recognition_mode"] == "smart":
                self.orion_info_label.setText("Adaptives Gespräch")

                self.orion_control_element = QLabel(f"Sprachaktivierung aktiv")
                self.orion_control_element.setObjectName("input_lable")
                self.orion_control_element.setAlignment(Qt.AlignmentFlag.AlignCenter)

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

        for index, chat_item in enumerate(self.get_current_chat_history()):
            new_item = TextBubble(chat_item, self)
            self.all_text_bubbles.append(new_item)
            if not init:
                self.page_history_layout.addWidget(new_item)

            if index == self.gui_setting["max_history_shown"]:
                break

        if not init:
            QTimer.singleShot(20, lambda : self.page_history.verticalScrollBar().setValue( self.page_history.verticalScrollBar().maximum() ))

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
        new_item = QPushButton(f"Neue Erinnerung erstelle", self)
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

        self.restart_app_button = QPushButton("App neustarten um Änderungen zu übernehmen")
        self.restart_app_button.setObjectName("restart_app_button")
        self.restart_app_button.clicked.connect(self.restart_app)
        self.restart_app_button.hide()

        layout.addWidget(self.restart_app_button)

        # <editor-fold desc="AI PROVIDER">
        # AI PROVIDER
        ai_type_categorie = SettingsCategorie("KI Typ", max_colums=4)

        ai_provider = SettingsDropDown("KI Anbieter:", ["Gemini (Cloud)", "Ollama (lokal)"])
        self.ollama_version_drop_down = SettingsDropDown("Ollama Version:", ["llama3.1:latest", "llama3.2:latest"])
        self.gemini_version_drop_down = SettingsDropDown("Gemini Version:", ["gemini-3.7-flash", "gemini-3.6-flash",
                                                              "gemini-3.5-flash", "gemini-3.5-flash-lite"])

        # set default
        if self.orion_setting["llm_provider"] == "google":
            ai_provider.select_option("Gemini (Cloud)")
            self.gemini_version_drop_down.select_option(self.orion_setting["llm_version"])
        else:
            ai_provider.select_option("Ollama (lokal)")
            self.ollama_version_drop_down.select_option(self.orion_setting["llm_version"])

        # connect to function
        ai_provider.item_selected.connect(self.change_ai_provider)
        self.ollama_version_drop_down.item_selected.connect(self.change_llm_modell)
        self.gemini_version_drop_down.item_selected.connect(self.change_llm_modell)

        ai_type_categorie.add_new_widget(ai_provider, 0, 0, 2, 2)
        ai_type_categorie.add_new_widget(self.ollama_version_drop_down, 0, 2, 1, 2)
        ai_type_categorie.add_new_widget(self.gemini_version_drop_down, 1, 2, 1, 2)

        layout.addWidget(ai_type_categorie)
        # </editor-fold>

        # <editor-fold desc="PROMPT">
        # Prompt
        prompt_categorie = SettingsCategorie("Prompt", max_colums=4)

        send_history_checkbox = QCheckBox("Sende Historie", self)
        send_history_checkbox.setToolTip("Sende den gespeicherten Chatverlauf (Historie) mit")
        send_history_checkbox.setChecked(self.orion_setting["send_history"])
        send_history_checkbox.clicked.connect(lambda value: self.prompt_items_changes("send_history", value))

        send_memory_checkbox = QCheckBox("Sende Erinnerungen", self)
        send_memory_checkbox.setChecked(self.orion_setting["send_memory"])
        send_memory_checkbox.setToolTip("Sende die gespeicherten Erinnerungen mit")
        send_memory_checkbox.clicked.connect(lambda value: self.prompt_items_changes("send_memory", value))

        prompt_categorie.add_new_widget(send_history_checkbox, 0, 0, 1, 2)
        prompt_categorie.add_new_widget(send_memory_checkbox, 1, 0, 1, 2)

        history_number_slider = SettingsSlider("Anzahl der Historie senden", 1, 2_000,
                                               toolTip="Wie viele Elemnete des Chatverlaufes sollen maximal versendet werden?",
                                               step_size=25)
        history_number_slider.set_curent_value(self.orion_setting["history_max_items"])
        history_number_slider.slider_value_changed.connect(lambda value: self.prompt_items_changes("history_max_items", value))

        memory_number_slider = SettingsSlider("Anzahl der Erinnerungen senden", 1, 2_000,
                                               toolTip="Wie viele Elemente aller Erinnerungen sollen maximal versendet werden?",
                                               step_size=25)
        memory_number_slider.set_curent_value(self.orion_setting["memory_max_items"])
        memory_number_slider.slider_value_changed.connect(lambda value: self.prompt_items_changes("memory_max_items", value))

        prompt_categorie.add_new_widget(history_number_slider, 0, 2, 1, 2)
        prompt_categorie.add_new_widget(memory_number_slider, 1, 2, 1, 2)

        layout.addWidget(prompt_categorie)
        # </editor-fold>

        # <editor-fold desc="INPUT TYPE">
        # input type
        orion_input_settings = SettingsCategorie("Input variante", max_colums=2)

        choose_input_type = SettingsDropDown("Input Variante:", ["Text-Eingabe", "Manuelle Sprachaktivierung", "Automatische Sprachaktivierung", "Adaptives Gespräch"])
        choose_input_type.index_selected.connect(self.change_prompt_input)
        orion_input_settings.add_new_widget(choose_input_type, 0, 0, 1, 2)

        self.orion_input_settings_stack = QStackedWidget(self)

        # <editor-fold desc="seperate widgets">
        text_input_widget = QWidget()
        text_input_layout = QGridLayout(text_input_widget)

        manually_widget = QWidget()
        manually_widget_layout = QGridLayout(manually_widget)

        voice_regignition_widget = QWidget()
        voice_regignition_layout = QGridLayout(voice_regignition_widget)

        adaptive_speach_widget = QWidget()
        adaptive_speach_layout = QGridLayout(adaptive_speach_widget)
        # </editor-fold>

        # <editor-fold desc="setting ui elemnts">
        # generell for nearly all: hotkey
        self.all_change_hotkey_buttons = []
        for _ in range(3):
            new_element = HotKeyButton(self, current = self.gui_setting['push-to-talk'], added_text="für Sprachaktivierung: ")
            new_element.hotkey_changed.connect(lambda value: self.change_gui_settings("push-to-talk", value))
            self.all_change_hotkey_buttons.append(new_element)

        ## text
        text_lable = QLabel("Keine spezifischen Einstellungen für diesen Input Typ.")
        text_lable.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_input_layout.addWidget(text_lable)

        ## manuel
        self.manuel_silence_time_out_slider = SettingsSlider("Stille Time-Out", min_value=500, max_value=5_000, step_size=25, unit="ms")
        self.manuel_silence_time_out_slider.slider_value_changed.connect(lambda value: self.change_audio_settings("SilenceTimeout", value / 1000))

        manuel_test_volume_buton  = QPushButton("Mikrofon-Lautstärke testen", self)
        manuel_test_volume_buton.clicked.connect(self.open_volume_test)

        self.manuel_volume_threashold = SettingsSlider("Lautstärke-schwelle", min_value=10, max_value=7_000, step_size=50, unit="")
        self.manuel_volume_threashold.slider_value_changed.connect(lambda value: self.change_audio_settings("threshold", value))

        manually_widget_layout.addWidget(self.manuel_silence_time_out_slider, 0, 0)
        manually_widget_layout.addWidget(self.manuel_volume_threashold, 1, 0)
        manually_widget_layout.addWidget(manuel_test_volume_buton, 2, 0)
        manually_widget_layout.addWidget(self.all_change_hotkey_buttons[0], 3, 0)

        ## voice
        self.voice_silence_time_out_slider = SettingsSlider("Stille Time-Out", min_value=500, max_value=5_000, step_size=25, unit="ms")
        self.voice_silence_time_out_slider.slider_value_changed.connect(lambda value: self.change_audio_settings("SilenceTimeout", value / 1000))

        self.voice_volume_threashold = SettingsSlider("Lautstärke-schwelle", min_value=10, max_value=7_000, step_size=50, unit="")
        self.voice_volume_threashold.slider_value_changed.connect(lambda value: self.change_audio_settings("threshold", value))

        voice_test_volume_buton  = QPushButton("Mikrofon-Lautstärke testen", self)
        voice_test_volume_buton.clicked.connect(self.open_volume_test)

        self.voice_score_threashold = SettingsSlider("Spracherkennung Score-Schwelle", min_value=0, max_value=100, step_size=1, unit="%")
        self.voice_score_threashold.slider_value_changed.connect(lambda value: self.change_audio_settings("score_threshold", value / 100))

        voice_test_score_buton  = QPushButton("Score testen", self)
        voice_test_score_buton.clicked.connect(self.open_score_test)

        voice_regignition_layout.addWidget(self.voice_silence_time_out_slider, 0, 0)
        voice_regignition_layout.addWidget(self.voice_volume_threashold, 1, 0)
        voice_regignition_layout.addWidget(voice_test_volume_buton, 2, 0)
        voice_regignition_layout.addWidget(self.voice_score_threashold, 3, 0)
        voice_regignition_layout.addWidget(voice_test_score_buton, 4, 0)
        voice_regignition_layout.addWidget(self.all_change_hotkey_buttons[1], 5, 0)

        ## adaptive
        self.adaptive_silence_time_out_slider = SettingsSlider("Stille Time-Out", min_value=500, max_value=5_000, step_size=25, unit="ms")
        self.adaptive_silence_time_out_slider.slider_value_changed.connect(lambda value: self.change_audio_settings("SilenceTimeout", value / 1000))

        self.adaptive_volume_threashold = SettingsSlider("Lautstärke-schwelle", min_value=10, max_value=7_000, step_size=50, unit="")
        self.adaptive_volume_threashold.slider_value_changed.connect(lambda value: self.change_audio_settings("threshold", value))

        adaptive_test_volume_buton  = QPushButton("Mikrofon-Lautstärke testen", self)
        adaptive_test_volume_buton.clicked.connect(self.open_volume_test)

        self.adaptive_score_threashold = SettingsSlider("Spracherkennung Score-Schwelle", min_value=0, max_value=100, step_size=1, unit="%")
        self.adaptive_score_threashold.slider_value_changed.connect(lambda value: self.change_audio_settings("score_threshold", value / 100))

        adaptive_test_score_buton  = QPushButton("Score testen", self)
        adaptive_test_score_buton.clicked.connect(self.open_score_test)

        adaptive_speach_layout.addWidget(self.adaptive_silence_time_out_slider, 0, 0)
        adaptive_speach_layout.addWidget(self.adaptive_volume_threashold, 1, 0)
        adaptive_speach_layout.addWidget(adaptive_test_volume_buton, 2, 0)
        adaptive_speach_layout.addWidget(self.adaptive_score_threashold, 3, 0)
        adaptive_speach_layout.addWidget(adaptive_test_score_buton, 4, 0)
        adaptive_speach_layout.addWidget(self.all_change_hotkey_buttons[2], 5, 0)

        ## set saved values
        for item in list(self.orion_setting["oww_settings"].keys()):
            self.change_audio_settings(item, self.orion_setting["oww_settings"][item])

        # </editor-fold>

        # <editor-fold desc="add everything">
        self.orion_input_settings_stack.addWidget(text_input_widget)
        self.orion_input_settings_stack.addWidget(manually_widget)
        self.orion_input_settings_stack.addWidget(voice_regignition_widget)
        self.orion_input_settings_stack.addWidget(adaptive_speach_widget)

        orion_input_settings.add_new_widget(self.orion_input_settings_stack, 1, 0, 1, 2)

        if not self.orion_setting["use_stt"]:
            choose_input_type.select_option("Text-Eingabe")
        else:
            current_setting = self.orion_setting["speaking_recognition_mode"]
            if current_setting == "manually":
                choose_input_type.select_option("Manuelle Sprachaktivierung")
                self.orion_input_settings_stack.setCurrentIndex(1)
            elif current_setting == "voice":
                choose_input_type.select_option("Automatische Sprachaktivierung")
                self.orion_input_settings_stack.setCurrentIndex(2)
            elif current_setting == "smart":
                choose_input_type.select_option("Adaptives Gespräch")
                self.orion_input_settings_stack.setCurrentIndex(3)

        layout.addWidget(orion_input_settings)
        # </editor-fold>
        # </editor-fold>

        # <editor-fold desc="TTS">
        tts_categorie = SettingsCategorie("Text-To-Speach", max_colums=4)

        choose_tts_voice = SettingsDropDown("Stimme", ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"])
        choose_tts_voice.select_option(self.orion_setting["tts_voice"])
        choose_tts_voice.item_selected.connect(lambda value: self.voice_change("voice", value=value))

        voice_speed_slider = SettingsSlider("Geschwindigkeit", min_value=70, max_value=200, step_size=1, unit="%")
        voice_speed_slider.set_curent_value(self.orion_setting["tts_settings"]["speed"] * 100)
        voice_speed_slider.slider_value_changed.connect(lambda value: self.voice_change("speed", value / 100))

        voice_steps_slider = SettingsSlider("Schritte", min_value=4, max_value=16, step_size=1, unit="")
        voice_steps_slider.set_curent_value(self.orion_setting["tts_settings"]["total_steps"])
        voice_steps_slider.slider_value_changed.connect(lambda value: self.voice_change("total_steps", value))

        test_voice_button = QPushButton("Stimme Testen", self)
        test_voice_button.clicked.connect(lambda: self.voice_change("voice_test", None))

        tts_categorie.add_new_widget(choose_tts_voice, 0, 0, 1, 2)
        tts_categorie.add_new_widget(voice_speed_slider, 0, 2, 1, 2)
        tts_categorie.add_new_widget(voice_steps_slider, 1, 2, 1, 2)
        tts_categorie.add_new_widget(test_voice_button, 2, 0, 1, 4)

        layout.addWidget(tts_categorie)
        # </editor-fold>

        # <editor-fold desc="EXECUTION">
        # execution
        execution_categorie = SettingsCategorie("Ausführung", max_colums=4)

        python_execution_drop_down = SettingsDropDown("Python-Ausführung", ["Aus", "Bestätigung erforderlich", "sofort ausführen"])
        if self.orion_setting["allow_python_execution"]:
            if self.orion_setting["auto_python_execution"]:
                python_execution_drop_down.select_option("sofort ausführen")
            else:
                python_execution_drop_down.select_option("Bestätigung erforderlich")
        else:
            python_execution_drop_down.select_option("Aus")
        python_execution_drop_down.index_selected.connect(lambda index: self.change_execution("python", index))

        cmd_execution_drop_down = SettingsDropDown("CMD-Ausführung", ["Aus", "Bestätigung erforderlich", "sofort ausführen"])
        if self.orion_setting["allow_cmd_execution"]:
            if self.orion_setting["auto_cmd_execution"]:
                cmd_execution_drop_down.select_option("sofort ausführen")
            else:
                cmd_execution_drop_down.select_option("Bestätigung erforderlich")
        else:
            cmd_execution_drop_down.select_option("Aus")
        cmd_execution_drop_down.index_selected.connect(lambda index: self.change_execution("cmd", index))

        cmd_black_list = QLineEdit(self)
        cmd_black_list.setText("".join(f"{item} ; " for item in self.orion_setting["cmd_blacklist"]))
        cmd_black_list.setPlaceholderText("Liste eingeben...")
        cmd_black_list.setToolTip("Die CMD-Ausführung wird automatisch gestoppt, sobald einer der Befehle erkannt wurde.\nElemente mit ; trennen.")
        cmd_black_list.editingFinished.connect(lambda: self.change_execution("cmd_blacklist", cmd_black_list.text()))

        py_timeout_slider = SettingsSlider("Python Time-Out", min_value=10, max_value=300, unit="sec.")
        py_timeout_slider.set_curent_value(self.orion_setting["python_timeout"])
        py_timeout_slider.slider_value_changed.connect(lambda value: self.change_execution("python_timeout", value))

        cmd_timeout_slider = SettingsSlider("CMD Time-Out", min_value=10, max_value=300, unit="sec.")
        cmd_timeout_slider.set_curent_value(self.orion_setting["cmd_timeout"])
        cmd_timeout_slider.slider_value_changed.connect(lambda value: self.change_execution("cmd_timeout", value))


        tavily_drop_down = SettingsDropDown("Online Suchanfrage",["Verbieten", "Erlauben"])
        tavily_drop_down.select_option("Verbieten" if not self.orion_setting["allow_tavily_search"] else "Erlauben")
        tavily_drop_down.index_selected.connect(lambda index: self.change_execution("allow_tavily_search", bool(index)))

        tavily_type = SettingsDropDown("Suchtiefe", ["basic", "advanced", "fast"])
        tavily_type.select_option(self.orion_setting["tavily_settings"]["search_depth"])
        tavily_type.item_selected.connect(lambda index: self.change_execution("search_depth", index))

        tavily_forbidden_domains = QLineEdit(self)
        tavily_forbidden_domains.setPlaceholderText("Liste eingeben...")
        tavily_forbidden_domains.setText("".join(f"{item} ; " for item in self.orion_setting["tavily_settings"]["exclude_domains"]))
        tavily_forbidden_domains.setToolTip("Alle Domains die bei der Suchanfagre ignoriert werden.\nDomains mit ; trennen.")
        tavily_forbidden_domains.editingFinished.connect(lambda : self.change_execution("exclude_domains", tavily_forbidden_domains.text()))

        execution_categorie.add_new_widget(python_execution_drop_down, 0, 0, 1, 2)
        execution_categorie.add_new_widget(cmd_execution_drop_down, 0, 2, 1, 2)

        execution_categorie.add_new_widget(QLabel("  CMD-Blacklist:"), 1, 0, 1, 1)
        execution_categorie.add_new_widget(cmd_black_list, 1, 1, 1, 3)

        execution_categorie.add_new_widget(py_timeout_slider, 2, 0, 1, 2)
        execution_categorie.add_new_widget(cmd_timeout_slider, 2, 2, 1, 2)

        execution_categorie.add_new_widget(tavily_drop_down, 3, 0, 1, 2)
        execution_categorie.add_new_widget(tavily_type, 3, 2, 1, 2)

        execution_categorie.add_new_widget(QLabel("  Domain-Blacklist:"), 4, 0, 1, 1)
        execution_categorie.add_new_widget(tavily_forbidden_domains, 4, 1, 1, 3)

        layout.addWidget(execution_categorie)
        # </editor-fold>

        # <editor-fold desc="CHARACTER">
        # character
        character_categorie = SettingsCategorie("Charakteristik",len(self.all_character_templates))

        self.all_character_slider = {}

        for index, item in enumerate(list(self.current_character.keys())):
            new_element = SettingsSlider(item, min_value=0, max_value=100, unit="%")
            new_element.set_curent_value(int(self.current_character[item] * 100))
            new_element.slider_value_changed.connect(lambda value, i=item: self.change_character(i, value / 100))
            
            character_categorie.add_new_widget(new_element, index, 0, 1, len(self.all_character_templates))

            self.all_character_slider[item] = new_element

        total_characters = len(self.current_character.keys())

        title_lable = QLabel("Vorlagen")
        title_lable.setAlignment(Qt.AlignmentFlag.AlignCenter)
        character_categorie.add_new_widget(QLabel(""), total_characters, 0, 1, len(self.all_character_templates))
        character_categorie.add_new_widget(title_lable, total_characters + 1, 0, 1, len(self.all_character_templates))

        for index, template in enumerate(self.all_character_templates):
            template_text = template.replace(".json", "")
            new_button = QPushButton(template_text)
            new_button.clicked.connect(lambda _, t=template: self.set_character_template(t))
            character_categorie.add_new_widget(new_button, total_characters + 2, index, 1, 1)

        layout.addWidget(character_categorie)
        # </editor-fold>

        # <editor-fold desc="GENERLL">
        # Generell
        genrell_categorie = SettingsCategorie("Allgemein", max_colums=4)

        ai_volume_slider = SettingsSlider("KI Lautstärke", min_value=0, max_value=200, unit="%")
        ai_volume_slider.set_curent_value(self.gui_setting["orion_volume"] * 100)
        ai_volume_slider.slider_value_changed.connect(lambda value: self.change_gui_settings("orion_volume", value / 100))

        ui_volume_slider = SettingsSlider("GUI Lautstärke", min_value=0, max_value=200, unit="%")
        ui_volume_slider.set_curent_value(self.gui_setting["ui_sounds"] * 100)
        ui_volume_slider.slider_value_changed.connect(lambda value: self.change_gui_settings("ui_sounds", value / 100))

        choose_device = SettingsDropDown("Datenverarbeitung über", ["CPU", "GPU"])
        choose_device.select_option(self.orion_setting["tts_and_stt_device"].upper())
        choose_device.setToolTip("Verarbeitung des TTS und STT Modell über GPU oder CPU?")
        choose_device.item_selected.connect(lambda index: self.change_execution("tts_and_stt_device", index))

        cuda_dir_input = QLineEdit(self)
        cuda_dir_input.setPlaceholderText("Pfad eingeben...")
        cuda_dir_input.setText(self.orion_setting["cuda_dir"])
        cuda_dir_input.setToolTip("Der Pfad zu Cuda 12.9.")
        cuda_dir_input.editingFinished.connect(lambda : self.change_execution("cuda_dir", cuda_dir_input.text()))

        added_ai_role_input = QLineEdit(self)
        added_ai_role_input.setPlaceholderText("Weitere KI-Anweisung eingeben...")
        added_ai_role_input.setToolTip("Hinzuzufügende KI-Anweisungen. (Alles möglich)")
        added_ai_role_input.setText(self.orion_setting["added_ai_role"])
        added_ai_role_input.editingFinished.connect(lambda : self.change_execution("added_ai_role", added_ai_role_input.text()))


        genrell_categorie.add_new_widget(ai_volume_slider, 0, 0, 1, 2)
        genrell_categorie.add_new_widget(ui_volume_slider, 0, 2, 1, 2)

        genrell_categorie.add_new_widget(choose_device, 1, 0, 1, 4)

        genrell_categorie.add_new_widget(QLabel("Cuda Pfad"), 2, 0, 1, 1)
        genrell_categorie.add_new_widget(cuda_dir_input, 2, 1, 1, 3)


        genrell_categorie.add_new_widget(QLabel("KI-Anweisung"), 3, 0, 1, 1)
        genrell_categorie.add_new_widget(added_ai_role_input, 3, 1, 1, 3)

        layout.addWidget(genrell_categorie)
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
            if not self.orion_setting["speaking_recognition_mode"] == "smart":
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
            if self.orion_setting["speaking_recognition_mode"] == "smart":
                sd.stop(True)  # stop every current sd.play-output
                self.home_animation_worker.rms_over_time([0], 0.5, 1)

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
        # update GUI
        self.update_memory_entrys(init=False)
        self.update_history_bubbles(init=False)

        if len(exec_results) > 0:
            self.orion_worklfow.feed_multiple_data(exec_results)
        else:
            self.ui_current_status.emit(0)

    def hotkey_pressed(self, key):
        # manages all hot keys
        if key == "push-to-talk":
            if self.orion_setting["use_stt"]:
                self.stt_thread.start_recording()

    def update_end_of_conv(self, value):
        self.stt_thread.keep_conversation = not value

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

    def changes_has_made(self):
        if self.change_detected:
            return

        self.change_detected = True
        self.show_toast("Bitte das Programm neustarten um Änderungen zu übernehmen.", 10_000)
        self.restart_app_button.show()

    def update_settings(self):
        self.save_new_settings_func(self.orion_setting)

    @has_something_changed
    def change_ai_provider(self, new_provider):
        if "gemini" in new_provider.lower():
            self.orion_setting["llm_provider"] = "google"
            self.gemini_version_drop_down.select_option("gemini-3.5-flash-lite")
            self.change_llm_modell("gemini-3.5-flash-lite")
        elif "ollama" in new_provider.lower():
            self.orion_setting["llm_provider"] = "ollama"
            self.ollama_version_drop_down.select_option("llama3.1:latest")
            self.change_llm_modell("llama3.1:latest")
        else:
            self.orion_setting["llm_provider"] = new_provider

    @has_something_changed
    def change_llm_modell(self, new_modell):
        if self.orion_setting["llm_provider"] == "gemini" and "gemini" in new_modell.lower():
            self.orion_setting["llm_version"] = new_modell
        elif self.orion_setting["llm_provider"] == "ollama" and "gemini" not in new_modell.lower():
            self.orion_setting["llm_version"] = new_modell

    @has_something_changed
    def prompt_items_changes(self, type_changed, value):
        self.orion_setting[type_changed] = value

    @has_something_changed
    def change_prompt_input(self, new_prompt_input):
        self.orion_input_settings_stack.setCurrentIndex(new_prompt_input)

        if new_prompt_input == 0:
            self.orion_setting["use_stt"] = False
        elif new_prompt_input == 1:
            self.orion_setting["use_stt"] = True
            self.orion_setting["speaking_recognition_mode"] = "manually"
        elif new_prompt_input == 2:
            self.orion_setting["use_stt"] = True
            self.orion_setting["speaking_recognition_mode"] = "voice"
        elif new_prompt_input == 3:
            self.orion_setting["use_stt"] = True
            self.orion_setting["speaking_recognition_mode"] = "smart"

    @has_something_changed
    def change_audio_settings(self, type_changed, value):
        self.orion_setting["oww_settings"][type_changed] = value

        if type_changed == "SilenceTimeout":
            self.manuel_silence_time_out_slider.set_curent_value(value * 1000)
            self.voice_silence_time_out_slider.set_curent_value(value * 1000)
            self.adaptive_silence_time_out_slider.set_curent_value(value * 1000)
        elif type_changed == "threshold":
            self.manuel_volume_threashold.set_curent_value(value)
            self.voice_volume_threashold.set_curent_value(value)
            self.adaptive_volume_threashold.set_curent_value(value)
        elif type_changed == "score_threshold":
            self.voice_score_threashold.set_curent_value(value * 100)
            self.adaptive_score_threashold.set_curent_value(value * 100)

    def open_volume_test(self):
        dialog_window = VolumeTest(self)

        self.stt_thread.send_rms_to_volume_test(True, dialog_window.recv_rms)
        dialog_window.apply_new_volume.connect(lambda value: self.change_audio_settings("threshold", value))

        dialog_window.exec()

        self.stt_thread.send_rms_to_volume_test(False)

    def open_score_test(self):
        dialog_window = ScoreTest(self)

        self.stt_thread.send_score_to_test(True, dialog_window.recv_score)
        dialog_window.apply_new_score.connect(lambda value: self.change_audio_settings("score_threshold", value / 100))

        dialog_window.exec()

        self.stt_thread.send_score_to_test(False)

    @has_something_changed
    def voice_change(self, type_changed, value):
        if type_changed == "voice":
            self.orion_setting["tts_voice"] = value

        elif type_changed == "speed":
            self.orion_setting["tts_settings"]["speed"] = value

        elif type_changed == "total_steps":
            self.orion_setting["tts_settings"]["total_steps"] = value

        elif type_changed == "voice_test":

            sentence = random.choice(self.gui_setting["voice_test_sentences"])

            sd.stop(True)
            self.ui_current_status.emit(2)
            self.play_sound("orion_finished")
            tts_response = self.tts_func(sentence, specific_voice_style=self.orion_setting["tts_voice"])
            self.process_orion_output({"content": sentence, "no_safe":True}, tts_response[0], tts_response[1])

    @has_something_changed
    def change_execution(self, type_changed, value):
        if type_changed == "python":
            if value == 0:
                self.orion_setting["allow_python_execution"] = False
            else:
                self.orion_setting["allow_python_execution"] = True
                if value == 1:
                    self.orion_setting["auto_python_execution"] = False
                else:
                    self.orion_setting["auto_python_execution"] = True
        elif type_changed == "cmd":
            if value == 0:
                self.orion_setting["allow_cmd_execution"] = False
            else:
                self.orion_setting["allow_cmd_execution"] = True
                if value == 1:
                    self.orion_setting["auto_cmd_execution"] = False
                else:
                    self.orion_setting["auto_cmd_execution"] = True
        elif type_changed == "cmd_blacklist":
            all_values = value.split(";")
            valid_values = []
            for v in all_values:
                if v.strip():
                    valid_values.append(v.strip())
            self.orion_setting["cmd_blacklist"] = valid_values
        elif type_changed in ["cmd_timeout", "python_timeout", "allow_tavily_search", "cuda_dir", "added_ai_role"]:
            self.orion_setting[type_changed] = value
        elif type_changed == "search_depth":
            self.orion_setting["tavily_settings"]["search_depth"] = value
        elif type_changed == "exclude_domains":
            all_values = value.split(";")
            valid_values = []
            for v in all_values:
                if v.strip():
                    valid_values.append(v.strip())
            self.orion_setting["tavily_settings"]["exclude_domains"] = valid_values
        elif type_changed == "tts_and_stt_device":
            self.orion_setting["tts_and_stt_device"] = value.lower()

    def change_gui_settings(self, type_changed, value):
        if type_changed in ["orion_volume", "ui_sounds", "push-to-talk"]:
            self.gui_setting[type_changed] = value

        if type_changed == "push-to-talk":
            for i in range(3):
                self.all_change_hotkey_buttons[i].set_hotkey(value)

        # save gui file
        with open(self.gui_settings_file, "w", encoding="utf-8") as f:
            json.dump(self.gui_setting, f, indent=4, ensure_ascii=False)

    @has_something_changed
    def change_character(self, type_changed, value):

        self.current_character[type_changed] = value

        self.save_new_character_func(self.current_character)

    @has_something_changed
    def set_character_template(self, template):
        with open(rf"{self.working_dir}{self.gui_setting['character_templates']}{template}", "r", encoding="utf-8") as f:
            new_character = json.load(f)

        all_current_character_slider = list(self.all_character_slider.keys())
        new_character_keys = list(new_character.keys())

        for c in all_current_character_slider:
            if c in new_character_keys:
                self.all_character_slider[c].set_curent_value(new_character[c] * 100)

        self.current_character = new_character
        self.save_new_character_func(self.current_character)

    def restart_app(self):
        self.restart_app_func()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # sample functions
    def ai_res(text: str, prompt_type: str):
        time.sleep(.5)
        test_comp = {"content":"Hallo, dies ist kein KI generierter Inhalt, sonder lediglich ein Test. Bitte starte die Datei: Mein punkt p y um das programm korrekt zu starten."}
        return test_comp

    def tts_res(text: str, specific_voice_style: str):
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

    def save_new_settings(n_settings):
        with open(rf"./assets/settings.json", "w", encoding="utf-8") as settings_file:
            json.dump(n_settings, settings_file, ensure_ascii=False, indent=4)

    def get_character():
        with open(rf"./assets/json_files/character_gemini.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def save_character(n_character):
        with open(rf"./assets/json_files/character_gemini.json", "w", encoding="utf-8") as f:
            json.dump(n_character, f, ensure_ascii=False, indent=4)

    def restart_app():
        QProcess.startDetached(sys.executable, sys.argv)
        QApplication.quit()

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
        predict_activation_word_func=lambda *args: [False, 0.1],
        get_current_chat_history=get_history,
        get_current_memory=get_memory,
        save_edited_memory_func=save_edited_memory,
        save_new_settings_func=save_new_settings,

        save_new_character_func=save_character,
        get_current_character_func=get_character,
        restart_app_func = restart_app
    )

    window.showMaximized()

    sys.exit(app.exec())