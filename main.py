import json
import os
import sys

with open(r"./assets/settings.json", "r", encoding="utf-8") as file:
    settings = json.load(file)

if settings.get("tts_and_stt_device") == "gpu":
    print("Setting everything up for GPU...")

    # adding cuda PATH
    cuda_path = settings.get("cuda_dir")
    if os.path.exists(cuda_path):
        os.add_dll_directory(cuda_path)
        os.environ["PATH"] = cuda_path + os.pathsep + os.environ["PATH"]

    # Adding enviroment PATH (/venv/)
    venv_site_packages = os.path.abspath(r".\venv\Lib\site-packages")
    nvidia_dir = os.path.join(venv_site_packages, "nvidia")

    if os.path.exists(nvidia_dir):
        for root, dirs, files in os.walk(nvidia_dir):
            if "bin" in dirs:
                bin_path = os.path.join(root, "bin")
                os.add_dll_directory(bin_path)
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]

import onnxruntime as ort

if settings.get("tts_and_stt_device") == "gpu":

    # force remove GPU-Instance
    _original_init = ort.InferenceSession.__init__

    def _gpu_forced_init(self, *args, **kwargs):
        sess_options = kwargs.get("sess_options", ort.SessionOptions())
        sess_options.log_severity_level = 3

        kwargs["providers"] = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        _original_init(self, *args, **kwargs)

    ort.InferenceSession.__init__ = _gpu_forced_init

import random
import time
from threading import Thread
from pydantic import BaseModel
import os
from datetime import datetime
from rich import print
import re
import sounddevice as sd
import numpy as np
from openwakeword.model import Model
import openwakeword
from faster_whisper import WhisperModel
import soundfile as sf
import queue
from tavily import TavilyClient
from supertonic import TTS
import tempfile
import subprocess
import orion_gui
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QProcess
from assets.ai_clients import google_gemini, ollama_llm

class AgentOutput(BaseModel):
    further_process: str
    content: str
    memory: str
    pythonCode: str
    end_of_conversation: bool
    summary: str
    online_search: str
    cmd_execution: str

# noinspection PyTypeChecker,PyMethodMayBeStatic
class ORION_GemAI:
    def __init__(self):
        self.working_dir: str = os.path.abspath("./")

        # files and keys
        self.api_key_file: str = rf"{self.working_dir}/api_key.json"
        self.json_files: str = rf"{self.working_dir}/assets/json_files/"
        self.tts_voice_files: str = rf"{self.working_dir}/assets/local_voice/"

        # LLM
        self.llm_provider = None
        self.LLM_client = None

        # history and memory
        self.chat_history = []
        self.permanent_memory: list = []
        self.permanent_memory_file: str = f"{self.json_files}/memory.json"
        self.chat_history_file: str = f"{self.json_files}/chat_history.json"

        # characteristics
        self.characteristics_json_files: str = rf"{self.json_files}/character.json"
        self.characteristics_dict: dict = {}

        # logging
        self.silent_log:bool = False

        # tts
        self.playback_worker_thread = None
        self.run_playback_worker_thread = True

        # settings
        self.settings_file: str = rf"{self.working_dir}/assets/settings.json"
        self.all_settings: dict = {}
        self.use_tts = True
        self.use_stt = False
        self.activation_sound = True
        self.send_history = False
        self.send_memory = False
        self.max_memorys: int = 0
        self.max_history: int = 0

        self.end_of_conversation: bool = True

        # voice tts
        self.tts_voice = None
        self.syn_config = None
        self.tts_voice_style = "M1"
        self.tts_settings: dict = {}
        self.tts_and_stt_device = "cpu"

        # voice stt
        self.oww_model = None
        self.activation_word_file:str = f"{self.working_dir}/assets/stt/Orion.onnx"
        self.active_signal:list = ["Ja? ", "Joo", "Ich Höre ", "Yup"]
        self.word_detected: bool = False
        self.messages_transcript: list = [False, ""]
        self.listen_and_transcript_thread_value = None
        self.stt_audio_queue = queue.Queue()
        self.oww_timeout: list = [1.5, time.time()]
        self.whisper_model = None
        self.whisper_model_size = "tiny"
        self.oww_settings = {"InitialSilenceTimeout":10, "SilenceTimeout":2, "threshold":30}
        self.speaking_recognition_mode = "smart" # possible: smart, voice, manually

        # console running
        self.init_console = True
        self.response = ""
        self.processed_response = ""

        # python
        self.auto_python_execution: bool = False # USE WITH CAUTION!
        self.py_timeout = 60
        self.allow_python_execution = True

        # searching
        self.allow_tavily_search: bool = True
        self.tavily_api_key: str = ""
        self.tavily = None
        self.tavily_settings: dict = {}

        # cmd
        self.allow_cmd_execution: bool = False
        self.cmd_auto_execution: bool = False
        self.cmd_blacklist: list = []
        self.cmd_timeout: int = 10

        # multiple promps at once
        self.total_promps = ""

        # inteface
        self.execute_interface = "console"

        # gui
        self.gui_class = None
        self.pyside_app = None

    """LOGGING"""

    def get_current_timestamp(self):
        return datetime.now().strftime("%d-%m-%y %H:%M:%S")

    def output(self, text: str, typ: str):
        # text: str = any str
        # typ str = "log", "warn", "error"

        # get current Time and format it
        cTime = self.get_current_timestamp()
        # object
        obj: dict = {
            "type": typ,
            "time": cTime,
            "content": text
        }
        # log in console
        if not self.silent_log:
            print(f"[{obj['type']}] [{obj['time']}] {obj['content']}")
        else:
            self.console_print(f"[{obj['type']}] [{obj['time']}] {obj['content']}", "dim")

    def console_print(self, text: str, color: str):
        print(f"[{color}]{text}[/{color}]")

    """INIT"""

    def initialise_all(self, start_thread=True):
        self.output("Initialise ORION...", "log")
        self.load_api_keys()
        self.load_settings()
        self.init_LLM()
        self.load_memory()
        self.load_chat_history()
        if self.use_tts:
            self.init_tts()
        if self.use_stt:
            self.init_stt(start_thread)
        if self.allow_tavily_search:
            self.init_tavily_search()
        if self.execute_interface == "gui":
            self.init_gui()

        self.output("Successfully initialised ORION", "log")

    def init_LLM(self):
        self.output(f"Init LLM ...", "log")

        if self.llm_provider == "google":
            self.LLM_client = google_gemini.GeminiLLM(
                output_func=self.output,
                orion_settings=self.all_settings
            )
        elif self.llm_provider == "ollama":
            self.LLM_client = ollama_llm.ollama_client(
                output_func=self.output,
                orion_settings=self.all_settings
            )
        else:
            self.output(f"Provider {self.llm_provider} not found. Switching to google gemini", "error")
            self.LLM_client = google_gemini.GeminiLLM(
                output_func=self.output,
                orion_settings=self.all_settings
            )

    def load_settings(self):
        self.output("Loading settings...", "log")
        with open(self.settings_file, "r", encoding="utf-8") as f:
            all_settings: dict = json.load(f)

            self.llm_provider: str = all_settings["llm_provider"]

            self.send_history: bool = all_settings["send_history"]
            self.send_memory: bool = all_settings["send_memory"]

            self.max_memorys = all_settings["memory_max_items"]
            self.max_history = all_settings["history_max_items"]

            self.auto_python_execution: bool = all_settings["auto_python_execution"]
            self.py_timeout = all_settings["python_timeout"]
            self.allow_python_execution = all_settings["allow_python_execution"]

            self.use_stt: bool = all_settings["use_stt"]
            self.oww_settings: dict = all_settings["oww_settings"]
            self.activation_sound: bool = all_settings["activation_sound"]
            self.active_signal: list = all_settings["active_words"]
            self.whisper_model_size = all_settings["whisper_model"]
            self.speaking_recognition_mode: str = all_settings["speaking_recognition_mode"]

            self.use_tts: list = all_settings["use_tts"]
            self.tts_voice_style: str = all_settings["tts_voice"]
            self.tts_settings: dict = all_settings["tts_settings"]

            self.tts_and_stt_device: str = all_settings["tts_and_stt_device"]

            self.allow_tavily_search: bool = all_settings["allow_tavily_search"]
            self.tavily_settings: dict = all_settings["tavily_settings"]

            self.allow_cmd_execution: bool = all_settings["allow_cmd_execution"]
            self.cmd_auto_execution: bool = all_settings["auto_cmd_execution"]
            self.cmd_blacklist: list = all_settings["cmd_blacklist"]
            self.cmd_timeout: int = all_settings["cmd_timeout"]

            self.execute_interface: str = all_settings["execute_interface"]

            self.all_settings: dict = all_settings

    def load_api_keys(self):
        self.output("Loading api key...", "log")
        with open(self.api_key_file, "r", encoding="utf-8") as f:
            all_keys = json.load(f)
            self.tavily_api_key = all_keys["tavily"]

    def init_tavily_search(self):
        self.output("Loading tavily...", "log")
        self.tavily = TavilyClient(api_key=self.tavily_api_key)

    def init_gui(self):
        self.output("Init GUI...", "log")

        self.pyside_app = QApplication(sys.argv)
        self.gui_class = orion_gui.MainWindow(
            ai_response_func=self.send_message,
            tts_func=lambda text, specific_voice_style: self.tts_say_text(text, deactivate_palyback_worker=True, return_wave=True, specific_voice_style=specific_voice_style),

            py_execution_func=self.run_python_script,
            cmd_execution_func=self.run_cmd_commands,
            online_serach_func=self.tavily_search,

            save_memory_func=self.save_new_memory,
            save_chat_history_func=self.save_chat_history,
            send_multiple_messages_func=self.send_multiple_mesages,

            record_audio_func=self.record_audio,
            transcript_audio_func=self.transcript_audio,

            predict_activation_word_func = self.predict_activation_word,
            get_current_chat_history = lambda : self.chat_history,
            get_current_memory= self.get_and_reload_memory,
            save_edited_memory_func=self.save_edited_new_memory,
            save_new_settings_func = self.save_new_settings,

            save_new_character_func = self.LLM_client.save_character,
            get_current_character_func = lambda : self.LLM_client.characteristics_dict,
            restart_app_func = self.restart_app
        )

    """TTS"""

    def init_tts(self):
        self.output("Loading Voice...", "log")
        self.tts_voice = TTS(auto_download=True, model_dir=self.tts_voice_files + "/")
        self.syn_config = self.tts_voice.get_voice_style(voice_name=self.tts_voice_style)

    def tts_say_text(self, text, wait_till_finish: bool = False, return_wave: bool = False, deactivate_palyback_worker: bool = False,
                     specific_voice_style = ""):
        self.output("Generating Voice...", "log")
        sentences = re.split(r'(?<=[.!?])\s+', text)
        audio_queue = queue.Queue()

        def playback_worker():
            while self.run_playback_worker_thread:
                item = audio_queue.get()
                if item is None:
                    break
                data, fs, t = item

                self.output("Playing Voice...", "log")
                sd.play(data, fs)
                sd.wait()
                audio_queue.task_done()
                time.sleep(0.1)

        if not deactivate_palyback_worker:
            if self.playback_worker_thread is not None:
                self.run_playback_worker_thread = False
                while self.playback_worker_thread.is_alive():
                    time.sleep(0.1)
                self.run_playback_worker_thread = True
            self.playback_worker_thread = Thread(target=playback_worker, daemon=True)
            self.playback_worker_thread.start()

        if not text.strip():
            self.output("TTS: Text is empty.", "warning")
            return None

        if specific_voice_style != "":
            use_syn_config = self.tts_voice.get_voice_style(voice_name=specific_voice_style)
        else:
            use_syn_config = self.syn_config

        wav, sr = self.tts_voice.synthesize(
            text=text,
            lang="na",
            voice_style=use_syn_config,
            total_steps=self.tts_settings["total_steps"],
            speed=self.tts_settings["speed"]
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            self.tts_voice.save_audio(wav, temp_path)
            data, fs = sf.read(temp_path, dtype='float32')

            if return_wave:
                return [data, fs]

            audio_queue.put((data, fs, text))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if wait_till_finish:
            audio_queue.join()

        audio_queue.put(None)
        return None

    def play_audio(self, file_path):
        data, fs = sf.read(file_path, dtype='float32')
        sd.play(data, fs)
        sd.wait()

    """STT"""

    def init_stt(self, start_thread=True):
        self.output(f"Init STT with model size {self.whisper_model_size}...", "log")
        whisper_model_device = "cuda" if self.tts_and_stt_device == "gpu" else "cpu"
        openwakeword.utils.download_models()
        self.oww_model = Model(wakeword_models=[self.activation_word_file], inference_framework="onnx")
        self.whisper_model = WhisperModel(self.whisper_model_size, device=whisper_model_device, compute_type="int8")

        self.output(f"Speaking recognition mode: {self.speaking_recognition_mode}", "log")
        if start_thread:
            self.listen_and_transcript_thread_value = Thread(target=self.listen_and_transcript_thread)
            self.listen_and_transcript_thread_value.start()

    def listen_and_transcript_thread(self):
        self.output("STT Model is running in background...", "log")

        with sd.InputStream(samplerate=16000, channels=1, blocksize=1280, dtype='int16', callback=self.audio_callback):
            while True:

                if self.speaking_recognition_mode == "manually":
                    time.sleep(0.1)
                    continue

                elif self.word_detected:
                    # stop all sounds
                    sd.stop()
                    self.oww_timeout[1] = time.time()

                    # play sound
                    if self.activation_sound and self.end_of_conversation:
                        self.tts_say_text(random.choice(self.active_signal), True)

                    audio_data = self.record_audio(-1)
                    if audio_data is False:
                        with self.stt_audio_queue.mutex:
                            self.stt_audio_queue.queue.clear()
                        self.word_detected = False
                        continue

                    user_transcript = self.transcript_audio(audio_data)

                    if user_transcript == "":
                        with self.stt_audio_queue.mutex:
                            self.stt_audio_queue.queue.clear()
                        self.word_detected = False
                        continue

                    self.messages_transcript = [True, user_transcript]

                    with self.stt_audio_queue.mutex:
                        self.stt_audio_queue.queue.clear()

                    self.word_detected = False
                    self.oww_timeout[1] = time.time()

                elif not self.end_of_conversation:

                    try:
                        chunk = self.stt_audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                    self.word_detected = rms > (self.oww_settings["threshold"] * 4)
                    if self.word_detected:
                        self.output("Adaptive Input detected", "log")

                else:

                    try:
                        audio_chunk = self.stt_audio_queue.get(timeout=0.1)

                        self.word_detected, _ = self.predict_activation_word(audio_chunk)
                    except queue.Empty:
                        continue

    def predict_activation_word(self, audio_chunk):
        prediction = self.oww_model.predict(audio_chunk)

        for model_name, score in prediction.items():

            if time.time() - self.oww_timeout[1] <= self.oww_timeout[0]:
                return False, score

            if score > self.oww_settings["score_threshold"]:
                self.output("Activation Word detected", "log")
                return True, score

        return False, 0

    def audio_callback(self,indata, frames, time_info, status):
        if status:
            self.output(f"Status from OWW: {status}", "log")
        self.stt_audio_queue.put(np.frombuffer(indata, dtype=np.int16).copy())

    def record_audio(self, duration: int = -1):
        if duration > 0:
            self.output(f"Recording audio for {duration} sec....", "log")
            fs = 16_000

            audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
            sd.wait()

            audio_flatten = audio_data.flatten()

            return audio_flatten
        else:
            self.output(f"Recording audio adaptive...", "log")
            recorded_chunks = []
            speech_started = False
            silence_chunks_count = 0

            chunk_duration = 1280 / 16_000 # ca. 0.08 sec @ 16khz
            max_silence_chunks = int(self.oww_settings["SilenceTimeout"] / chunk_duration)

            with self.stt_audio_queue.mutex:
                self.stt_audio_queue.queue.clear()

            start_time = time.time()

            while True:
                if not speech_started and (time.time() - start_time > self.oww_settings["InitialSilenceTimeout"]):
                    self.output("Timeout: No speech detected", "log")
                    return False

                try:
                    chunk = self.stt_audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # get how loud
                rms = np.sqrt(np.mean(chunk.astype(np.float32)**2))

                # check if loud enough
                is_speech = rms > self.oww_settings["threshold"]

                if not speech_started:
                    if is_speech:
                        speech_started = True
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

    def transcript_audio(self, audio_data):
        self.output("Transcript audio...", "log")

        segments, info = self.whisper_model.transcribe(audio_data, language="de")
        text = "".join([segment.text for segment in segments])
        self.output(f"Transcript: {text.strip()}", "log")
        return text.strip()

    """PROMPT"""

    def get_prompt_by_type(self, prompt, prompt_type: str = "user", no_memorys = False, no_history=False):

        self.save_chat_history(prompt, prompt_type)

        full_prompt: str = ""
        if prompt_type == "user":
            full_prompt = f"""
                            Zeit: {self.get_current_timestamp()} 
                            Nachricht des Nutzers: {prompt}
                            """
        elif prompt_type == "pythonCode":
            full_prompt = f"""
                            DIES IST KEINE NACHRICHT DES NUTZERS
                            Der Python Code aus deiner Letzten Nachricht wurde Ausgeführt. Dies sind die Ergebnisse
                            Zeit: {self.get_current_timestamp()} 
                            Ausgabe: {prompt}
                            """
        elif prompt_type == "further_process":
            full_prompt = f"""
                            DIES IST KEINE NACHRICHT DES NUTZERS
                            Du hast further Proces in deiner Letzten Nachricht aktiviert
                            Zeit: {self.get_current_timestamp()} 
                            further_process Wert: {prompt}
                            """
        elif prompt_type == "transcript":
            full_prompt = f"""
                                   Die Folgende Nachricht wurde Transkribiert vom Audio des Nutzers
                                   Zeit: {self.get_current_timestamp()} 
                                   Transcriptions des Nutzers: {prompt}
                                   """
        elif prompt_type == "search_result":
            full_prompt = f"""
                                Folgende Nachricht enthält Ergebnisse aus einer Suchanfrage
                                Zeit: {self.get_current_timestamp()} 
                                Suchanfrage: {prompt[0]}
                                Ergebnisse: {prompt[1]}
                                """
        elif prompt_type == "cmd":
            full_prompt = f"""
                                Rückgabewert aus der CMD Execution
                                Zeit: {self.get_current_timestamp()} 
                                Ausgabe: {prompt}
                                """
        else:
            full_prompt = f"""
                            Prompt-Typ: {prompt_type}
                            Zeit: {self.get_current_timestamp()} 
                            Prompt: {prompt}
                            """

        # add history
        if self.send_history and not no_history:
            self.chat_history.reverse()
            full_prompt += f"\nChat History: {self.chat_history[0:self.max_history]}"
            self.chat_history.reverse()

        if self.send_memory and not no_memorys:
            self.permanent_memory.reverse()
            full_prompt += f"\nErinnerungen: {self.permanent_memory[0:self.max_memorys]}"
            self.permanent_memory.reverse()

        return full_prompt

    def send_message(self, prompt, prompt_type: str = "user", promp_is_full_prompt=False):
        self.output(f"Requesting AI on type: {prompt_type}", "log")

        if promp_is_full_prompt:
            full_prompt = prompt
        else:
            full_prompt = self.get_prompt_by_type(prompt, prompt_type)

        response = self.LLM_client.send_message(full_prompt,)
        return response

    def send_multiple_mesages(self, prompt, prompt_type: str = "user", is_final = False):
        if not is_final:
            f_p = self.get_prompt_by_type(prompt, prompt_type, no_memorys=True, no_history=True)
            self.total_promps += f"\n{f_p}"
            return None
        else:
            final_prompt = self.total_promps
            if self.send_history:
                self.chat_history.reverse()
                final_prompt += f"\nChat History: {self.chat_history[0:self.max_history]}"
                self.chat_history.reverse()
            if self.send_memory:
                self.permanent_memory.reverse()
                final_prompt += f"\nErinnerungen: {self.permanent_memory[0:self.max_memorys]}"
                self.permanent_memory.reverse()
            self.total_promps = ""

            return self.send_message(final_prompt, "Multiple", promp_is_full_prompt=True)

    """LOCAL STORAGE"""

    def save_new_memory(self, n_memory):
        self.output("Saving new Memory", "log")
        if not n_memory.strip():
            return

        self.permanent_memory.append({"time": self.get_current_timestamp(), "content":n_memory})
        with open(self.permanent_memory_file, "w", encoding="utf-8") as f:
            json.dump(self.permanent_memory, f, indent=4, ensure_ascii=False)

    def save_edited_new_memory(self, n_memory):
        self.output("Saving edited Memory", "log")

        self.permanent_memory = n_memory
        with open(self.permanent_memory_file, "w", encoding="utf-8") as f:
            json.dump(self.permanent_memory, f, indent=4, ensure_ascii=False)

    def load_memory(self):
        self.output("Loading Memory...", "log")

        if not os.path.exists(self.permanent_memory_file):
            return

        with open(self.permanent_memory_file, "r", encoding="utf-8") as f:
            self.permanent_memory = json.load(f)

    def save_chat_history(self, n_chat, role):
        self.output("Saving new Chat History", "log")

        self.load_chat_history()

        self.chat_history.append({"time": self.get_current_timestamp(), "content":n_chat, "role":role})
        with open(self.chat_history_file, "w", encoding="utf-8") as f:
            json.dump(self.chat_history, f, indent=4, ensure_ascii=False)

    def load_chat_history(self):
        self.output("Loading Chat History...", "log")

        if not os.path.exists(self.chat_history_file):
            return

        with open(self.chat_history_file, "r", encoding="utf-8") as f:
            self.chat_history = json.load(f)

    def save_new_settings(self, new_settings):
        with open(self.settings_file, "w", encoding="utf-8") as f:
            json.dump(new_settings, f, indent=4, ensure_ascii=False)
        self.load_settings()

    def get_and_reload_memory(self):
        self.load_memory()
        return self.permanent_memory

    """ONLINE SEARCH"""

    def tavily_search(self, search_item):
        self.output(f"Searchin for: {search_item}", "log")
        if not self.allow_tavily_search:
            return "Not allowed"

        response = self.tavily.search(
            query=search_item,
            max_results=self.tavily_settings["max_results"],
            search_depth=self.tavily_settings["search_depth"],
            exclude_domains=self.tavily_settings["exclude_domains"],
            topic=self.tavily_settings["topic"],
            time_range=self.tavily_settings["time_range"],
            auto_parameters=self.tavily_settings["auto_parameters"]
        )

        return response

    """CMD EXECUTION"""

    def run_cmd_commands(self, command):

        self.save_chat_history(command, "cmd_script")

        if not self.allow_cmd_execution:
            return "FEHLER: CMD Execution ist abgeschaltet"

        if not self.cmd_auto_execution:
            self.console_print(f"Möchtest du folgenden Befehl ausführen?\n {command}", "yellow")
            if input("(Y/N) ").upper() == "N":
                return "FEHLER: Der Befehl wurde vom Nutzer abgelehnt"

        self.output(f"Running CMD Command: {command}", "log")

        if any(bad in command.lower() for bad in self.cmd_blacklist):
            self.output(f"Blacklist detected: {command}", "warn")
            return "FEHLER: Befehl aus Sicherheitsgründen abgelehnt."

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.cmd_timeout,
                encoding="cp850"
            )

            output = result.stdout.strip()
            errors = result.stderr.strip()

            if result.returncode == 0:
                return output if output else "Befehl erfolgreich ausgeführt."
            else:
                return f"Fehler (Code {result.returncode}): {errors if errors else output}"
        except subprocess.TimeoutExpired:
            return f"FEHLER: Befehl hat nach {self.cmd_timeout} Sekunden gedauert und wurde abgebrochen (evtl. wartet er auf Tastatureingabe)."
        except Exception as e:
            return f"FEHLER beim Ausführen: {str(e)}"

    """PYTHON EXECUTION"""

    def run_python_script(self, script):
        python_response = []

        self.save_chat_history(script, "python_code_script")

        try:

            if "subprocess.Popen()" in script or "subprocess.Popen([sys.executable, filename])" in script:
                if "subprocess.Popen()" in script:
                    script = script.replace(
                        "subprocess.Popen()",
                        "subprocess.Popen([sys.executable, filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)"
                    )
                else:
                    script = script.replace(
                        "subprocess.Popen([sys.executable, filename])",
                        "subprocess.Popen([sys.executable, filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)"
                    )
                self.output(f"Apply Auto-Fix: subprocess.Popen(\\[sys.executable, filename\\], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)", "warn")
                print(script)


            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py", encoding="utf-8") as tmp:
                tmp.write(script)
                temp_path = tmp.name

            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.py_timeout,
                encoding="utf-8",
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )

            if result.stdout:
                python_response.append(result.stdout.strip())

            if result.returncode != 0:
                python_response.append(f"Fehler (Code {result.returncode}): {result.stderr.strip()}")
                self.console_print(f"Ausgabe: {python_response}", color="red")
            else:
                self.console_print(f"Ausgabe: {python_response}", color="green")

        except subprocess.TimeoutExpired:
            python_response.append(f"FEHLER: Skript-Ausführung hat das Zeitlimit von {self.py_timeout} Sekunden überschritten.")
            self.console_print(f"Ausgabe: {python_response}", color="red")

        except Exception as e:
            python_response.append(f"Es trat ein Fehler auf: {e}")
            self.console_print(f"Ausgabe: {python_response}", color="red")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        return python_response

    """RUN"""

    def run_in_console(self, no_while_loop: bool = False):
        if self.init_console:
            self.silent_log = True
            self.response = self.send_message("Begrüße den Nutzer, basierend auf deinem gespeicherten Informationen angemessen. Setze speziell für diesen prompt end_of_conversation auf True", "system")
            self.processed_response = ""
            self.init_console = False
            self.end_of_conversation: bool = True

        def check_if_empty(item: str):
            return item.upper() in ["", "NONE", "NULL"]

        while True:

            self.console_print(f"\n{self.response['content']}\n", "italic cyan")
            try:
                self.save_chat_history(self.response["content"], "ai")
                # speaking_recognition_mode
                if self.speaking_recognition_mode == "smart":
                    self.end_of_conversation = self.response["end_of_conversation"]
                else:
                    self.end_of_conversation = True

            except Exception as e:
                self.output(f"An error occur: {e}", "error")

            if self.use_tts:
                sd.stop()
                self.tts_say_text(self.response['content'])

            if self.response["memory"] not in ["", "NONE", "NULL"]:
                self.save_new_memory(self.response["memory"])

            if check_if_empty(self.response["pythonCode"]) and check_if_empty(self.response["further_process"]) and check_if_empty(self.response["online_search"]):

                # Fallback when using stt. Input() is not needed.
                if self.use_stt:
                    if self.speaking_recognition_mode == "manually":
                        self.console_print(f"Press -enter- to activate voice input", "green")
                        input()
                        self.console_print(f"Listening...", "green")

                        sd.stop()

                        while True:
                            audio_data = self.record_audio(-1)
                            if audio_data is not False:
                                break

                        self.messages_transcript[1] = self.transcript_audio(audio_data)

                        self.response = self.send_message(self.messages_transcript[1], "transcript")
                        continue

                    elif self.end_of_conversation:
                        self.console_print(f"Inactive", "green")
                    else:
                        self.console_print(f"Adaptive listening...", "green")

                    while not self.messages_transcript[0]:
                        time.sleep(0.1)
                    self.response = self.send_message(self.messages_transcript[1], "transcript")
                    self.messages_transcript[0] = False
                    continue

                user_input = input(">>>")
                if user_input == "exit":
                    self.response = self.send_message("Verabschiede dich kurz vom Nutzer. Dieser beendet jetzt das Programm.", "system")
                    self.tts_say_text(self.response['content'], True)
                    return

                self.response = self.send_message(user_input)

                continue

            else:

                if not check_if_empty(self.response["cmd_execution"]):
                    result = self.run_cmd_commands(self.response["cmd_execution"])
                    self.send_multiple_mesages(result, prompt_type="cmd")

                if not check_if_empty(self.response["online_search"]):
                    res = self.tavily_search(self.response["online_search"])
                    self.send_multiple_mesages([self.response["online_search"], res], prompt_type="search_result")

                if not check_if_empty(self.response["pythonCode"]):
                    if not self.allow_python_execution:
                        self.send_multiple_mesages("Die Ausführung von Python Code wurde ausgeschaltet.", prompt_type="System")


                    if self.auto_python_execution:
                        self.console_print(f"Starte programm...", "yellow")
                        execute_py = True
                    else:
                        self.console_print(f"Möchtest du folgenden Code ausführen?\n{self.response["pythonCode"]}", "yellow")
                        execute_py = input("[Y/N]").upper() == "Y"

                    if execute_py:
                        python_response = self.run_python_script(self.response["pythonCode"])
                        self.send_multiple_mesages(str(python_response), prompt_type="pythonCode")

                    else:
                        self.send_multiple_mesages("Der Nutzer hat die ausführung des Programmes verweigert.", prompt_type="System")

                if not check_if_empty(self.response["further_process"]):
                    pass
                    # self.send_multiple_mesages(self.response["further_process"], prompt_type="further_process")

                self.response = self.send_multiple_mesages(None, None, is_final=True)

    def run_in_gui(self):

        self.gui_class.showMaximized()
        sys.exit(self.pyside_app.exec())

    def auto_run(self):
        self.load_settings()

        if self.execute_interface == "console":
            self.output(f"Running programm in console...", "log")
            self.initialise_all()
            self.run_in_console()
        elif self.execute_interface == "gui":
            self.output(f"Running programm in GUI...", "error")
            self.initialise_all(start_thread=False)
            self.run_in_gui()
        else:
            self.output(f"Interface {self.execute_interface} not found.", "error")

    def restart_app(self):
        QProcess.startDetached(sys.executable, sys.argv)
        QApplication.quit()

if __name__ == '__main__':
    app = ORION_GemAI()
    app.auto_run()
