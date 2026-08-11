import json
import random
import time
from threading import Thread
import google.genai.errors
from google import genai
from google.genai import types
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

class AgentOutput(BaseModel):
    further_process: str
    content: str
    memory: str
    pythonCode: str
    end_of_conversation: bool
    summary: str
    online_search: str

class ORION_GemAI:
    def __init__(self):
        self.working_dir: str = os.path.abspath("./")

        # files and keys
        self.api_key_file: str = rf"{self.working_dir}/api_key.json"
        self.json_files: str = rf"{self.working_dir}/assets/json_files/"
        self.tts_voice_files: str = rf"{self.working_dir}/assets/local_voice/"
        self.gemini_api_key: str = ""

        # gemini
        self.ai_instructions: str = ""
        self.gemini_client = None
        self.gemini_chat = None
        self.gemini_version = "gemini-3.5-flash-lite"

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
        self.use_tts = True
        self.use_stt = False
        self.activation_sound = True
        self.send_history = False

        # voice tts
        self.tts_voice = None
        self.syn_config = None
        self.tts_voice_style = "M1"
        self.tts_voice_device = "cpu"

        # voice stt
        self.oww_model = None
        self.activation_word_file:str = f"{self.working_dir}/assets/stt/Orion.onnx"
        self.active_signal:list = ["Ja? ", "Joo", "Ich Höre ", "Yup"]
        self.word_detected: bool = False
        self.messages_transcript: list = [False, ""]
        self.listen_and_transcript_thread_value = None
        self.stt_audio_queue = queue.Queue()
        self.oww_timeout: list = [2, time.time()]
        self.whisper_model = None
        self.whisper_model_size = "tiny"
        self.stt_settings = {"InitialSilenceTimeout":10, "SilenceTimeout":2, "threshold":30}

        # console running
        self.init_console = True
        self.response = ""
        self.processed_response = ""

        # chat
        self.end_of_conversation: bool = True
        self.auto_python_execution: bool = False # USE WITH CAUTION!

        # searching
        self.allow_tavily_search: bool = True
        self.tavily_api_key: str = ""
        self.tavily = None
        self.tavily_settings: dict = {}

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

    "INIT"

    def initialise_all(self):
        self.output("Initialise ORION...", "log")
        self.load_api_keys()
        self.load_settings()
        self.init_character()
        self.init_ai_role()
        self.init_gemini()
        self.load_memory()
        self.load_chat_history()
        if self.use_tts:
            self.init_tts()
        if self.use_stt:
            self.init_stt()
        if self.allow_tavily_search:
            self.init_tavily_search()

        self.output("Successfully initialised ORION", "log")

    def init_gemini(self):
        self.output(f"Init gemini with version {self.gemini_version} ...", "log")

        self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        self.gemini_chat = self.gemini_client.chats.create(model=self.gemini_version)

    def init_ai_role(self):
        self.ai_instructions: str = f"""
        Du bist ORION, ein lokaler KI-Assistent. Deine Antwort muss AUSSCHLIESSLICH im vorgegebenen JSON-Format erfolgen.

        AUFBAU DER FELDER:
        - "content" (str): Deine direkte Antwort an den Nutzer. Falls du im Hintergrund Aufgaben ausführst (z. B. Python-Code), schreibe hier lediglich "Einen Augenblick...".
        - "further_process" (str): Für interne Denkprozesse. MUSS zwingend Text enthalten, wenn du "pythonCode" nutzt. Falls nicht benötigt, gib einen leeren String "" zurück.
        - "pythonCode" (str): Valider Python-Code für Windows 11 (zur Informationsbeschaffung oder Steuerung). Ausführung erfolgt über exec(), nutze also Zeilenumbrüche statt Semikolons. Falls kein Code nötig ist, gib "" zurück. Für Rückgaben MUSST du die funktion print() verwenden. Fehlende Module MÜSSEN mit pip installiert werden.
          * STRIKTE REGEL: Rückgaben erhältst du AUSSCHLIESSLICH über die print() funktion. Lokale Variable werden NICHT zurückgegeben, NUR print()-Ausgaben.
        - "memory" (str): Dein Langzeitgedächtnis. 
          * STRIKTE REGEL: Speichere hier KEINE Gesprächszusammenfassungen oder Nichtigkeiten!
          * FORMAT: Nutze ausschließlich extrem kurze, kommagetrennte Stichpunkte. (Beispiel: "User programmiert in Python, Wohnort ist Dinslaken, Termin am 15.08.").
          * WICHTIG: Wenn der Nutzer in diesem Prompt KEINE neuen, dauerhaft relevanten Fakten genannt hat, gib hier ZWINGEND einen leeren String "" zurück!
        - "end_of_conversation" (bool): Du musst ZWINGEND entscheiden, ob die Konversation erstmal beendet (True) ist oder noch weiter läuft (FALSE). False bedeutet, dass du auf eine Antwort des Nutzers wartest.
        - "summary" (str): Eine sehr kurze Zusammenfassung was du und der Nutzer gesagt haben. Du MUSST es kurz halten.
        - "online_search" (str): Suchanfrage für eine Online-Suche. Falls keine Online-Suche nötig ist, gib "" zurück.
       
        PERSÖNLICHKEIT:
        Du besitzt folgende Charakterwerte (0.0 = 0% bis 1.0 = 100%): 
        {self.characteristics_dict}
        Du bist berechtigt, diese Werte über Python-Code in der Datei 'assets/json_files/character.json' (encoding="utf-8", indent=4) anzupassen, falls du deine Persönlichkeit verändern möchtest.
        Du MUSST in Deutsch antworten.
        """

    def init_character(self):
        self.output("Reading characteristics...", "log")
        if not os.path.exists(self.characteristics_json_files):
            with open(self.characteristics_json_files, "w", encoding="utf-8") as f:
                default_obj = {"Ironie": 0.4, "Sarkasmus": 0.8, "Ernsthaftigkeit": 0.6, "Empathie": 0.8, "Kreativität": 0.7, "Analytik": 0.9, "Humor": 0.8, "Geduld": 0.8, "Spontanität": 0.6, "Zielstrebigkeit": 0.7, "Offenheit": 0.9}
                json.dump(default_obj, f, indent=4)

        with open(self.characteristics_json_files, "r", encoding="utf-8") as f:
            self.characteristics_dict = json.load(f)

    def load_settings(self):
        self.output("Loading settings...", "log")
        with open(self.settings_file, "r") as f:
            all_settings: dict = json.load(f)

            self.gemini_version: str = all_settings["gemini_version"]

            self.send_history: bool = all_settings["send_history"]
            self.auto_python_execution: bool = all_settings["auto_python_execution"]

            self.use_stt: bool = all_settings["use_stt"]
            self.stt_settings: dict = all_settings["stt_settings"]
            self.activation_sound: bool = all_settings["activation_sound"]
            self.active_signal: list = all_settings["active_words"]
            self.whisper_model_size = all_settings["whisper_model"]

            self.use_tts: list = all_settings["use_tts"]
            self.tts_voice_style: str = all_settings["tts_voice"]
            self.tts_voice_device: str = all_settings["tts_device"]

            self.allow_tavily_search: bool = all_settings["allow_tavily_search"]
            self.tavily_settings: dict = all_settings["tavily_settings"]

    def load_api_keys(self):
        self.output("Loading api key...", "log")
        with open(self.api_key_file, "r") as f:
            all_keys = json.load(f)
            self.gemini_api_key = all_keys["gemini"]
            self.tavily_api_key = all_keys["tavily"]

    def init_tavily_search(self):
        self.output("Loading tavily...", "log")
        self.tavily = TavilyClient(api_key=self.tavily_api_key)

    """TTS"""

    def init_tts(self):
        self.output("Loading Voice...", "log")
        self.tts_voice = TTS(auto_download=True, model_dir=self.tts_voice_files + "/")
        self.syn_config = self.tts_voice.get_voice_style(voice_name=self.tts_voice_style)

    def tts_say_text(self, text, wait_till_finish: bool = False):
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

        if self.playback_worker_thread is not None:
            self.run_playback_worker_thread = False
            while self.playback_worker_thread.is_alive():
                time.sleep(0.1)
            self.run_playback_worker_thread = True
        self.playback_worker_thread = Thread(target=playback_worker, daemon=True)
        self.playback_worker_thread.start()


        wav, sr = self.tts_voice.synthesize(
            text=text,
            lang="de",
            voice_style=self.syn_config,
            total_steps=16,
            speed=1.1
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            self.tts_voice.save_audio(wav, temp_path)
            data, fs = sf.read(temp_path, dtype='float32')
            audio_queue.put((data, fs, text))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if wait_till_finish:
            audio_queue.join()

        audio_queue.put(None)

    def play_audio(self, file_path):
        data, fs = sf.read(file_path, dtype='float32')
        sd.play(data, fs)
        sd.wait()

    """STT"""

    def init_stt(self):
        self.output(f"Init STT with model size {self.whisper_model_size}...", "log")
        openwakeword.utils.download_models()
        self.oww_model = Model(wakeword_models=[self.activation_word_file], inference_framework="onnx")
        self.whisper_model = WhisperModel(self.whisper_model_size, device="cpu", compute_type="int8")

        self.listen_and_transcript_thread_value = Thread(target=self.listen_and_transcript_thread)
        self.listen_and_transcript_thread_value.start()

    def listen_and_transcript_thread(self):
        self.output("STT Model is running in background...", "log")

        with sd.InputStream(samplerate=16000, channels=1, blocksize=1280, dtype='int16', callback=self.audio_callback):
            while True:


                if self.word_detected:
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
                    self.word_detected = rms > (self.stt_settings["threshold"] * 4)
                    if self.word_detected:
                        self.output("Adaptive Input detected", "log")

                else:

                    try:
                        audio_chunk = self.stt_audio_queue.get(timeout=0.1)
                        prediction = self.oww_model.predict(audio_chunk)

                        for model_name, score in prediction.items():

                            if time.time() - self.oww_timeout[1] <= self.oww_timeout[0]:
                                continue

                            if score > 0.5:
                                self.output("Activation Word detected", "log")
                                self.word_detected = True
                    except queue.Empty:
                        continue

    def audio_callback(self,indata, frames, time_info, status):
        if status:
            self.output(f"Status from OWW: {status}", "log")
        self.stt_audio_queue.put(np.frombuffer(indata, dtype=np.int16).copy())

    def record_audio(self, duration: int = 4):
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
            max_silence_chunks = int(self.stt_settings["SilenceTimeout"] / chunk_duration)

            with self.stt_audio_queue.mutex:
                self.stt_audio_queue.queue.clear()

            start_time = time.time()

            while True:
                if not speech_started and (time.time() - start_time > self.stt_settings["InitialSilenceTimeout"]):
                    self.output("Timeout: No speech detected", "log")
                    return False

                try:
                    chunk = self.stt_audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # get how loud
                rms = np.sqrt(np.mean(chunk.astype(np.float32)**2))

                # check if loud enough
                is_speech = rms > self.stt_settings["threshold"]

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

    def send_message(self, prompt, prompt_type: str = "user"):
        self.output(f"Requesting AI on type: {prompt_type}", "log")

        full_prompt: str = ""
        if prompt_type == "user":
            full_prompt = f"""
                    Erinnerungen: {self.permanent_memory} 
                    Zeit: {self.get_current_timestamp()} 
                    Nachricht des Nutzers: {prompt}
                    """
        elif prompt_type == "pythonCode":
            full_prompt = f"""
                    DIES IST KEINE NACHRICHT DES NUTZERS
                    Der Python Code aus deiner Letzten Nachricht wurde Ausgeführt. Dies sind die Ergebnisse
                    Erinnerungen: {self.permanent_memory} 
                    Zeit: {self.get_current_timestamp()} 
                    Ausgabe: {prompt}
                    """
        elif prompt_type == "further_process":
            full_prompt = f"""
                    DIES IST KEINE NACHRICHT DES NUTZERS
                    Du hast further Proces in deiner Letzten Nachricht aktiviert
                    Erinnerungen: {self.permanent_memory} 
                    Zeit: {self.get_current_timestamp()} 
                    further_process Wert: {prompt}
                    """
        elif prompt_type == "transcript":
            full_prompt = f"""
                           Die Folgende Nachricht wurde Transkribiert vom Audio des Nutzers
                           Erinnerungen: {self.permanent_memory} 
                           Zeit: {self.get_current_timestamp()} 
                           Transcriptions des Nutzers: {prompt}
                           """
        elif prompt_type == "search_result":
            full_prompt = f"""
                        Folgende Nachricht enthält Ergebnisse aus einer Suchanfrage
                        Erinnerungen: {self.permanent_memory} 
                        Zeit: {self.get_current_timestamp()} 
                        Suchanfrage: {prompt[0]}
                        Ergebnisse: {prompt[1]}
                        """
        else:
            full_prompt = f"""
                    Prompt-Typ: {prompt_type}
                    Erinnerungen: {self.permanent_memory} 
                    Zeit: {self.get_current_timestamp()} 
                    Prompt: {prompt}
                    """

        # add history
        if self.send_history:
            full_prompt += f"\nChat History: {self.chat_history}"


        try:
            response = self.gemini_chat.send_message(
                full_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AgentOutput,
                    system_instruction=self.ai_instructions,
                    # tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            )

            data = json.loads(response.text)

            return data
        except google.genai.errors.ClientError as e:

            if e.status == "RESOURCE_EXHAUSTED":
                self.output("RESOURCE_EXHAUSTED on Gemini API", "warn")
                return {"further_process": "", "content": "Achtung: Du hast dein Limit deiner API erreicht. Du kannst somit derzeit nicht weiter mit mir weiter sprechen oder schreiben. Bitte versuche es später erneut.", "memory": "", "pythonCode": "" }
            else:
                self.output(f"Something went wrong  on Gemini API: {e}", "error")
                return {"further_process": "", "content": "", "memory": "", "pythonCode": "" }

    """LOCAL STORAGE"""

    def save_new_memory(self, n_memory):
        self.output("Saving new Memory", "log")
        self.permanent_memory.append({"time": self.get_current_timestamp(), "content":n_memory})
        with open(self.permanent_memory_file, "w", encoding="utf-8") as f:
            json.dump(self.permanent_memory, f, indent=4, ensure_ascii=False)

    def load_memory(self):
        self.output("Loading Memory...", "log")

        if not os.path.exists(self.permanent_memory_file):
            return

        with open(self.permanent_memory_file, "r") as f:
            self.permanent_memory = json.load(f)

    def save_chat_history(self, n_chat):
        self.output("Saving new Chat History", "log")

        self.chat_history.append({"time": self.get_current_timestamp(), "content":n_chat})
        with open(self.chat_history_file, "w", encoding="utf-8") as f:
            json.dump(self.chat_history, f, indent=4, ensure_ascii=False)

    def load_chat_history(self):
        self.output("Loading Chat History...", "log")

        if not os.path.exists(self.chat_history_file):
            return

        with open(self.chat_history_file, "r") as f:
            self.chat_history = json.load(f)

    """Online Search"""

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
                self.end_of_conversation = self.response["end_of_conversation"]
                self.save_chat_history(self.response["summary"])

            except Exception as e:
                self.output(f"An error occur: {e}", "error")

            if self.use_tts:
                self.tts_say_text(self.response['content'])

            if self.response["memory"] not in ["", "NONE", "NULL"]:
                self.save_new_memory(self.response["memory"])

            if (check_if_empty(self.response["pythonCode"]) and check_if_empty(self.response["further_process"]) and check_if_empty(self.response["online_search"])):

                # Fallback when using stt. Input() is not needed.
                if self.use_stt:
                    if self.end_of_conversation:
                        self.console_print(f"Inaktiv", "green")
                    else:
                        self.console_print(f"Höhere zu...", "green")

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

            elif not check_if_empty(self.response["pythonCode"]):
                self.console_print(f"Möchtest du folgenden Code ausführen?\n{self.response["pythonCode"]}", "yellow")

                if self.auto_python_execution:
                    execute_py = True
                else:
                    execute_py = input("[Y/N]").upper() == "Y"

                if execute_py:
                    python_response = []
                    code = self.response["pythonCode"]

                    local_scope = {"python_response": python_response, "json": __import__("json")}

                    try:
                        local_scope["print"] = lambda *args: python_response.append(" ".join(map(str, args)))

                        exec(code, {}, local_scope)
                    except Exception as e:
                        python_response.append(f"Es trat ein Fehler auf: {e}")
                        self.console_print(f"Ausgabe: {python_response}", color="red")
                    else:
                        self.console_print(f"Ausgabe: {python_response}", color="green")



                    self.response = self.send_message(str(python_response), prompt_type="pythonCode")

                    continue
                else:
                    response = self.send_message("Der Nutzer hat die ausführung des Programmes verweigert.", prompt_type="System")

                    continue

            elif not check_if_empty(self.response["online_search"]):
                res = self.tavily_search(self.response["online_search"])
                self.response = self.send_message([self.response["online_search"], res], prompt_type="search_result")
                continue

            elif not check_if_empty(self.response["further_process"]):
                self.response = self.send_message(self.response["further_process"], prompt_type="further_process")
                continue



    def auto_run(self, type="console"):
        self.initialise_all()

        if type == "console":
            self.output(f"Running programm in console.", "log")

            self.run_in_console()

if __name__ == '__main__':
    app = ORION_GemAI()
    app.auto_run()
