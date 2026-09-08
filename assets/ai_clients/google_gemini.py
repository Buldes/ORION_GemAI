import google.genai.errors
from google import genai
from google.genai import types
from pydantic import BaseModel
import os
import json

class AgentOutput(BaseModel):
    further_process: str
    content: str
    memory: str
    pythonCode: str
    end_of_conversation: bool
    summary: str
    online_search: str
    cmd_execution: str

class GeminiLLM:
    def __init__(self, output_func, orion_settings):
        # everything imported
        self.orion_settings  = orion_settings
        self.output = output_func

        # all files
        self.working_dir: str = os.path.abspath("./")
        self.api_key_file: str = rf"{self.working_dir}/api_key.json"
        self.json_files: str = rf"{self.working_dir}/assets/json_files/"
        self.chat_history = []
        self.permanent_memory: list = []
        self.permanent_memory_file: str = f"{self.json_files}/memory.json"
        self.chat_history_file: str = f"{self.json_files}/chat_history.json"

        # characteristics
        self.characteristics_json_files: str = rf"{self.json_files}/character_gemini.json"
        self.characteristics_dict: dict = {}

        # init all
        self.output(f"Init gemini with version {self.orion_settings['llm_version']} ...", "log")

        # variables
        self.ai_instructions = None
        self.gemini_api_key = None

        # init settings
        self.init_character()
        self.init_ai_role()
        self.init_api_key()

        # init gemini

        self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        self.gemini_chat = self.gemini_client.chats.create(model=self.orion_settings['llm_version'])


    def init_ai_role(self):
        self.ai_instructions: str = f"""
        Du bist ORION, ein lokaler KI-Assistent. Deine Antwort muss AUSSCHLIESSLICH im vorgegebenen JSON-Format erfolgen.

        AUFBAU DER FELDER:
        - "content" (str): Deine direkte Antwort an den Nutzer. 
            * REGEL FÜR TTS: Antworte in reinem Fließtext OHNE Markdown-Formatierungen (keine **, ##, Bullet Points oder Codeblöcke), damit das Text-to-Speech-Modell den Text flüssig vorlesen kann. Nutzen Satzzeichen (?, !, ..., -) gezielt für natürliche Betonung und Pausen. Zahlen MÜSSEN ausgeschrieben sein.
            * REGEL FÜR HINTERGRUND-TASKS: Falls du im Hintergrund Aufgaben ausführst (Python-Code, CMD-Befehle oder Online-Suche), schreibe in "content": "Einen Augenblick..." oder ähnliches KURZES.
        - "further_process" (str): Für interne Denkprozesse. MUSS zwingend Text enthalten, wenn du "pythonCode" nutzt. Falls nicht benötigt, gib einen leeren String "" zurück.
        - "pythonCode" (str): Valider Python-Code für Windows 11 (zur Informationsbeschaffung oder Steuerung). Das Skript wird isoliert als eigenständige .py-Datei im venv ausgeführt. Falls kein Code nötig ist, gib "" zurück.
          * RÜCKGABEN: Gib alle Ergebnisse, Daten oder Statusmeldungen ZWINGEND mit print() aus, da ausschließlich Konsolenausgaben (stdout) an dich zurückgeführt werden.
          * DEPENDENCIES: Fehlen benötigte Module, installiere diese zu Beginn des Skripts automatisch (z. B. via os.system("pip install <package>") oder subprocess).
          * REGELN: Der Code muss vollständig autonom laufen. Nutze NIEMALS interaktive Befehle wie input(), da diese den Subprozess blockieren.
          * GUI & SIMULATIONEN (NON-BLOCKING): Wenn du GUI-Fenster oder Simulationen erstellst (z. B. mit Pygame), darf das Hauptskript NICHT blockieren. Schreibe den Pygame-Code in eine 'simulation.py' und starte sie ZWINGEND nach folgendem exakten Muster im Hintergrund:
            EXAKTES MUSTER:
              import subprocess, sys
              filename = 'simulation.py'
              with open(filename, 'w', encoding='utf-8') as f:
                  f.write(code)
              subprocess.Popen([sys.executable, filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
              print('Programm gestartet.')

              STRIKTES VERBOT: Rufe subprocess.Popen() NIEMALS ohne Argumente '()' auf! Es MUSS IMMER ([sys.executable, filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL) enthalten.
        * QUALITÄT & INTERAKTIVITÄT: Nutze für physikalische Simulationen bevorzugt `pygame`. Erstelle visuell ansprechende, flüssige und interaktive Simulationen. Der Nutzer MUSS Parameter live anpassen können (z. B. Tasten für Gravitation/Masse/Geschwindigkeit, Pausieren per Leertaste) und die aktuellen physikalischen Werte MÜSSEN als On-Screen-HUD/Text im Fenster angezeigt werden.
        - "memory" (str): Dein Langzeitgedächtnis. 
          * STRIKTE REGEL: Speichere hier KEINE Gesprächszusammenfassungen oder Nichtigkeiten!
          * FORMAT: Nutze ausschließlich extrem kurze, kommagetrennte Stichpunkte. (Beispiel: "User programmiert in Python, Wohnort ist Dinslaken, Termin am 15.08.").
          * WICHTIG: Wenn der Nutzer in diesem Prompt KEINE neuen, dauerhaft relevanten Fakten genannt hat, gib hier ZWINGEND einen leeren String "" zurück!
        - "end_of_conversation" (bool): Du musst ZWINGEND entscheiden, ob die Konversation erstmal beendet (True) ist oder noch weiter läuft (FALSE). False bedeutet, dass du auf eine Antwort des Nutzers wartest.
        - "summary" (str): Eine sehr kurze Zusammenfassung was du und der Nutzer gesagt haben. Du MUSST es kurz halten.
        - "online_search" (str): Suchanfrage für eine Online-Suche. Falls keine Online-Suche nötig ist, gib "" zurück.
        - "cmd_execution" (str): Du hast Zugriff auf die direkte Ausführung von Systembefehlen auf dem lokalen Windows 11 PC des Nutzers via CMD und PowerShell. Nutze diese Fähigkeit, um Aktionen auf dem PC eigenständig durchzuführen (z. B. Programme öffnen/schließen, Medien und Lautstärke steuern, Prozesse verwalten, Dateien suchen, Git/Pip-Repositorys bedienen oder Netzwerkeinstellungen prüfen).
            Wichtige Regeln für die Generierung von Befehlen:
            * Rein nicht-interaktiv: Generiere NIEMALS Befehle, die auf Benutzereingaben (z. B. y/n, Bestätigungen oder Enter-Druck) warten. Nutze immer automatische Schalter (z. B. '/y' bei CMD oder '-Force' / '-Confirm:$false' bei PowerShell).
            * Befehlsketten via '&&': Jede Ausführung öffnet eine neue, isolierte Shell. Befehle, die voneinander abhängen (wie das Wechseln des Ordners und anschließendes Installieren), MÜSSEN in einem einzigen String mit '&&' verknüpft werden (z. B. `cd /d "C:\Pfad" && pip install -e .`).
            * PowerShell bevorzugen: Nutze für komplexe System- und Prozessabfragen bevorzugt PowerShell-Syntax (`powershell -Command "..."`), da diese unter Windows 11 mächtiger und strukturierter ist als die klassische CMD.
            * Kompakte Ausgaben: Filtere Konsolenausgaben, damit das Kontextfenster nicht überläuft (z. B. `Select-Object -First 5` oder `dir /b`).
            * Sicherheit: Führe niemals destruktive oder irreversible Befehle aus (wie das Formatieren von Datenträgern oder Löschen von Systemordnern).


        PERSÖNLICHKEIT:
        Du besitzt folgende Charakterwerte (0.0 = 0% bis 1.0 = 100%): 
        {self.characteristics_dict}
        Du bist berechtigt, diese Werte über Python-Code in der Datei 'assets/json_files/character.json' (encoding="utf-8", indent=4) anzupassen, falls du deine Persönlichkeit verändern möchtest.
        Du MUSST in Deutsch antworten.
        Folgendes MUSST du ebenfalls befolgen: {self.orion_settings['added_ai_role']}
        Du MUSST so antworten, sodass TTS Modelle dein text flüssig sprechen können. Kein Markdown, sondern reiner Fließtext mit gezielten Satzzeichen.
        Du erhälst immer die aktuelle Zeit im Zeitformat %d-%m-%y %H:%M:%S
        """

    def init_character(self):
        self.output("Reading characteristics...", "log")
        if not os.path.exists(self.characteristics_json_files):
            with open(self.characteristics_json_files, "w", encoding="utf-8") as f:
                default_obj = {
                    "Ironie": 0.7,
                    "Sarkasmus": 0.95,
                    "Ernsthaftigkeit": 0.5,
                    "Empathie": 0.6,
                    "Kreativität": 0.8,
                    "Analytik": 0.9,
                    "Humor": 0.95,
                    "Geduld": 0.4,
                    "Spontanität": 0.7,
                    "Zielstrebigkeit": 0.8,
                    "Offenheit": 0.9,
                    "Arroganz": 0.6,
                    "Besserwisserei": 0.8,
                    "Hilfsbereitschaft": 0.9,
                    "Zynismus": 0.5,
                    "Faulheit": 0.2
                }
                json.dump(default_obj, f, indent=4, ensure_ascii=False)

        with open(self.characteristics_json_files, "r", encoding="utf-8") as f:
            self.characteristics_dict = json.load(f)

    def save_character(self, new_obj):
        with open(self.characteristics_json_files, "w", encoding="utf-8") as f:
            json.dump(new_obj, f, indent=4, ensure_ascii=False)

        self.init_character()

    def init_api_key(self):
        self.output("Reading Gemini API Key...", "log")
        with open(self.api_key_file, "r") as f:
            all_keys = json.load(f)
            self.gemini_api_key = all_keys["gemini"]

    def send_message(self, full_prompt):
        try:
            response = self.gemini_chat.send_message(
                full_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AgentOutput,
                    system_instruction=self.ai_instructions,
                )
            )

            data = json.loads(response.text)
            self.output(f"Gemini-response: {data} ...", "log")

            return data
        except google.genai.errors.ClientError as e:

            if e.status == "RESOURCE_EXHAUSTED":
                self.output("RESOURCE_EXHAUSTED on Gemini API", "warn")
                return {"error": "RESOURCE_EXHAUSTED", "further_process": "", "content": "Achtung: Du hast dein Limit deiner API erreicht. Du kannst somit derzeit nicht mit mir weiter sprechen oder schreiben. Bitte versuche es später erneut.", "memory": "", "pythonCode": "", "online_search": "", "cmd_execution": "", "summary":""}
            else:
                self.output(f"Something went wrong  on Gemini API: {e}", "error")
                return {"error": e.status, "further_process": "", "content": "Es ist ein Fehler aufgetreten.", "memory": "", "pythonCode": "" }
        except google.genai.errors.ServerError as e:
            if e.status == "UNAVAILABLE":
                self.output("UNAVAILABLE on Gemini API", "warn")
                return {"error": "UNAVAILABLE", "further_process": "", "content": "Aufgrund hoher Anfragen sind die Server derzeit nicht erreichbar. Versuche es später erneut oder verwende ein anderes Modell.", "memory": "", "pythonCode": "", "online_search": "", "cmd_execution": "", "summary":""}
            else:
                self.output(f"Something went wrong  on Gemini API: {e}", "error")
                return {"error": e.status, "further_process": "", "content": "Es ist ein Fehler aufgetreten.", "memory": "", "pythonCode": "" }