from pydantic import BaseModel, Field
from typing import Literal
import json
import os
import ollama

class AgentOutput(BaseModel):
    further_process: str = Field(description="Interne Denkprozesse oder Erkärungen vor/während Code-Ausführungen.")
    content: str = Field(description="Direkte Antwort an den Nutzer in reinem Fließtext (TTS-konform).")
    memory: str = Field(description="Langzeitgedächtnis als kurze Stichpunkte oder leer.")
    pythonCode: str = Field(description="Isolierter Python-Code zur Ausführung oder leer.")
    end_of_conversation: bool = Field(description="True wenn fertig/warten auf Nutzer, False bei Zwischenschritten.")
    summary: str = Field(description="Kurze Zusammenfassung des aktuellen Dialogschritts.")
    online_search: str = Field(description="Suchanfrage für Online-Suche oder leer.")
    cmd_execution: str = Field(description="CMD/PowerShell Befehl oder leer.")

class ollama_client:
    def __init__(self, output_func, orion_settings):
        # everything imported
        self.orion_settings = orion_settings
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
        self.characteristics_json_files: str = rf"{self.json_files}/character_ollama.json"
        self.characteristics_dict: dict = {}

        # init all
        self.output(f"Init Ollama with version {self.orion_settings['llm_version']} ...", "log")

        # variables
        self.ai_instructions = None
        self.model_name = self.orion_settings['llm_version']

        # init settings
        self.init_character()
        self.init_ai_role()

        # init ollama chat
        self.output("Pulling model if needed...", "log")
        ollama.pull(self.model_name)

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
                    "Geduld": 0.4,
                    "Zielstrebigkeit": 0.8,
                    "Offenheit": 0.9,
                    "Hilfsbereitschaft": 0.9,

                    "Arroganz": 0.6,
                    "Besserwisserei": 0.8,
                    "Zynismus": 0.5,
                    "Faulheit": 0.2
                }
                json.dump(default_obj, f, indent=4)

        with open(self.characteristics_json_files, "r", encoding="utf-8") as f:
            self.characteristics_dict = json.load(f)

    def send_message(self, full_prompt):
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role":"system", "content": self.ai_instructions},
                    {"role":"user", "content": full_prompt},
                ],
                format=AgentOutput.model_json_schema(),
                options={
                    "temperature": 0.1,
                    "num_ctx": 4096
                }
            )
            json_data = json.loads(response.message.content)

            self.output("Ollama-response: {}".format(json_data["content"]), "log")

            return json_data

        except Exception as e:
            self.output(f"Ollama Fehler: {e}", "error")

            return {
                "thought": "",
                "content": "Es ist ein fehler bei der lokalen Modellverarbeitung aufgetreten",
                "execution_type": "",
                "execution_payload": "",
                "is_finished": True
            }


if __name__ == '__main__':
    test_client = ollama_client(
        output_func=lambda x, y: print(x, y),
        orion_settings={
            "llm_version": "llama3.1:latest",
            "added_ai_role": "",
        },
    )

    print("Bitte warten...")
    i = "Begrüße den Nutzer basierend auf deinen Informationen"
    while True:
        response = test_client.send_message(i)
        for i in response.keys():
            print(f"{i:<20}: {response[i]}")

        i = input(">>>")