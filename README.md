# ORION_GemAI

## EN Notice
Currently, the GUI Interface only supports german. You can still change the TTS Model to English by adding ```You MUST response in English.``` in ```./assets/settings.json``` in ```added_ai_role```.

## Überblick
Orion ist ein lokaler KI Assistent, der sich von anderen unterscheidet. Er entscheidet selbst, was er tun möchte. Python code oder Befehle ausführen, Online Suche durchführen oder Infos in das Langzeitgedächtnis speichern. Sie entscheidet selbst! Und du kannst alles mit einem Blick sehen!

Python Ausführung verbieten, automatisieren oder doch erst nach einer Bestätigung erlauben? 
Seriös, analytisch, kreativ oder doch eine ( lustig beleidigende :) ) humorvolle KI?
Allwissend oder doch lieber garnichts wissen? - Egal wie du sie haben möchtest, du entscheidest es. 

Und das beste: Du entscheidest selbst welches KI-Modell du verwenden möchtest. Lieber Gemini 3.5 Flasch oder Gemini 3.7 Pro? Oder vielleicht doch llama3.1 für lokale KI Kontrolle?

Aber keine Sorge: Die Einstellungen sind simple gehalten. Du kannst die Standardeinstellung nutzen, zwischen vorgefertigten Persönlichkeiten wechseln oder ALLES selbst entscheiden. 


## Setup
**Achtung** Derzeit gibt es noch KEINEN fertigen Installer der alles alleine erledigt. Da es sich um ein komplexes Projekt handelt ist eine fertige .exe Datei schwierig. Ich arbeite jedoch dran dies zu bewältigen. 

### Voraussetzung 
1. Python 3.12
2. Internet (auch wenn es in Deutschland schwierig ist)

### Step-by-Step
Es wird empfohlen ein venv zu erstellen, da Orion beim ausführen von Python Code nicht installiert Pakete und Bibliotheken automatisch nachinstalliert. 
1. Venv erstellen
2. venv aktivieren
3. Pakete und Bibliotheken installieren
4. Programm ausführen, um fehlende Datein automatisch zu erstellen oder herunterzuladen.
5. Fertig

Das TTS-Modell kann Ressourcen intensiv sein und mit der CPU länger benötigen. Es wird deswegen empfohlen die GPU-Nutzung einzurichten. Leider konnte dies nur für die NVIDIA RTX 3060 TI getestet werden, sodass nicht garantiert werden kann, dass nicht Nvidia GPUs unterstützt werden. 

## Setup für GPU Nutzung 
1. Führe ```./configure_for_gpu.cmd``` aus
2. Installieren Cuda 12.9 ( **KEINE ANDERE VERSION** )
3. Setze ``"cuda_dir"`` in ```./assets/settings.json``` (falls es verändert wurde)
4. Setze in den Einstellungen ``"tts_device"`` zu ``gpu``
5. Programm starten und genießen
