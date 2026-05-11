# AI Review Detector — Data Pipeline

Local VS Code project to build training data for the AI review detector.

Group 24

Ganesh Prasad D M (MT24AAI161)

Nikhil G Bharadwaj (MT24AAC003)

Nandisha K S (MT24AAC032)

## One-time setup

### 1. Install Ollama (once)

Download from https://ollama.com and install. Then pull the models:

```bash
ollama pull llama3.1:8b
```

Models live in `~/.ollama/models` and persist across reboots.

### 2. Python environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Verify GPU + Ollama

```bash
ollama list                    # should show 4 models
ollama run llama3.1:8b "hi"    # should respond quickly, GPU usage spikes
```
### 4. Run 12_app.py
```bash
Run below in the command prompt window
.venv\...\streamlit run 12_app.py
```
### 5. Enter the customer review
```bash
Enter the customer review in the text box provided.
```
<img width="2553" height="1337" alt="image" src="https://github.com/user-attachments/assets/6a9d8e1f-737d-4010-851a-00b83c850075" />

<img width="2539" height="1316" alt="image" src="https://github.com/user-attachments/assets/c575432b-e92e-47c1-b366-44a32b0a1d82" />

<img width="2518" height="1294" alt="image" src="https://github.com/user-attachments/assets/554e0814-753c-4f10-8f5f-0bb538ea15a2" />

<img width="2528" height="1310" alt="image" src="https://github.com/user-attachments/assets/fc061981-5519-4aa7-814f-f3431274edce" />

<img width="2338" height="1351" alt="image" src="https://github.com/user-attachments/assets/78c0d65a-e2a5-4415-bf45-9a3fc3239dd9" />

<img width="2528" height="1313" alt="image" src="https://github.com/user-attachments/assets/d8432213-31a3-492a-b8a9-dc490863e9c3" />







