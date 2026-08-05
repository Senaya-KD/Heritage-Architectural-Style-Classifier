# NHPT Heritage Assistant — Setup Guide

A prototype AI system for the National Heritage Preservation Trust.
It can look at a photo of a building and tell you its architectural
style, and it can answer questions about UK architectural history using
a knowledge base of 20 documents.

---

## What You Need Before Starting

- **Python 3.10** installed on your computer
- A **Gemini API key** (free) — get one at https://aistudio.google.com
- About 2 GB of free disk space (for the AI libraries)

---

## Step 1: Get the Project Files

Unzip the project folder anywhere on your computer, for example:

```
D:\nhpt-heritage-assistant
```

Open a terminal (Command Prompt) and go into that folder:

```
cd D:\nhpt-heritage-assistant
```

---

## Step 2: Create a Virtual Environment

This keeps this project's Python packages separate from everything
else on your computer.

```
python -m venv .venv
.venv\Scripts\activate
```

Your terminal should now show `(.venv)` at the start of the line.

---

## Step 3: Install the Required Packages

```
pip install -r requirements.txt
```

This will take a few minutes — it downloads TensorFlow and several
other large libraries. Let it finish completely.

---

## Step 4: Add Your Gemini API Key

Create a new file in the project folder called exactly `.env`
(no filename before the dot). Open it with Notepad and add this one
line, replacing the text with your real key:

```
GEMINI_API_KEY=your_actual_key_here
```

Save and close the file.

**Important:** never share this file or upload it anywhere. It contains
your private API key.

---

## Step 5: Check Everything Is in Place

Make sure these folders and files exist inside the project:

```
knowledge_base\styles\      -> should contain 16 files
knowledge_base\general\     -> should contain 4 files
models\final_model.keras    -> the trained image classifier
```

If any of these are missing, the system will not work correctly.

---

## Step 6: Build the Knowledge Base

This step reads all the documents and prepares them for searching.
You only need to do this once.

```
python -m src.knowledge_base
```

Wait for it to finish. It should end with a message saying the vector
store was built and saved, followed by a small search test.

---

## Step 7: Run the App

```
streamlit run streamlit_app.py
```

This will automatically open the app in your web browser
(usually at `http://localhost:8501`).

If it does not open by itself, copy that address into your browser
manually.

---

## Using the App

- **To identify a building:** click the paperclip/attach icon in the
  message box, choose a photo, and press send.
- **To ask a question:** just type it in the message box and press
  Enter.
- The assistant remembers the conversation, so you can ask follow-up
  questions like *"Which one came first?"*
- Click **Clear conversation** in the sidebar to start fresh.

---

## Stopping the App

Go back to the terminal window and press `Ctrl + C`.

---

## Common Problems

**"ModuleNotFoundError: No module named 'src'"**
You are running the command from the wrong folder. Make sure you are
in the main project folder (the one containing `streamlit_app.py`),
not inside a subfolder.

**"GEMINI_API_KEY not found"**
The `.env` file is missing, in the wrong folder, or the key was typed
incorrectly. It must be in the same folder as `streamlit_app.py`.

**The app seems stuck on "Loading model and knowledge base..."**
This is normal the first time — it can take 20–30 seconds. If it never
finishes after a minute, check the terminal for an error message.

**Changes to the code don't seem to take effect**
Stop the app completely (`Ctrl + C`) and run `streamlit run
streamlit_app.py` again. Just refreshing the browser page is not
always enough.
