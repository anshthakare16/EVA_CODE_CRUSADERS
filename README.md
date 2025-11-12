# EVA - Integrated Logic Assistant

EVA is a voice-controlled AI assistant for your Windows desktop. It can understand your voice commands and perform actions on your computer, such as opening applications, searching the web, and interacting with UI elements on the screen.

## Features

- **Voice-activated**: Listens for the wake word "Jarvis" to start accepting commands.
- **Speech Recognition and Synthesis**: Converts your speech to text and provides voice feedback.
- **Command Classification**: Uses a local machine learning model to understand the intent of your commands.
- **Extensible Action System**: A step-based system to execute a sequence of actions for a given command.
- **Vision Capabilities**: Can "see" the screen, analyze UI elements, and perform clicks on them.
- **GUI**: A user interface to monitor the assistant's status and logs.

## How it Works

The workflow of EVA is as follows:

1.  **Wake Word Detection**: The application continuously listens for the wake word "Jarvis" using a `faster_whisper` model.

2.  **Speech-to-Text**: Once the wake word is detected, it records your command and transcribes it to text using another `faster_whisper` model.

3.  **Command Classification**: The transcribed command is processed by a machine learning model (TF-IDF + Logistic Regression) to classify it into one of the predefined categories, such as:
    - `OPEN_APP`
    - `CLOSE_APP`
    - `WEB_SEARCH`
    - `TYPE_TEXT`
    - `MOUSE_CLICK`
    - and more.

4.  **Keyword Extraction**: Based on the classified command type, relevant keywords are extracted. For example, in the command "open chrome", the application name "chrome" is extracted.

5.  **Step Generation**: A sequence of steps is generated based on the command type and extracted keywords. These steps are defined in templates and can include actions like:
    - `PRESS_KEY`
    - `TYPE_TEXT`
    - `MOUSE_CLICK`
    - `SCREEN_ANALYSIS`
    - `OPEN_APP`
    - and more.

6.  **Execution**: The `ActionRouter` executes the generated steps one by one.
    - For simple actions like keyboard presses and typing, it uses the `SystemExecutor`.
    - For vision-based actions, it captures a screenshot, uses the `OmniParserExecutor` to identify UI elements, and then uses the Gemini API via `ScreenAnalyzer` to determine the precise coordinates for a mouse click.

## Project Structure

```
e:\EVA-main\
├───.env                # Environment variables (API keys)
├───config.py           # Configuration settings
├───main.py             # Main application entry point (PySide6 GUI)
├───gui.py              # Alternative GUI (Tkinter)
├───requirements.txt    # Python dependencies
├───execution/          # Command execution modules
│   ├───action_router.py
│   └───system_executor.py
├───models/             # Command processing and classification models
│   ├───command_classifier.py
│   └───command_processor.py
├───speech/             # Speech recognition and synthesis
│   ├───speech_to_text.py
│   ├───text_to_speech.py
│   └───wake_word_detector.py
└───vision/             # Screen analysis and interaction
    ├───screen_analyzer.py
    └───screenshot_handler.py
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Create a `.env` file** in the root directory and add your Gemini API key:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```

4.  **Run the application:**
    ```bash
    python main.py
    ```

## Usage

1.  Start the application.
2.  Say "Jarvis" to activate the assistant.
3.  Speak your command clearly, for example:
    - "Open Chrome"
    - "Search for the weather"
    - "Click on the login button"

## Example Query Workflow

Let's trace the query: **"Open Chrome and search for MrBeast on YouTube"** to understand how EVA processes it.

### 1. Wake Word and Speech-to-Text
- **Action**: You say "Jarvis".
- **`WakeWordDetector.listen()`**: Detects the wake word.
- **Action**: You say "Open Chrome and search for MrBeast on YouTube".
- **`SpeechToText.listen()`**: Records your voice and uses a `faster-whisper` model to transcribe it into the string `"Open Chrome and search for MrBeast on YouTube"`.

### 2. Command Classification
- **`EvaGui._analyze_query_with_model()`**: The transcribed text is passed to this function.
- **`TfidfVectorizer` & `LogisticRegression`**: The local classification model, trained on data in `main.py`, processes the text and classifies it with the command type **`WEB_SEARCH`**.

### 3. Keyword Extraction
- **`EvaGui._extract_keywords_by_command_type()`**: This function is called with the command and its `WEB_SEARCH` type.
- It calls helper functions to parse the string:
    - **`_extract_profile_name()`**: Finds no specific profile, so it returns `"Default"`.
    - **`_extract_website_and_action()`**: Identifies "youtube" and maps it to `youtube.com`. It also isolates the search query "MrBeast".
- **Result**: The function returns a dictionary of extracted keywords:
    ```python
    {
        'app_name': 'chrome',
        'profile_name': 'Default',
        'website': 'youtube.com',
        'search_query': 'mrbeast'
    }
    ```

### 4. Step Generation
- **`EvaGui._generate_steps_model2()`**: This function uses the `WEB_SEARCH` rule from the `MODEL2_STEP_RULES` dictionary in `main.py`.
- It combines several templates (`chrome_with_profile`, `navigate_to_website`, `search_on_page`) and populates them with the extracted keywords.
- **Result**: A list of precise action steps is generated:
    1.  `PRESS_KEY 'win'` (Open Start Menu)
    2.  `TYPE_TEXT 'chrome'`
    3.  `PRESS_KEY 'enter'` (Launch Chrome)
    4.  `WAIT` for the app to load.
    5.  `FOCUS_WINDOW 'Chrome'`
    6.  `SCREEN_ANALYSIS` to click on the "Default" Chrome profile.
    7.  `PRESS_KEY 'ctrl+l'` (Focus address bar)
    8.  `TYPE_TEXT 'youtube.com'`
    9.  `PRESS_KEY 'enter'` (Navigate to YouTube)
    10. `WAIT` for the page to load.
    11. `PRESS_KEY '/'` (Focus YouTube's search bar)
    12. `TYPE_TEXT 'MrBeast'`
    13. `PRESS_KEY 'enter'` (Execute search)

### 5. Execution
- **`ActionRouter.execute()`**: Receives the list of steps and executes them sequentially.
- For each step, it calls the appropriate function in the `SystemExecutor`.
- **Vision-Powered Clicks**: For the `SCREEN_ANALYSIS` step (clicking the Chrome profile), the `ActionRouter` performs a more complex series of actions:
    1.  **`ScreenshotHandler.capture()`**: Takes a picture of the current screen.
    2.  **`OmniParserExecutor.parse_screen()`**: Analyzes the screenshot to identify all clickable elements.
    3.  **`ScreenAnalyzer.select_coordinate()`**: Sends the list of elements and the target description ("Default") to the Gemini API to get the exact (x, y) coordinate of the target.
    4.  **`SystemExecutor.executor.execute_action("MOUSE_CLICK", ...)`**: Clicks on the coordinate returned by the analyzer.

This entire process, from voice to action, allows EVA to robustly understand and interact with your system.
