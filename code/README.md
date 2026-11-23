# Smart Campus Route Planner — Hanoi University (HANU)

**Project 01 (P01) — Artificial Intelligence Final Product**

**Class:** 66FIT3AIN-TT03  
**Group:** G06

A fully offline AI-powered campus wayfinding application developed for Hanoi University (HANU).

The application implements three search algorithms:

- Breadth-First Search (BFS)
- Greedy Best-First Search (GBFS)
- A* Search

## 1. Requirements

- Python 3.11 or higher
- pip
- Required Python packages listed in `requirements.txt`

## 2. Installation

**Step 1.** Open a terminal and navigate to the `code` directory:

```bash
cd code
```

**Step 2.** Create a virtual environment:

```bash
python -m venv .venv
```

**Step 3.** Activate the virtual environment.

Windows (Command Prompt):

```cmd
.venv\Scripts\activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

**Step 4.** Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 3. Run the Application

After completing the installation, make sure your terminal is in the `code` directory and the virtual environment is activated.

Run the application using this single command:

```bash
streamlit run starter/app.py
```

Streamlit will start the application and provide a local URL in the terminal. Open that URL in your web browser to use the Campus Route Planner.

## 4. Main Features

- Campus route planning using BFS, GBFS, and A*.
- Standard and stress-test scenarios.
- Side-by-side route comparison.
- Dynamic route blocking to simulate campus maintenance.
- Accessible-only navigation mode.
- Turn-by-turn route directions.
- Search performance and execution metrics.

## 5. Project Structure

```text
code/
├── README.md
├── requirements.txt
├── data/
├── starter/
│   ├── app.py
│   ├── graph.py
│   └── student_core.py
├── scripts/
│   ├── run_experiments.py
│   ├── profile_search_logs.py
│   └── output/
└── tests/
```

## 6. GitHub Repository

https://github.com/dori1001/66FIT3AIN-TT03_G06_P01

---

