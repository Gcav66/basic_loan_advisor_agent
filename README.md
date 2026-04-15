# Building a Loan Advisor Agent

A hands-on tutorial for building a conversational mortgage advisor using Gemini function calling and Streamlit. The agent looks up rates from a governed rate sheet and calculates monthly payments using the standard amortization formula — the LLM never guesses numbers.

Built for financial services audiences (bankers, credit union analysts, data teams) who want to understand the agent pattern: **the LLM handles the conversation; the tools handle the data and math.**


## The Core Idea

In regulated industries, the things you least want an LLM to improvise are pricing data, regulatory calculations, and eligibility logic. This project demonstrates how to keep the LLM in the conversation layer while delegating governed work to deterministic tools.

| Tool | What It Does | Why It's a Tool |
|---|---|---|
| `get_rate` | Looks up the current interest rate from a CSV rate sheet | Rates are governed data — they change daily and must come from the institution's source of truth, not the LLM's training data |
| `calculate_monthly_payment` | Runs the standard amortization formula | Payment math must be deterministic and exact — even small rounding errors are unacceptable in a lending context |

The `google-genai` SDK's **automatic function calling** handles the orchestration: you pass raw Python functions as tools, and the SDK auto-generates the schemas from your docstrings, calls your functions when Gemini requests them, and feeds the results back to the model.


## Quick Start

```bash
# 1. Clone or download this project

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
#    Get one from https://aistudio.google.com/apikey
#    Then edit .env:
echo "GOOGLE_API_KEY=your_key_here" > .env

# 5. Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.


## Project Structure

```
loan_advisor_project/
├── app.py                          # Streamlit chat UI (google-genai, no ADK)
├── requirements.txt                # Python dependencies
├── .env                            # API key (local dev only)
├── .replit                         # Replit run configuration
│
├── loan_advisor_agent/             # Agent package + governed data
│   ├── __init__.py
│   ├── agent.py                    # ADK version of the agent (for reference / adk web)
│   ├── rate_sheet.csv              # Governed rate sheet (16 rows)
│   └── .env                        # API key for ADK runner
│
├── TUTORIAL.md                     # Step-by-step ADK tutorial walkthrough
└── loan_advisor_tutorial.pptx      # Slide deck (17 slides, 3 sections)
```


## Example Prompts

**Single tool call — rate lookup:**
> What's the rate for a 30-year conventional loan with good credit?

**Tool chaining — rate lookup then payment calculation:**
> What would my monthly payment be on a $275,000 conventional loan, 30-year, good credit?

**Multiple chained calls with comparison:**
> Compare 15 vs 30 year payments on a $350K conventional loan, excellent credit.

**Error handling — not in the rate sheet:**
> What about a 20-year FHA loan?


## How It Works

The entire integration is ~15 lines of code:

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=...)

chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[get_rate, calculate_monthly_payment],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=10,
        ),
    ),
)

response = chat.send_message("What's the rate for a 30-year conventional, good credit?")
print(response.text)
```

When a user asks about a monthly payment, Gemini reasons through the request, calls `get_rate` to look up the interest rate, receives the result, then calls `calculate_monthly_payment` with that rate, and finally summarizes everything in a natural language response. The SDK handles the full loop automatically.


## The Rate Sheet

The rate sheet is a simple CSV with 16 rows covering 4 loan types, 2 terms, and 3 credit tiers:

| loan_type | term_years | credit_tier | rate_pct | max_ltv_pct |
|---|---|---|---|---|
| conventional | 30 | excellent | 6.250 | 97 |
| conventional | 30 | good | 6.625 | 95 |
| fha | 30 | excellent | 6.000 | 96.5 |
| va | 30 | excellent | 5.750 | 100 |
| jumbo | 30 | excellent | 6.750 | 80 |
| ... | ... | ... | ... | ... |

In production, `get_rate` might call an internal pricing API, a Snowflake query, or a database. The tool interface stays the same — you just swap the implementation.


## Docstrings Matter

The `google-genai` SDK reads your function's docstring and type hints to auto-generate the tool schema that Gemini sees. This means:

- The **description** tells Gemini when to call the tool
- The **Args section** becomes the parameter schema (names, types, valid values)
- The **Returns section** tells Gemini what to expect back

Better docstrings = more reliable tool calls. It's prompt engineering applied to function signatures.


## Deploy on Replit

1. Create a new Replit — choose "Import from GitHub" or "Upload folder" and upload all the files.
2. In the Replit sidebar, go to **Secrets** (the lock icon) and add:
   - Key: `GOOGLE_API_KEY`
   - Value: your Google AI Studio API key
3. Hit **Run**. The `.replit` file is pre-configured to launch Streamlit.
4. Click **Deploy** to get a public URL.


## Two Versions of the Agent

This project contains two implementations of the same agent:

| File | Framework | Best For |
|---|---|---|
| `app.py` | `google-genai` SDK directly | Simplicity — no async, no sessions, ~15 lines of integration code. Best for Streamlit apps and quick demos. |
| `loan_advisor_agent/agent.py` | Google ADK | Production features — session management, memory, multi-agent orchestration, dev UI with event inspection. Run with `adk web` from the parent directory. |

Both use the same tools and the same rate sheet. The ADK version is documented in `TUTORIAL.md`.


## Slide Deck

`loan_advisor_tutorial.pptx` is a 17-slide presentation organized into three sections:

1. **The Tools** — rate sheet lookup, amortization calculator, docstrings as schema, tool chaining walkthrough
2. **The Streamlit App** — architecture, the key integration code, demo prompts
3. **Next Steps** — add a third tool, swap the data source, add session memory, deploy on Replit


## Extending This

- **Add a third tool** — a DTI (debt-to-income) qualifier that takes income and monthly obligations, computes front-end/back-end ratios, and returns pass/fail against the tier's thresholds.
- **Swap the data source** — replace the CSV with a Snowflake query (`pd.read_sql()`), REST API call, or database lookup. The tool interface stays the same.
- **Add session memory** — use the ADK version with session/state features, or maintain conversation context through the genai chat object.
- **Add more products** — expand the rate sheet with ARM loans, HELOCs, or construction loans.
