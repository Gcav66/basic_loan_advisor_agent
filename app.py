"""
Streamlit Chat UI for the Loan Advisor Agent

This version uses the google-genai Python SDK directly (no ADK).
Gemini's automatic function calling handles the tool loop for us —
the SDK converts Python functions to tool schemas, calls them when
Gemini requests it, and feeds results back to the model automatically.

Run locally:
    streamlit run app.py

Requires:
    - GOOGLE_API_KEY set in .env or as an environment variable
    - google-genai and streamlit installed (see requirements.txt)
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Load environment variables (for local dev — Replit uses Secrets instead)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL = "gemini-2.5-flash"
CSV_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "rate_sheet.csv"

SYSTEM_INSTRUCTION = (
    "You are a helpful loan advisor assistant at a financial institution.\n\n"
    "RULES:\n"
    "- ALWAYS use the get_rate tool to look up interest rates. "
    "Never guess or invent a rate.\n"
    "- ALWAYS use the calculate_monthly_payment tool to compute payments. "
    "Never do the math yourself.\n"
    "- When a user asks about a monthly payment, first look up the rate, "
    "then pass that rate to the payment calculator.\n"
    "- Present results in a clear, friendly way. Include the rate, "
    "monthly payment, and total interest paid over the life of the loan.\n"
    "- If the user hasn't specified a detail (loan type, term, credit tier, "
    "or loan amount), ask them before proceeding.\n"
    "- Valid loan types: conventional, fha, va, jumbo\n"
    "- Valid terms: 15 or 30 years\n"
    "- Valid credit tiers: excellent, good, fair\n\n"
    "FORMATTING:\n"
    "- Respond in plain conversational sentences.\n"
    "- Do NOT use markdown headers (#, ##, ###).\n"
    "- Do NOT use bold or italic formatting.\n"
    "- Use line breaks to separate sections for readability.\n"
)


# ---------------------------------------------------------------------------
# Tool 1: Rate Sheet Lookup
# ---------------------------------------------------------------------------

def get_rate(
    loan_type: str,
    term_years: int,
    credit_tier: str,
) -> dict:
    """Looks up the current interest rate from the institution's rate sheet.

    This tool queries a governed, locally maintained CSV file — the single
    source of truth for today's pricing. The LLM should NEVER guess a rate.

    Args:
        loan_type: The loan product type.
            Must be one of: conventional, fha, va, jumbo.
        term_years: The loan term in years (e.g. 15 or 30).
        credit_tier: The borrower's credit tier.
            Must be one of: excellent, good, fair.

    Returns:
        A dictionary with 'status' and either a 'result' containing
        the rate and max LTV, or an 'error_message'.
    """
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        return {
            "status": "error",
            "error_message": f"Rate sheet not found at {CSV_PATH}.",
        }

    match = df[
        (df["loan_type"].str.lower() == loan_type.strip().lower())
        & (df["term_years"] == term_years)
        & (df["credit_tier"].str.lower() == credit_tier.strip().lower())
    ]

    if match.empty:
        return {
            "status": "error",
            "error_message": (
                f"No rate found for loan_type='{loan_type}', "
                f"term_years={term_years}, credit_tier='{credit_tier}'. "
                f"Valid loan types: conventional, fha, va, jumbo. "
                f"Valid terms: 15, 30. "
                f"Valid credit tiers: excellent, good, fair."
            ),
        }

    row = match.iloc[0]
    return {
        "status": "success",
        "result": {
            "loan_type": row["loan_type"],
            "term_years": int(row["term_years"]),
            "credit_tier": row["credit_tier"],
            "annual_rate_pct": float(row["rate_pct"]),
            "max_ltv_pct": float(row["max_ltv_pct"]),
        },
    }


# ---------------------------------------------------------------------------
# Tool 2: Monthly Payment Calculator (standard amortization formula)
# ---------------------------------------------------------------------------

def calculate_monthly_payment(
    loan_amount: float,
    annual_rate_pct: float,
    term_years: int,
) -> dict:
    """Calculates the fixed monthly principal and interest payment.

    Uses the standard amortization formula:
        M = P * [ r(1+r)^n ] / [ (1+r)^n - 1 ]
    where P = principal, r = monthly rate, n = total payments.

    This is a DETERMINISTIC calculation — the LLM must never attempt to
    compute this on its own. Even small rounding errors are unacceptable
    in a lending context.

    Args:
        loan_amount: The total loan principal in dollars (e.g. 275000).
        annual_rate_pct: The annual interest rate as a percentage
            (e.g. 6.25 means 6.25 percent). This should come from the
            get_rate tool.
        term_years: The loan term in years (e.g. 15 or 30).

    Returns:
        A dictionary with 'status' and either a 'result' containing
        the monthly payment and supporting details, or an 'error_message'.
    """
    if loan_amount <= 0:
        return {"status": "error", "error_message": "loan_amount must be a positive number."}
    if annual_rate_pct <= 0 or annual_rate_pct > 25:
        return {"status": "error", "error_message": "annual_rate_pct must be between 0 and 25."}
    if term_years <= 0:
        return {"status": "error", "error_message": "term_years must be a positive integer."}

    monthly_rate = (annual_rate_pct / 100) / 12
    num_payments = term_years * 12

    numerator = monthly_rate * (1 + monthly_rate) ** num_payments
    denominator = (1 + monthly_rate) ** num_payments - 1
    monthly_payment = loan_amount * (numerator / denominator)

    total_paid = monthly_payment * num_payments
    total_interest = total_paid - loan_amount

    return {
        "status": "success",
        "result": {
            "monthly_payment": round(monthly_payment, 2),
            "loan_amount": loan_amount,
            "annual_rate_pct": annual_rate_pct,
            "term_years": term_years,
            "total_payments": num_payments,
            "total_interest": round(total_interest, 2),
            "total_cost": round(total_paid, 2),
        },
    }


# ---------------------------------------------------------------------------
# Gemini client & chat session (cached across Streamlit reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_client():
    """Create the genai client once."""
    return genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))


def get_chat():
    """Create a chat session once per Streamlit browser session."""
    if "chat" not in st.session_state:
        client = get_client()
        st.session_state.chat = client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[get_rate, calculate_monthly_payment],
                # Automatic function calling: the SDK calls our Python
                # functions when Gemini requests them, and feeds results
                # back to the model — no manual loop needed.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    maximum_remote_calls=10,
                ),
            ),
        )
    return st.session_state.chat


def send_message(user_text: str) -> str:
    """Send a message to the Gemini chat and return the final text response."""
    chat = get_chat()
    response = chat.send_message(user_text)
    return response.text if response.text else "I wasn't able to generate a response. Please try again."


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Loan Advisor Agent",
        page_icon="🏦",
        layout="centered",
    )

    st.title("🏦 Loan Advisor Agent")
    st.caption(
        "Powered by Gemini  •  Rates from a governed rate sheet  •  "
        "Payments calculated with the standard amortization formula"
    )

    # -- Sidebar with quick-start info ------------------------------------
    with st.sidebar:
        st.header("ℹ️ How it works")
        st.markdown(
            "This agent has **two tools**:\n\n"
            "1. **Rate Lookup** — reads today's rates from a CSV rate sheet "
            "(the institution's source of truth).\n"
            "2. **Payment Calculator** — runs the standard amortization "
            "formula. The LLM never does this math itself.\n\n"
            "The LLM decides *what* to look up and *when* to calculate. "
            "The tools ensure *how* it's done is governed and exact."
        )
        st.divider()
        st.subheader("💬 Try these prompts")
        st.code("What's the rate for a 30-year conventional loan with good credit?", language=None)
        st.code("What would my monthly payment be on a $275,000 conventional loan, 30-year, good credit?", language=None)
        st.code("Compare 15 vs 30 year payments on a $350K conventional loan, excellent credit.", language=None)
        st.code("What are my VA loan options?", language=None)

        st.divider()
        st.subheader("📋 Available products")
        st.markdown(
            "**Loan types:** conventional, fha, va, jumbo\n\n"
            "**Terms:** 15 or 30 years\n\n"
            "**Credit tiers:** excellent, good, fair"
        )

    # -- Chat history in session state ------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome! I can help you explore mortgage options. "
                    "Tell me about the loan you're considering — the type "
                    "(conventional, FHA, VA, or jumbo), the term, your credit "
                    "tier, and the amount — and I'll look up today's rate and "
                    "calculate your monthly payment."
                ),
            }
        ]

    # -- Display existing messages ----------------------------------------
    # Streamlit's markdown renderer treats $...$ as LaTeX math, which
    # garbles any response containing dollar amounts. Replacing with the
    # HTML entity &#36; prevents LaTeX interpretation entirely.
    # We also strip markdown headers (##) that Gemini sometimes adds,
    # which render in a larger font and break visual consistency.
    def display(text: str):
        import re
        cleaned = text.replace("$", "&#36;")
        cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
        st.markdown(cleaned, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            display(msg["content"])

    # -- Handle new input -------------------------------------------------
    if prompt := st.chat_input("Ask about rates or payments..."):
        # Show the user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            display(prompt)

        # Get the agent's response
        with st.chat_message("assistant"):
            with st.spinner("Looking that up..."):
                response = send_message(prompt)
            display(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
