"""Text preprocessing, tweet normalization, and conversation turn extraction."""

import re
import html
from typing import List, Dict, Tuple, Optional


def normalize_tweet_text(text: str) -> str:
    """Normalizes noisy Twitter support text while retaining essential semantics."""
    if not isinstance(text, str):
        return ""

    # Unescape HTML entities (&amp; -> &, &lt; -> <, etc.)
    cleaned = html.unescape(text)

    # Normalize URLs
    cleaned = re.sub(r"https?://\S+", "[URL]", cleaned)

    # Normalize numeric anonymized Twitter handles (@115858 -> @customer)
    cleaned = re.sub(r"@\d+", "@customer", cleaned)

    # Normalize multiple whitespace characters
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def parse_conversation_turns(conversation_str: str) -> List[Dict[str, str]]:
    """Parses raw multi-turn conversation string into a list of speaker turns.

    Format in raw dataset:
    Customer: <text>
    Support: <text>
    """
    if not isinstance(conversation_str, str) or not conversation_str.strip():
        return []

    turns: List[Dict[str, str]] = []
    lines = conversation_str.strip().split("\n")

    current_role = None
    current_text_parts = []

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        if line_clean.startswith("Customer:"):
            if current_role and current_text_parts:
                turns.append({"role": current_role, "text": " ".join(current_text_parts).strip()})
                current_text_parts = []
            current_role = "customer"
            current_text_parts.append(line_clean[len("Customer:"):].strip())
        elif line_clean.startswith("Support:"):
            if current_role and current_text_parts:
                turns.append({"role": current_role, "text": " ".join(current_text_parts).strip()})
                current_text_parts = []
            current_role = "support"
            current_text_parts.append(line_clean[len("Support:"):].strip())
        else:
            if current_role:
                current_text_parts.append(line_clean)

    if current_role and current_text_parts:
        turns.append({"role": current_role, "text": " ".join(current_text_parts).strip()})

    return turns


def extract_customer_support_pair(conversation_str: str) -> Optional[Tuple[str, str, List[str]]]:
    """Extracts the initial customer query, the brand resolution/reply, and preceding context.

    Returns:
        (customer_message, brand_response, context_turns) or None if invalid.
    """
    turns = parse_conversation_turns(conversation_str)
    if not turns:
        return None

    # Find the first customer message and first subsequent support response
    first_cust_idx = None
    first_supp_idx = None

    for i, t in enumerate(turns):
        if t["role"] == "customer" and first_cust_idx is None:
            first_cust_idx = i
        elif t["role"] == "support" and first_cust_idx is not None:
            first_supp_idx = i
            break

    if first_cust_idx is None or first_supp_idx is None:
        return None

    customer_msg = normalize_tweet_text(turns[first_cust_idx]["text"])
    support_reply = normalize_tweet_text(turns[first_supp_idx]["text"])

    # Any turns prior to this support reply
    context = [f"{t['role'].capitalize()}: {normalize_tweet_text(t['text'])}" for t in turns[:first_supp_idx]]

    if len(customer_msg) < 5 or len(support_reply) < 5:
        return None

    return customer_msg, support_reply, context
