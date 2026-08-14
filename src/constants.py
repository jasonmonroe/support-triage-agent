# src/constants.py
# +---------------------------------------------------------------------------+
# |                            CONSTANTS                                       |
# +---------------------------------------------------------------------------+

import os

# General Variables
MSEC = 1000
PEP8_LINE_CNT = 79

# Data Files
SAMPLE_SUPPORT_TICKETS_FILE = os.path.join("support_tickets", "sample_support_tickets.csv")
SUPPORT_TICKETS_FILE = os.path.join("support_tickets", "support_tickets.csv")
OUTPUT_FILE = os.path.join("support_tickets", "output.csv")

# Environment Variables
MODEL_API_KEY = os.getenv("MODEL_API_KEY")
MODEL_API_URL = os.getenv("MODEL_API_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

# Prompts
SYSTEM_PROMPT = """""".strip()
USER_PROMPT_TEMPLATE = """""".strip()