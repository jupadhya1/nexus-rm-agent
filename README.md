# Functional Specification: Enhanced RM Assistant Agent with MCP Integration

---

ABC
import tempfile

from dotenv import load_dotenv
from gen_ai_hub.proxy.native.amazon.clients import Session
from botocore.exceptions import ClientError
from pathlib import Path
import logging
from logger_setup import get_logger
from env_config import MODEL_ID, IMAGE_EXTENSIONS
import tenacity
from typing import List, Dict
import re
import os
from destination_srv import get_destination_service_credentials, generate_token, fetch_destination_details, \
    extract_aicore_credentials
import requests
import json
import time
from api_client import fetch_bank_details_from_odata

from db_connection import get_db_connection

# Configure logging
# logging.basicConfig(level=logging.INFO,
#                    format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
logger=get_logger()

# Initialize AIC Credentials
logger.info("====>image_processor.py -> AIC CREDENTIALS <====")

load_dotenv()

vcap_services=os.environ.get("VCAP_SERVICES")
destination_service_credentials=get_destination_service_credentials(vcap_services)

try:
    oauth_token=generate_token(
        uri = destination_service_credentials['dest_auth_url'] + "/oauth/token",
        client_id = destination_service_credentials['clientid'],
        client_secret = destination_service_credentials['clientsecret']
    )
except requests.exceptions.HTTPError as e:
    if e.response is not None and e.response.status_code == 500:
        raise Exception("HTTP 500: Check if the client secret is correct.") from e
    else:
        raise

AIC_CREDENTIALS=None
dest_AIC="GENAI_AI_CORE"
aicore_details=fetch_destination_details(
    destination_service_credentials['dest_base_url'],
    dest_AIC,
    oauth_token
)
AIC_CREDENTIALS=extract_aicore_credentials(aicore_details)

from gen_ai_hub.proxy import GenAIHubProxyClient
logger.info(f"AIC Credentials: {json.dumps(AIC_CREDENTIALS)}")

proxy_client=GenAIHubProxyClient(
            base_url = AIC_CREDENTIALS['aic_base_url'],
            auth_url = AIC_CREDENTIALS['aic_auth_url'],
            client_id = AIC_CREDENTIALS['clientid'],
            client_secret = AIC_CREDENTIALS['clientsecret'],
            resource_group = AIC_CREDENTIALS['resource_group']
)


def get_bedrock_client_with_proxy(model_id: str, creds: dict):
    proxy_client = GenAIHubProxyClient(
        base_url = creds['aic_base_url'],
        auth_url = creds['aic_auth_url'],
        client_id = creds['clientid'],
        client_secret = creds['clientsecret'],
        resource_group = creds['resource_group']
    )

    session = Session()
    return session.client(model_name=model_id, proxy_client=proxy_client)

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type(ClientError),
    before_sleep=lambda retry_state: logger.info(
        f"Retrying Bedrock API call (attempt {retry_state.attempt_number})..."
    )
)
def generate_image_conversation(bedrock_client, model_id: str, input_text: str, input_image: Path) -> str:
    """
    Sends a message with text and image to a Bedrock model and returns the text response.
    
    Args:
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id: The model ID to use.
        input_text: The text prompt accompanying the image.
        input_image: The path to the input image.
    
    Returns:
        str: The text response from the model.
    
    Raises:
        ValueError: If the image format is unsupported.
        FileNotFoundError: If the image file cannot be read.
        Exception: For other unexpected errors during API call.
    """
    try:
        # Validate image extension
        image_ext = input_image.suffix.lstrip(".").lower()
        
        if image_ext not in IMAGE_EXTENSIONS:
            logger.error(f"Unsupported image format: {image_ext}")
            raise ValueError(f"Unsupported image format: {image_ext}. Supported formats: {IMAGE_EXTENSIONS}")

        # Read image as bytes
        with input_image.open("rb") as f:
            image = f.read()

        message = {
            "role": "user",
            "content": [
                {"text": input_text},
                {
                    "image": {
                        "format": image_ext,
                        "source": {"bytes": image}
                    }
                }
            ]
        }

        # Send the message to Bedrock
        response = bedrock_client.converse(
            modelId=model_id,
            messages=[message]
        )
        
        # Extract text from response
        output_message = response['output']['message']
        result_text = ""
        for content in output_message['content']:
            if 'text' in content:
                result_text += content['text'] + "\n"
        
        logger.info(f"Successfully processed image: {input_image}")
        return result_text.strip()

    except FileNotFoundError:
        logger.error(f"Image file not found: {input_image}")
        raise
    except ValueError as ve:
        logger.error(f"Invalid image: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error processing image {input_image}: {str(e)}")
        raise

def normalize_period_for_db(quarter: str) -> str:
    """
    Convert '1Q25' / '1Q' + '25' like variants to 'Q1 2025' for HANA PERIOD column.
    Assumes quarter like '1Q25' or 'Q1 2025' or already normalized.
    """
    q = quarter.strip().upper().replace("'", "")
    # Already like 'Q1 2025'
    if re.match(r"^Q[1-4]\s+20\d{2}$", q):
        return q
    # Match 1Q25 or 1Q'25
    m = re.match(r"^([1-4])Q\s*'?(\d{2})$", q) or re.match(r"^([1-4])Q(\d{2})$", q)
    if m:
        num = m.group(1)
        yy  = int(m.group(2))
        yyyy = 2000 + yy
        return f"Q{num} {yyyy}"
    # Match Q1 25 / Q1-25
    m = re.match(r"^Q([1-4])[\s\-]?(\d{2})$", q)
    if m:
        num = m.group(1)
        yy  = int(m.group(2))
        yyyy = 2000 + yy
        return f"Q{num} {yyyy}"
    # Fallback: if looks like 'Q1' only, pin to current year
    m = re.match(r"^Q([1-4])$", q)
    if m:
        from datetime import datetime
        yyyy = datetime.now().year
        return f"Q{m.group(1)} {yyyy}"
    return q  # best-effort fallback

def detect_image_ext(image_bytes: bytes, filename: str | None) -> str:
    """
    Return 'png' | 'jpg' | 'jpeg'. Prefer filename if valid; fall back to magic bytes.
    """
    # 1) trust filename if present and allowed
    if filename:
        ext = os.path.splitext(filename)[1].lstrip(".").lower()
        if ext in IMAGE_EXTENSIONS:
            return ext
        # common alias: treat 'jpg'/'jpeg' uniformly
        if ext == "jpg" and "jpg" in IMAGE_EXTENSIONS:
            return "jpg"
        if ext == "jpeg" and "jpeg" in IMAGE_EXTENSIONS:
            return "jpeg"

    # 2) sniff magic numbers
    sig = image_bytes[:8]
    if sig[:3] == b"\xff\xd8\xff":
        # choose one consistently; Bedrock typically accepts 'jpeg'
        return "jpeg" if "jpeg" in IMAGE_EXTENSIONS else "jpg"
    if sig.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"

    # 3) fallback
    return "png"


def process_images(folder_path: str, user_prompt: str = "") -> List[Dict[str, str]]:
    """
    Loads images from a folder, filters them based on bank code and quarter from user prompt,
    processes each with the LLM using combined default and user prompts,
    prints the response, and returns the responses.
    
    Args:
        folder_path: Path to the folder containing images.
        user_prompt: User-provided prompt containing bank name and quarter details, combined with the default prompt.
    
    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing the image path and its analysis.
    """
    # Define default prompt
    default_prompt = """
    Summarize the details with meaningful insights in a concise paragraph format:
    Include Stock Price movement in Percentage with closing stock price and timestamp.
    Stock Insights by identifying meaningful Key phases along with timestamp.
    Concisely explain these key phases with all indicators (price, volume, On Balance Volume, Money Flow Index) and numbers
    """
    
    
    # Combine prompts
    combined_prompt = default_prompt
    if user_prompt.strip():
        combined_prompt += "\n" + user_prompt.strip()

    # Extract bank name and quarter from user_prompt
    bank_code = None
    quarter = None
    known_banks = fetch_bank_details_from_odata()

    # Try structured prompt format with fix for both "Ban:" and "Bank:"
    bank_match = re.search(r'[Bb]an[k]?:\s*([^\n,]+)', user_prompt)
    
    # Look for both "Period:" and "Quarter:" in the prompt
    period_match = re.search(r'(?:[Pp]eriod|[Qq]uarter):\s*([^\n,]+)', user_prompt)

    if bank_match:
        bank_name = bank_match.group(1).strip()
        
        # Try exact match first
        for code, name in known_banks.items():
            if bank_name.lower() == name.lower():
                bank_code = code
                break
                
        # If no exact match, try code match
        if not bank_code:
            for code, name in known_banks.items():
                if bank_name.lower() == code.lower():
                    bank_code = code
                    break
                    
        # If still no match, try contains match
        if not bank_code:
            for code, name in known_banks.items():
                if bank_name.lower() in name.lower() or name.lower() in bank_name.lower():
                    bank_code = code
                    break

    if period_match:
        period = period_match.group(1).strip()
        
        # More flexible quarter format handling
        quarter_match = re.search(r'(?:q)?([1-4])[- ]?q[- ]?(?:20)?([0-9]{2})|([1-4])q(?:\')?([0-9]{2})', 
                                 period, re.IGNORECASE)
        
        if quarter_match:
            if quarter_match.group(1) and quarter_match.group(2):  # e.g., "Q1 2025" or "Q1-25"
                quarter = f"{quarter_match.group(1)}Q{quarter_match.group(2)}"
            elif quarter_match.group(3) and quarter_match.group(4):  # e.g., "1Q25" or "1Q'25"
                quarter = f"{quarter_match.group(3)}Q{quarter_match.group(4)}"
        else:
            # Try to directly extract numbers
            numbers = re.findall(r'\d+', period)
            if len(numbers) >= 2:
                # Assume first number is quarter, second is year
                quarter_num = numbers[0][-1]  # Get last digit if multiple digits
                year = numbers[1][-2:] if len(numbers[1]) > 2 else numbers[1]  # Get last 2 digits if longer
                quarter = f"{quarter_num}Q{year}"
            elif len(numbers) == 1 and 'q' in period.lower():
                # Try to handle formats like 'Q1'
                q_match = re.search(r'q([1-4])', period.lower())
                if q_match:
                    quarter_num = q_match.group(1)
                    # Default to current year if only quarter is specified
                    import datetime
                    current_year = str(datetime.datetime.now().year)[-2:]
                    quarter = f"{quarter_num}Q{current_year}"

    # Fallback to free-form prompt if structured format not found
    if not bank_code or not quarter:
        prompt_lower = user_prompt.lower()

        if not bank_code:
            for code, name in known_banks.items():
                # Check full word match for bank name
                if re.search(rf'\b{re.escape(name.lower())}\b', prompt_lower):
                    bank_code = code
                    break
                # Check full word match for bank code
                elif re.search(rf'\b{re.escape(code.lower())}\b', prompt_lower):
                    bank_code = code
                    break
        
        # Find quarter
        if not quarter:
            # More comprehensive pattern for various quarter formats
            quarter_patterns = [
                r'(?:q)?([1-4])[- ]?q[- ]?(?:20)?([0-9]{2})',  # Q1 2025, Q1-25, 1Q 25
                r'([1-4])q(?:\')?([0-9]{2})',                  # 1Q25, 1Q'25
                r'q([1-4])[\s\-\/](?:20)?([0-9]{2})',          # Q1/25, Q1-2025
                r'(?:20)?([0-9]{2})[\s\-\/]q([1-4])'           # 25-Q1, 2025/Q1
            ]
            
            for pattern in quarter_patterns:
                quarter_match = re.search(pattern, prompt_lower)
                if quarter_match:
                    if quarter_match.group(1) and quarter_match.group(2):
                        # Check if first group is year or quarter
                        if len(quarter_match.group(1)) >= 2:  # Likely a year
                            quarter = f"{quarter_match.group(2)}Q{quarter_match.group(1)[-2:]}"
                        else:  # Likely a quarter
                            quarter = f"{quarter_match.group(1)}Q{quarter_match.group(2)}"
                        break
            
            # If still not found, try to find any numbers and 'q' in the prompt
            if not quarter:
                q_positions = [m.start() for m in re.finditer(r'q', prompt_lower)]
                
                for pos in q_positions:
                    # Look for digits before and after 'q'
                    before = prompt_lower[max(0, pos-5):pos]
                    after = prompt_lower[pos+1:min(len(prompt_lower), pos+6)]
                    
                    # Try to extract quarter number and year
                    before_digits = re.findall(r'\d+', before)
                    after_digits = re.findall(r'\d+', after)
                    
                    if after_digits and len(after_digits[0]) <= 2 and int(after_digits[0]) <= 4:
                        # Format like "Q1"
                        quarter_num = after_digits[0]
                        # Default to current year
                        import datetime
                        current_year = str(datetime.datetime.now().year)[-2:]
                        quarter = f"{quarter_num}Q{current_year}"
                        break
                    elif before_digits and len(before_digits[-1]) <= 2 and int(before_digits[-1]) <= 4:
                        # Format like "1Q"
                        quarter_num = before_digits[-1]
                        # Look for year after
                        import datetime
                        year = after_digits[0][-2:] if after_digits else str(datetime.datetime.now().year)[-2:]
                        quarter = f"{quarter_num}Q{year}"
                        break

    # Try to extract quarter directly from the prompt as a last resort
    if not quarter:
        # Look for "1Q25" or similar patterns directly in the prompt
        direct_quarter_match = re.search(r'([1-4])Q[\'"]?(\d{2})', user_prompt, re.IGNORECASE)
        if direct_quarter_match:
            quarter = f"{direct_quarter_match.group(1)}Q{direct_quarter_match.group(2)}"
        else:
            # Look for Q1 followed by 2025 or 25
            q_year_match = re.search(r'Q([1-4])[^0-9]*(?:20)?(\d{2})', user_prompt, re.IGNORECASE)
            if q_year_match:
                quarter = f"{q_year_match.group(1)}Q{q_year_match.group(2)}"
            else:
                # Check for a number 1-4 followed by digit(s) that could be a year
                num_year_match = re.search(r'[^0-9]([1-4])[^0-9]*(\d{2,4})', user_prompt)
                if num_year_match:
                    year = num_year_match.group(2)[-2:]  # Get last 2 digits of year
                    quarter = f"{num_year_match.group(1)}Q{year}"
                else:
                    # If user prompt contains both "Q1" and "2025" separately
                    q_match = re.search(r'Q([1-4])', user_prompt, re.IGNORECASE)
                    year_match = re.search(r'(?:20)?(\d{2})', user_prompt)
                    if q_match and year_match:
                        quarter = f"{q_match.group(1)}Q{year_match.group(1)[-2:]}"
                    else:
                        # Absolute fallback: if we have a bank but no quarter, use simple heuristics
                        for i in range(1, 5):
                            if f"q{i}" in user_prompt.lower() or f"q {i}" in user_prompt.lower():
                                import datetime
                                current_year = str(datetime.datetime.now().year)[-2:]
                                quarter = f"{i}Q{current_year}"
                                break
    logger.info(
        f"[Bank Extraction] Identified bank_code: '{bank_code}', quarter: '{quarter}' from user prompt: '{user_prompt}'")

    if not bank_code:
        logger.warning(f"Could not identify bank in prompt: {user_prompt}")
        return []
        
    if not quarter:
        logger.warning(f"Could not identify quarter in prompt: {user_prompt}")
        return []

    # ----- NEW: fetch image(s) from HANA instead of scanning folder -----
    period_db = normalize_period_for_db(quarter)
    rows = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Primary: exact match (BANK, PERIOD as normalized e.g. "Q1 2025")
        cursor.execute("""
            SELECT IMAGE_FILE, FILENAME
            FROM CONTENT_INGESTION_IMAGES
            WHERE BANK = ? AND PERIOD = ?
        """, (bank_code, period_db))
        rows = cursor.fetchall()

        # Fallback #1: PERIOD stored as raw quarter (e.g., "1Q25")
        if not rows:
            cursor.execute("""
                SELECT IMAGE_FILE, FILENAME
                FROM CONTENT_INGESTION_IMAGES
                WHERE BANK = ? AND PERIOD = ?
            """, (bank_code, quarter))
            rows = cursor.fetchall()

        # Fallback #2: PERIOD stored with apostrophe (e.g., "1Q'25")
        if not rows:
            alt_q = quarter.replace("Q", "Q'")
            cursor.execute("""
                SELECT IMAGE_FILE, FILENAME
                FROM CONTENT_INGESTION_IMAGES
                WHERE BANK = ? AND PERIOD = ?
            """, (bank_code, alt_q))
            rows = cursor.fetchall()

        cursor.close()

    except Exception as e:
        logger.error(f"[HANA Fetch Error] {str(e)}", exc_info=True)
        return []

    if not rows:
        logger.warning(
            f"No images found in HANA for bank='{bank_code}', period='{period_db}' (from '{quarter}')"
        )
        return []

    # ----- Create Bedrock client and process images -----
    try:
        bedrock_client = get_bedrock_client_with_proxy(MODEL_ID, AIC_CREDENTIALS)
    except Exception as e:
        logger.error(f"Failed to create Bedrock client: {str(e)}")
        return []

    results: List[Dict[str, str]] = []

    for image_bytes, filename in rows:
        # write to temp file so generate_image_conversation(Path) stays unchanged
        ext = detect_image_ext(image_bytes, filename)  # 'png' | 'jpg' | 'jpeg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmpf:
            tmpf.write(image_bytes)
            tmp_path = Path(tmpf.name)

        try:
            resp = generate_image_conversation(bedrock_client, MODEL_ID, combined_prompt, tmp_path)
            logger.info(f"Analyzed image from HANA: {filename}")
            results.append({"image_path": filename or str(tmp_path), "analysis": resp})
        except Exception as e:
            logger.error(f"Failed to process {filename or tmp_path.name}: {str(e)}")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return results

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Architecture Deep Dive](#3-architecture-deep-dive)
4. [Module Specifications](#4-module-specifications)
5. [Data Flow & Processing](#5-data-flow--processing)
6. [Integration Points](#6-integration-points)
7. [Security & Compliance](#7-security--compliance)
8. [Deployment Architecture](#8-deployment-architecture)
9. [API Reference](#9-api-reference)
10. [Configuration Reference](#10-configuration-reference)

---

## 1. Executive Summary

### 1.1 Purpose

The Enhanced RM Assistant Agent is an enterprise-grade AI system designed to augment Relationship Managers (RMs) in Corporate & Institutional Banking (CIB) operations. It provides intelligent automation for:

- Customer profile management and risk assessment
- Investment product recommendations with suitability checks
- Regulatory compliance verification (KYC, AML, Sanctions)
- Risk and Control Self-Assessment (RCSA) automation
- Document analysis and information retrieval

### 1.2 Key Differentiators

| Feature | Description | Business Value |
|---------|-------------|----------------|
| **Multi-Agent Orchestration** | Supervisor pattern with specialized agents | Complex query handling, domain expertise |
| **MCP Integration** | Model Context Protocol for extensible tools | Future-proof architecture, third-party integrations |
| **Compliance-First Design** | Built-in audit logging, PII masking | Regulatory adherence, audit readiness |
| **Enterprise Caching** | Semantic + LRU caching layers | Reduced latency, cost optimization |

### 1.3 Target Users

- **Primary:** Relationship Managers (RMs) in CIB
- **Secondary:** Compliance Officers, Risk Analysts
- **Tertiary:** Operations Support, Product Teams

---

## 2. System Overview

### 2.1 High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                   │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Streamlit   │  │   AI        │  │    REST     │  │   Slack/     │   │
│  │  Web App     │  │  Playground │  │    API      │  │   Teams Bot  │   │
│  └──────┬───────┘  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘   │
│         └──────────────────┼────────────────┼────────────────┘           │
│                            ▼                ▼                             │
├────────────────────────────────────────────────────────────────────────────┤
│                           GATEWAY LAYER                                    │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                 Databricks Model Serving Endpoint                    │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │  │
│  │  │ Rate Limit  │  │   Auth &    │  │   Request   │                  │  │
│  │  │   Control   │  │   AuthZ     │  │   Routing   │                  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────────────────────┤
│                         ORCHESTRATION LAYER                                │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                     SUPERVISOR AGENT                                 │  │
│  │  ┌───────────────────────────────────────────────────────────────┐  │  │
│  │  │  • Query Analysis & Intent Classification                      │  │  │
│  │  │  • Agent Selection & Task Routing                              │  │  │
│  │  │  • Response Synthesis & Quality Control                        │  │  │
│  │  │  • Conversation State Management                               │  │  │
│  │  └───────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                       │
│         ┌────────────┬─────────────┼─────────────┬────────────┐           │
│         ▼            ▼             ▼             ▼            ▼           │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐   │
│  │ Customer  │ │ Product   │ │Compliance │ │ Analytics │ │ Document  │   │
│  │Specialist │ │Specialist │ │ Officer   │ │  Expert   │ │ Analyst   │   │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘   │
│        └─────────────┴─────────────┴─────────────┴─────────────┘         │
│                                    │                                       │
├────────────────────────────────────────────────────────────────────────────┤
│                            TOOL LAYER                                      │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐               │
│  │  UC Function   │  │   MCP Tool     │  │   Native       │               │
│  │    Tools       │  │   Adapters     │  │   Tools        │               │
│  │                │  │                │  │                │               │
│  │ • lookup_cust  │  │ • Compliance   │  │ • Currency     │               │
│  │ • lookup_ut    │  │   Server       │  │   Conversion   │               │
│  │ • vector_search│  │ • RCSA Server  │  │ • Portfolio    │               │
│  │ • convert_usd  │  │ • Document     │  │   Analysis     │               │
│  │                │  │   Server       │  │ • Risk Calc    │               │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘               │
│          └───────────────────┼───────────────────┘                        │
│                              ▼                                             │
├────────────────────────────────────────────────────────────────────────────┤
│                            DATA LAYER                                      │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐               │
│  │  Unity Catalog │  │  Vector Search │  │   External     │               │
│  │    Tables      │  │     Index      │  │   Systems      │               │
│  │                │  │                │  │                │               │
│  │ • customer_    │  │ • ut_pdf_docs  │  │ • Sanctions    │               │
│  │   profile      │  │   _vs_index    │  │   Lists        │               │
│  │ • unit_trust   │  │                │  │ • PEP Database │               │
│  │ • audit_log    │  │                │  │ • FX Rates     │               │
│  └────────────────┘  └────────────────┘  └────────────────┘               │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| Orchestration | LangGraph | Multi-agent workflow management |
| LLM | Llama 3.3 70B | Natural language understanding and generation |
| Vector Search | Databricks VS | Semantic document retrieval |
| Tools | Unity Catalog Functions | Governed data access |
| MCP | Custom Implementation | Extensible tool integration |
| Storage | Delta Tables | Structured data persistence |
| Caching | Custom LRU + Semantic | Performance optimization |

---

## 3. Architecture Deep Dive

### 3.1 Multi-Agent Orchestration

#### 3.1.1 Supervisor Agent

The Supervisor Agent is the central coordinator that:

1. **Analyzes incoming queries** to determine intent and complexity
2. **Selects appropriate specialist agents** based on task requirements
3. **Manages conversation state** across multi-turn interactions
4. **Synthesizes final responses** from specialist outputs

**Decision Logic:**

```python
class SupervisorDecision:
    """
    Supervisor routing decision structure
    """
    reasoning: str          # Explanation of routing decision
    next_agent: str         # Target agent or "FINISH"
    instructions: str       # Specific instructions for agent
    requires_tools: list    # Expected tools to be used
```

**Routing Rules:**

| Query Pattern | Target Agent | Example |
|---------------|--------------|---------|
| Customer lookup, profile, balance | customer_specialist | "Look up Brian Long's profile" |
| Product recommendation, unit trust, fund | product_specialist | "Recommend funds for risk level 3" |
| KYC, AML, sanctions, compliance | compliance_officer | "Check sanctions status for client" |
| Calculate, analyze, convert, metrics | analytics_expert | "Convert 100 GBP to USD" |
| Document, factsheet, summary | document_analyst | "Summarize the Global Growth Fund" |
| Complex multi-domain | Sequential routing | "Recommend compliant products for customer X" |

#### 3.1.2 Specialist Agents

Each specialist agent has:
- **Domain-specific system prompt** defining expertise boundaries
- **Curated tool access** limited to relevant functions
- **Output formatting guidelines** for consistent responses

**Customer Specialist Agent:**
```
Responsibilities:
├── Customer profile retrieval
├── Risk rating interpretation
├── Account balance queries
├── Investment history analysis
└── Cross-sell opportunity identification

Tools Available:
├── lookup_customer_profile
├── assess_customer_risk
└── analyze_portfolio
```

**Product Specialist Agent:**
```
Responsibilities:
├── Unit trust information retrieval
├── Product comparison and ranking
├── Risk-return analysis
├── Suitability matching
└── Fee structure explanation

Tools Available:
├── search_investment_products
├── compare_products
├── lookup_unit_trust
└── search_documents
```

**Compliance Officer Agent:**
```
Responsibilities:
├── KYC status verification
├── Sanctions screening
├── PEP identification
├── AML risk assessment
├── Regulatory limit checking
└── SAR report generation

Tools Available:
├── check_compliance_status
├── assess_product_suitability
├── screen_customer_sanctions
├── check_pep_status
└── check_regulatory_limits
```

**Analytics Expert Agent:**
```
Responsibilities:
├── Numerical calculations
├── Currency conversions
├── Portfolio metrics
├── Risk score computation
├── Statistical analysis
└── Trend identification

Tools Available:
├── convert_currency
├── analyze_portfolio
├── calculate_residual_risk
└── Custom calculation functions
```

**Document Analyst Agent:**
```
Responsibilities:
├── Document search and retrieval
├── Content summarization
├── Entity extraction
├── Document comparison
├── Table data extraction
└── Factsheet analysis

Tools Available:
├── search_documents
├── extract_document_entities
├── summarize_document
├── compare_documents
└── extract_table_data
```

### 3.2 MCP (Model Context Protocol) Integration

#### 3.2.1 MCP Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Client Manager                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │ MCP Server 1  │  │ MCP Server 2  │  │ MCP Server N  │        │
│  │ (Compliance)  │  │ (RCSA)        │  │ (Documents)   │        │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘        │
│          │                  │                  │                 │
│          └──────────────────┼──────────────────┘                 │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Transport Layer                            ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          ││
│  │  │   STDIO     │  │    SSE      │  │  WebSocket  │          ││
│  │  │  Transport  │  │  Transport  │  │  Transport  │          ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘          ││
│  └─────────────────────────────────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 MCP Protocol Handler                         ││
│  │  • JSON-RPC 2.0 Message Processing                          ││
│  │  • Request/Response Correlation                              ││
│  │  • Notification Handling                                     ││
│  │  • Capability Negotiation                                    ││
│  └─────────────────────────────────────────────────────────────┘│
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                 Tool Registry                                ││
│  │  • Tool Discovery & Registration                             ││
│  │  • Schema Conversion (MCP → LangChain)                       ││
│  │  • Execution Routing                                         ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.2.2 MCP Protocol Implementation

**Connection Lifecycle:**

```
1. INITIALIZE
   Client → Server: initialize(protocolVersion, capabilities, clientInfo)
   Server → Client: {serverInfo, capabilities}
   Client → Server: notifications/initialized

2. DISCOVER
   Client → Server: tools/list
   Server → Client: {tools: [{name, description, inputSchema}]}
   
   Client → Server: resources/list
   Server → Client: {resources: [{uri, name, mimeType}]}

3. EXECUTE
   Client → Server: tools/call(name, arguments)
   Server → Client: {content: [{type, text}]}

4. TERMINATE
   Client closes transport connection
```

**Message Format:**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "check_customer_kyc_status",
    "arguments": {
      "customer_id": "CUST0001"
    }
  }
}
```

#### 3.2.3 Banking MCP Servers

**Compliance Tool Server:**

| Tool | Description | Input | Output |
|------|-------------|-------|--------|
| `check_customer_kyc_status` | Verify KYC status | customer_id | KYC status, documents, next review |
| `screen_customer_sanctions` | Screen against sanctions lists | name, country, dob | Match status, lists checked |
| `check_pep_status` | Check PEP status | name, customer_id | PEP category, related PEP |
| `calculate_customer_risk_score` | Calculate risk score | customer_id, include_transactions | Score, breakdown, recommendations |
| `generate_sar_report` | Generate SAR draft | customer_id, transaction_ids, narrative | Report ID, template sections |
| `check_regulatory_limits` | Check reporting thresholds | amount, currency, jurisdiction | Alerts, reporting requirements |

**RCSA Tool Server:**

| Tool | Description | Input | Output |
|------|-------------|-------|--------|
| `assess_control_effectiveness` | Assess control effectiveness | control_id, period | Rating, score, findings |
| `identify_control_gaps` | Find control gaps | process_id, risk_category | Gaps, coverage analysis |
| `generate_rcsa_summary` | Generate RCSA report | business_unit, period | Summary, KRIs, remediation |
| `map_risks_to_controls` | Map risks to controls | risk_ids | Mappings, coverage status |
| `calculate_residual_risk` | Calculate residual risk | inherent_risk, control_effectiveness | Residual score, level |

**Document Analysis Server:**

| Tool | Description | Input | Output |
|------|-------------|-------|--------|
| `extract_document_entities` | Extract entities from text | document_content, entity_types | Entities by type |
| `compare_documents` | Compare two documents | document_a, document_b, type | Similarity score, differences |
| `classify_document` | Classify document type | document_content, scheme | Type, confidence, categories |
| `extract_table_data` | Extract table data | document_content, format | Tables, rows, columns |
| `summarize_document` | Summarize document | document_content, length, focus | Summary, key points |

### 3.3 Tool Layer Architecture

#### 3.3.1 Unity Catalog Function Tools

**Function Registration Pattern:**

```sql
CREATE OR REPLACE FUNCTION {catalog}.{schema}.{function_name}(
    param1 TYPE COMMENT 'Parameter description'
)
RETURNS {return_type}
LANGUAGE {SQL|PYTHON}
COMMENT 'Function description for LLM'
AS
$$
    -- Implementation
$$;
```

**Registered Functions:**

| Function | Type | Description |
|----------|------|-------------|
| `unit_trust_vector_search` | SQL | Vector search on UT documents |
| `lookup_customer_info` | SQL | Customer profile lookup |
| `lookup_ut_info` | SQL | Unit trust details by risk |
| `convert_to_usd` | Python | Currency conversion |

#### 3.3.2 Native LangChain Tools

**Tool Implementation Pattern:**

```python
class BankingTool(BaseTool):
    name: str = "tool_name"
    description: str = "Tool description for LLM"
    args_schema: Type[BaseModel] = ToolArgsModel
    
    def _run(self, **kwargs) -> str:
        """Synchronous execution"""
        pass
    
    async def _arun(self, **kwargs) -> str:
        """Asynchronous execution"""
        pass
```

**Native Tools:**

| Tool | Category | Purpose |
|------|----------|---------|
| `CustomerProfileTool` | Customer | Profile lookup with history |
| `CustomerRiskAssessmentTool` | Customer | Comprehensive risk assessment |
| `ProductSearchTool` | Product | Semantic product search |
| `ProductComparisonTool` | Product | Side-by-side comparison |
| `ComplianceCheckTool` | Compliance | Multi-type compliance check |
| `SuitabilityAssessmentTool` | Compliance | Product suitability |
| `PortfolioAnalysisTool` | Analytics | Portfolio metrics |
| `CurrencyConversionTool` | Analytics | FX conversion |
| `DocumentSearchTool` | Document | Semantic doc search |

---

## 4. Module Specifications

### 4.1 Configuration Module (`config/`)

#### 4.1.1 Settings Classes

```python
@dataclass
class Settings:
    databricks: DatabricksConfig      # Catalog, schema, endpoints
    mcp: MCPConfig                    # MCP server configurations
    agent: AgentConfig                # Agent behavior settings
    compliance: ComplianceConfig      # Compliance thresholds
    vector_search: VectorSearchConfig # VS settings
    cache: CacheConfig                # Caching parameters
    observability: ObservabilityConfig # Logging, tracing
```

**Key Configuration Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `agent.max_iterations` | 15 | Maximum agent loop iterations |
| `agent.max_execution_time_seconds` | 120 | Timeout for agent execution |
| `compliance.high_value_threshold` | 100,000 | High-value transaction threshold |
| `cache.default_ttl_seconds` | 3600 | Cache entry lifetime |
| `vector_search.num_results` | 5 | Default VS results |

### 4.2 MCP Module (`mcp/`)

#### 4.2.1 Client Classes

```
mcp/
├── client.py
│   ├── MCPTransport (ABC)
│   │   ├── StdioTransport
│   │   └── SSETransport
│   ├── MCPServerClient
│   │   ├── connect()
│   │   ├── call_tool()
│   │   ├── read_resource()
│   │   └── get_prompt()
│   └── MCPClientManager
│       ├── add_server()
│       ├── remove_server()
│       ├── call_tool()
│       └── get_all_tools()
│
├── tools.py
│   ├── MCPToolWrapper (BaseTool)
│   ├── MCPToolRegistry
│   ├── MCPResourceTool
│   └── create_mcp_toolkit()
│
└── banking_servers.py
    ├── ComplianceToolServer
    ├── RCSAToolServer
    └── DocumentAnalysisToolServer
```

#### 4.2.2 Tool Adapter Flow

```
MCP Tool Definition          LangChain Tool
┌─────────────────┐         ┌─────────────────┐
│ name: "tool_x"  │         │ name: "tool_x"  │
│ description:... │   ───►  │ description:... │
│ inputSchema: {  │         │ args_schema:    │
│   properties:{} │         │   PydanticModel │
│ }               │         │ _run(): str     │
└─────────────────┘         └─────────────────┘
```

### 4.3 Agents Module (`agents/`)

#### 4.3.1 Class Hierarchy

```
agents/
├── base.py
│   ├── AgentRole (Enum)
│   ├── AgentConfig (dataclass)
│   ├── AgentState (dataclass)
│   └── BaseAgent
│       ├── __init__(config, llm, tools)
│       ├── invoke(state) → dict
│       └── _setup_prompt()
│
├── specialists.py
│   ├── CustomerSpecialistAgent(BaseAgent)
│   ├── ProductSpecialistAgent(BaseAgent)
│   ├── ComplianceOfficerAgent(BaseAgent)
│   ├── AnalyticsExpertAgent(BaseAgent)
│   ├── DocumentAnalystAgent(BaseAgent)
│   ├── RCSASpecialistAgent(BaseAgent)
│   └── KYCSpecialistAgent(BaseAgent)
│
└── orchestrator.py
    ├── SupervisorAgent
    │   ├── __init__(agents, llm)
    │   └── route(state) → dict
    └── MultiAgentOrchestrator
        ├── __init__(model_endpoint, tools)
        ├── run(messages) → dict
        ├── stream(messages) → Generator
        └── get_graph_visualization() → str
```

#### 4.3.2 State Schema

```python
class GraphState(TypedDict):
    messages: Annotated[Sequence[dict], add]  # Conversation messages
    current_agent: str                         # Active agent name
    context: dict                              # Shared context
    agent_history: Annotated[list, add]       # Agent execution log
    completed_tasks: Annotated[list, add]     # Completed task log
    iteration_count: int                       # Loop counter
```

### 4.4 Tools Module (`tools/`)

#### 4.4.1 Banking Tools

```python
# Customer Tools
CustomerProfileTool      # Full profile with optional history
CustomerRiskAssessmentTool  # Risk score + compliance + transactions

# Product Tools
ProductSearchTool        # Semantic search with filters
ProductComparisonTool    # Multi-product comparison

# Compliance Tools
ComplianceCheckTool      # KYC/AML/Sanctions/PEP checks
SuitabilityAssessmentTool  # Product-customer suitability

# Analytics Tools
PortfolioAnalysisTool    # Allocation, risk metrics, recommendations
CurrencyConversionTool   # Multi-currency conversion

# Document Tools
DocumentSearchTool       # Vector search on documents
```

#### 4.4.2 UC Tools Generator

```python
class VectorSearchToolGenerator:
    """Generate tools from VS indexes"""
    create_search_tool(index_name, tool_name, description)
    create_hybrid_search_tool(index_name, table_name, ...)

class UCTableToolGenerator:
    """Generate tools from UC tables"""
    create_lookup_tool(table_name, lookup_column, ...)
    create_aggregation_tool(table_name, group_by, ...)

class GraphRAGToolGenerator:
    """Generate tools for knowledge graph queries"""
    create_entity_search_tool(entity_type, ...)
    create_path_finding_tool(...)
```

### 4.5 Utils Module (`utils/`)

#### 4.5.1 Caching

```python
class LRUCache:
    """Thread-safe LRU cache with TTL"""
    get(key) → Optional[Any]
    set(key, value) → None
    stats → dict  # hits, misses, hit_rate

class SemanticCache:
    """Cache with embedding similarity matching"""
    get(query) → Optional[Any]  # Exact + semantic match
    set(query, response) → None
```

#### 4.5.2 Logging

```python
class AuditLogger:
    """Compliance audit logging"""
    log(action, user_id, customer_id, details)
    log_tool_call(tool_name, arguments, result)
    log_customer_access(customer_id, access_type, user_id)
    get_entries(filters) → list[AuditLogEntry]

class StructuredLogger:
    """JSON structured logging"""
    set_context(**kwargs)  # Persistent fields
    info/warning/error/debug(message, **extra)
```

#### 4.5.3 Utilities

```python
# Decorators
@cached(ttl_seconds=3600)      # Sync function caching
@async_cached(ttl_seconds=3600) # Async function caching
@timed                          # Execution timing
@retry(max_attempts=3)          # Retry with backoff
@rate_limited(limiter, tokens)  # Rate limiting

# Classes
DataMasker        # PII masking in logs/responses
RateLimiter       # Token bucket rate limiter
```

---

## 5. Data Flow & Processing

### 5.1 Query Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         QUERY PROCESSING FLOW                            │
└─────────────────────────────────────────────────────────────────────────┘

User Query: "What unit trusts would you recommend for customer Brian Long?"

Step 1: INTAKE
┌─────────────────────────────────────────────────────────────────────────┐
│ • Receive query via API/UI                                              │
│ • Validate input format                                                 │
│ • Check rate limits                                                     │
│ • Create audit log entry                                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 2: SUPERVISOR ANALYSIS
┌─────────────────────────────────────────────────────────────────────────┐
│ Supervisor Agent Analysis:                                              │
│ • Intent: Product recommendation                                        │
│ • Entities: customer_name="Brian Long"                                  │
│ • Required data: Customer profile, risk rating                          │
│ • Required checks: Suitability, compliance                              │
│                                                                         │
│ Decision: Route to customer_specialist first                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 3: CUSTOMER SPECIALIST
┌─────────────────────────────────────────────────────────────────────────┐
│ Tool Call: lookup_customer_profile(customer_name="Brian Long")          │
│                                                                         │
│ Result:                                                                 │
│ {                                                                       │
│   "customer_id": "CUST0006",                                           │
│   "name": "Brian Long",                                                │
│   "risk_rating": 3,                                                    │
│   "account_balance": 9104.05,                                          │
│   "customer_since": "2020-03-24"                                       │
│ }                                                                       │
│                                                                         │
│ Agent Response: "Customer Brian Long has risk rating 3 (moderate)"      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 4: SUPERVISOR RE-ROUTING
┌─────────────────────────────────────────────────────────────────────────┐
│ Context Updated: customer_risk_rating = 3                               │
│ Decision: Route to product_specialist                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 5: PRODUCT SPECIALIST
┌─────────────────────────────────────────────────────────────────────────┐
│ Tool Call: search_investment_products(                                  │
│   query="suitable unit trusts",                                        │
│   max_risk_rating=3                                                    │
│ )                                                                       │
│                                                                         │
│ Result:                                                                 │
│ {                                                                       │
│   "products": [                                                        │
│     {"id": "UT02", "name": "Income Opportunity Fund", "risk": 3},      │
│     {"id": "UT03", "name": "Balanced Advantage Fund", "risk": 3},      │
│     {"id": "UT04", "name": "Equity Income Fund", "risk": 2}            │
│   ]                                                                     │
│ }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 6: SUPERVISOR RE-ROUTING
┌─────────────────────────────────────────────────────────────────────────┐
│ Context Updated: product_recommendations available                      │
│ Decision: Route to compliance_officer for suitability check             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 7: COMPLIANCE OFFICER
┌─────────────────────────────────────────────────────────────────────────┐
│ Tool Call: check_compliance_status(customer_id="CUST0006")              │
│                                                                         │
│ Result: { "kyc_status": "verified", "overall_status": "CLEAR" }         │
│                                                                         │
│ Tool Call: assess_product_suitability(                                  │
│   customer_id="CUST0006",                                              │
│   product_id="UT02"                                                    │
│ )                                                                       │
│                                                                         │
│ Result: { "suitability_status": "SUITABLE", "risk_match": true }        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
Step 8: SUPERVISOR SYNTHESIS
┌─────────────────────────────────────────────────────────────────────────┐
│ All required information gathered                                       │
│ Decision: FINISH - Synthesize final response                            │
│                                                                         │
│ Final Response:                                                         │
│ "Based on Brian Long's moderate risk profile (rating 3), I recommend:   │
│                                                                         │
│ 1. **Income Opportunity Fund (UT02)** - Balanced fund with 0.99%        │
│    expense ratio, suitable for moderate risk tolerance                  │
│                                                                         │
│ 2. **Balanced Advantage Fund (UT03)** - Balanced fund with 0.55%        │
│    expense ratio, good for capital preservation with growth             │
│                                                                         │
│ 3. **Equity Income Fund (UT04)** - Lower risk (rating 2), bond fund     │
│    with 1.01% expense ratio for conservative allocation                 │
│                                                                         │
│ All recommendations have passed suitability checks. Customer's KYC      │
│ status is verified and current."                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Tool Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TOOL EXECUTION FLOW                              │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │   Agent decides     │
                    │   to call tool      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Tool Router       │
                    │   identifies tool   │
                    │   type              │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  UC Function │     │  MCP Tool   │     │ Native Tool │
    │    Tool      │     │             │     │             │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ Execute via │     │ Send MCP    │     │ Execute     │
    │ Spark SQL   │     │ tools/call  │     │ Python      │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ Query UC    │     │ MCP Server  │     │ Local       │
    │ Table/Index │     │ processes   │     │ computation │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Result Formatter  │
                    │   (JSON output)     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Audit Logger      │
                    │   records call      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Return to Agent   │
                    └─────────────────────┘
```

---

## 6. Integration Points

### 6.1 Databricks Platform Integration

| Integration | Method | Purpose |
|-------------|--------|---------|
| Unity Catalog | SQL Functions | Governed tool execution |
| Vector Search | REST API | Semantic document retrieval |
| Model Serving | MLflow | Agent deployment |
| Delta Tables | Spark SQL | Structured data access |
| Secrets | dbutils.secrets | Credential management |
| Apps | Streamlit | Custom UI deployment |

### 6.2 External System Integration

| System | Protocol | Use Case |
|--------|----------|----------|
| Sanctions Lists | REST/Batch | OFAC, UN, EU screening |
| PEP Database | REST | Political exposure check |
| FX Rate Service | REST | Live exchange rates |
| Document Storage | S3/ADLS | PDF factsheet access |
| Audit System | Kafka/REST | Compliance log shipping |

### 6.3 MCP Server Integration

```yaml
# MCP Server Configuration
servers:
  - name: compliance
    transport: stdio
    command: python
    args: ["-m", "compliance_mcp_server"]
    
  - name: external_data
    transport: sse
    url: https://mcp.internal.bank.com/data
    headers:
      Authorization: Bearer ${MCP_TOKEN}
```

---

## 7. Security & Compliance

### 7.1 Authentication & Authorization

| Layer | Mechanism | Details |
|-------|-----------|---------|
| API Gateway | OAuth 2.0 / SAML | SSO integration |
| Model Serving | Service Principal | Databricks auth passthrough |
| UC Functions | ACLs | Function-level permissions |
| Data Tables | Row-level security | Customer data isolation |

### 7.2 Data Protection

**PII Handling:**
```python
# Automatic masking in logs
DataMasker.mask("Email: john@bank.com") 
# → "Email: jo**********om"

# Sensitive field detection
DataMasker.mask_dict({
    "customer_id": "CUST001",
    "ssn": "123-45-6789",
    "account_number": "1234567890"
})
# → {"customer_id": "CUST001", "ssn": "***MASKED***", "account_number": "***MASKED***"}
```

**Encryption:**
- Data at rest: AES-256 (Delta Lake encryption)
- Data in transit: TLS 1.3
- Secrets: Databricks Secrets with scope isolation

### 7.3 Audit Trail

**Logged Events:**
- All tool invocations with parameters
- Customer data access with user attribution
- Compliance check results
- Agent routing decisions
- Response generation

**Audit Log Schema:**
```json
{
  "timestamp": "2024-12-28T10:30:00Z",
  "action": "tool_call",
  "user_id": "rm_user_123",
  "customer_id": "CUST0006",
  "details": {
    "tool": "lookup_customer_profile",
    "arguments": {"customer_name": "Brian Long"},
    "result_summary": "Profile retrieved successfully",
    "execution_time_ms": 145
  },
  "status": "success",
  "session_id": "sess_abc123"
}
```

### 7.4 Regulatory Compliance

| Regulation | Implementation |
|------------|----------------|
| MAS TRM | Audit logging, access controls, data protection |
| PDPA | PII masking, consent tracking, data minimization |
| AML/CFT | Transaction monitoring, SAR generation, sanctions screening |
| FAA | Suitability assessment, risk disclosure |

---

## 8. Deployment Architecture

### 8.1 Databricks Deployment

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATABRICKS WORKSPACE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    MODEL SERVING LAYER                              │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │              rm-agent-enhanced-endpoint                       │  │ │
│  │  │  • Workload: Small (auto-scale)                              │  │ │
│  │  │  • Scale to Zero: Enabled                                     │  │ │
│  │  │  • Concurrency: 4 requests                                    │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                │                                         │
│                                ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    UNITY CATALOG                                    │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │ │
│  │  │   Models    │  │  Functions  │  │   Tables    │                 │ │
│  │  │             │  │             │  │             │                 │ │
│  │  │ rm_agent_   │  │ lookup_*    │  │ customer_   │                 │ │
│  │  │ enhanced    │  │ convert_*   │  │ profile     │                 │ │
│  │  │             │  │ vector_*    │  │ unit_trust  │                 │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                │                                         │
│                                ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    VECTOR SEARCH                                    │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  Index: ut_pdf_docs_vs_index                                  │  │ │
│  │  │  Endpoint: vs_endpoint                                        │  │ │
│  │  │  Embedding: databricks-gte-large-en                          │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                │                                         │
│                                ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    DATABRICKS APPS                                  │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  rm-assistant-app (Streamlit)                                 │  │ │
│  │  │  • Custom chat UI                                             │  │ │
│  │  │  • SSO authentication                                         │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 MLflow Model Registration

```python
# Model logging
with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        resources=[
            DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
            DatabricksVectorSearchIndex(index_name=VS_INDEX),
            DatabricksTable(table_name=CUSTOMER_TABLE),
            DatabricksFunction(function_name=UC_FUNCTION),
        ],
        pip_requirements=[...],
    )

# Model registration
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(
    model_uri=logged_agent_info.model_uri,
    name="workspace.rm_agent.rm_agent_enhanced"
)
```

### 8.3 Scaling Configuration

| Parameter | Development | Production |
|-----------|-------------|------------|
| Workload Size | Small | Medium/Large |
| Min Instances | 0 | 1 |
| Max Instances | 1 | 10 |
| Scale to Zero | Yes | Optional |
| Concurrency | 4 | 8-16 |

---

## 9. API Reference

### 9.1 Chat Endpoint

**Request:**
```json
POST /serving-endpoints/{endpoint}/invocations

{
  "messages": [
    {"role": "user", "content": "Look up customer Brian Long"}
  ],
  "max_tokens": 500
}
```

**Response:**
```json
{
  "messages": [
    {
      "role": "assistant",
      "content": "I found Brian Long's profile...",
      "tool_calls": [...]
    }
  ]
}
```

### 9.2 Streaming Endpoint

**Request:**
```json
POST /serving-endpoints/{endpoint}/invocations
Content-Type: application/json
Accept: text/event-stream

{
  "messages": [...],
  "stream": true
}
```

**Response (SSE):**
```
data: {"delta": {"content": "I"}}
data: {"delta": {"content": " found"}}
data: {"delta": {"content": " Brian"}}
...
data: [DONE]
```

---

## 10. Configuration Reference

### 10.1 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABRICKS_CATALOG` | Yes | workspace | Unity Catalog name |
| `DATABRICKS_SCHEMA` | Yes | rm_agent | Schema name |
| `LLM_ENDPOINT` | Yes | - | Model serving endpoint |
| `VS_ENDPOINT` | No | vs_endpoint | Vector search endpoint |
| `SERVING_ENDPOINT` | Yes | - | Agent serving endpoint |
| `LOG_LEVEL` | No | INFO | Logging verbosity |
| `CACHE_TTL_SECONDS` | No | 3600 | Cache lifetime |
| `MAX_ITERATIONS` | No | 15 | Agent loop limit |

### 10.2 Agent Configuration

```python
AgentConfig(
    mode=AgentMode.MULTI_AGENT,       # SINGLE, MULTI_AGENT, SUPERVISOR
    max_iterations=15,                 # Loop limit
    max_execution_time_seconds=120,    # Timeout
    enable_human_in_loop=False,        # HITL for sensitive ops
    enable_checkpointing=True,         # Conversation persistence
    supervisor_model="llama-3-3-70b",  # Orchestrator model
    worker_model="llama-3-1-8b",       # Specialist model
    enable_parallel_tools=True,        # Parallel tool execution
    max_parallel_calls=5               # Max concurrent tools
)
```

### 10.3 Compliance Configuration

```python
ComplianceConfig(
    level=ComplianceLevel.ENHANCED,
    enable_audit_logging=True,
    enable_pii_detection=True,
    enable_data_masking=True,
    enable_regulatory_checks=True,
    high_value_transaction_threshold=100000.0,
    suspicious_activity_threshold=50000.0,
    pep_screening_enabled=True,
    sanctions_screening_enabled=True
)
```

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| RM | Relationship Manager - bank employee managing client relationships |
| CIB | Corporate & Institutional Banking division |
| MCP | Model Context Protocol - standard for LLM tool integration |
| UC | Unity Catalog - Databricks data governance layer |
| VS | Vector Search - semantic similarity search |
| KYC | Know Your Customer - identity verification |
| AML | Anti-Money Laundering - financial crime prevention |
| PEP | Politically Exposed Person - high-risk individual |
| SAR | Suspicious Activity Report - regulatory filing |
| RCSA | Risk and Control Self-Assessment |
| LangGraph | LangChain library for agent orchestration |

---

## Appendix B: Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Dec 2024 | Initial release with MCP integration |

---

*End of Functional Specification*
