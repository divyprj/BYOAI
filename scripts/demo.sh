#!/usr/bin/env bash
# ============================================================
# BYOAI - Quick Demo Script
# ============================================================
# Walks through the full API workflow with curl commands.
# Usage: bash scripts/demo.sh [BASE_URL]
# ============================================================

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
SESSION_ID="demo-session-$(date +%s)"

# --- Colors -----------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# --- Helpers ----------------------------------------------
pretty_print() {
    if command -v jq &> /dev/null; then
        echo "$1" | jq .
    else
        echo "$1"
    fi
}

banner() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

step() {
    echo ""
    echo -e "${YELLOW}▶ $1${NC}"
    echo -e "${GREEN}  $2${NC}"
    echo ""
}

# --- Banner ------------------------------------------------
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}${BOLD}            🤖 BYOAI - Conversational AI Demo               ${NC}${CYAN}║${NC}"
echo -e "${CYAN}║${NC}     Conversational Automation with Intent Classification     ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}                                                              ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  Gateway: ${GREEN}${BASE_URL}${NC}                            ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  Session: ${GREEN}${SESSION_ID}${NC}         ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"

# ===========================================================
# 1. Health Check
# ===========================================================
banner "Step 1 - Health Check"
step "Checking if the gateway is alive..." \
     "curl ${BASE_URL}/health"

RESPONSE=$(curl -s "${BASE_URL}/health")
pretty_print "$RESPONSE"

# ===========================================================
# 2. Readiness Check
# ===========================================================
banner "Step 2 - Readiness Check (ML Service Connectivity)"
step "Verifying the ML service is connected and model is loaded..." \
     "curl ${BASE_URL}/health/ready"

RESPONSE=$(curl -s "${BASE_URL}/health/ready")
pretty_print "$RESPONSE"

# ===========================================================
# 3. Send a Greeting
# ===========================================================
banner "Step 3 - Chat: Greeting"
step "Sending a friendly greeting message..." \
     "POST ${BASE_URL}/api/v1/chat"

RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/chat" \
    -H "Content-Type: application/json" \
    -d "{\"message\": \"Hello! How are you doing today?\", \"session_id\": \"${SESSION_ID}\"}")
pretty_print "$RESPONSE"

sleep 1

# ===========================================================
# 4. Send a Complaint
# ===========================================================
banner "Step 4 - Chat: Complaint"
step "Sending a customer complaint..." \
     "POST ${BASE_URL}/api/v1/chat"

RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/chat" \
    -H "Content-Type: application/json" \
    -d "{\"message\": \"I'm very unhappy with the service. My order arrived damaged and nobody is helping me.\", \"session_id\": \"${SESSION_ID}\"}")
pretty_print "$RESPONSE"

sleep 1

# ===========================================================
# 5. Send a Booking Request
# ===========================================================
banner "Step 5 - Chat: Booking Request"
step "Sending a booking/reservation request..." \
     "POST ${BASE_URL}/api/v1/chat"

RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/chat" \
    -H "Content-Type: application/json" \
    -d "{\"message\": \"I'd like to book an appointment for next Tuesday at 3 PM please.\", \"session_id\": \"${SESSION_ID}\"}")
pretty_print "$RESPONSE"

sleep 1

# ===========================================================
# 6. View Conversation History
# ===========================================================
banner "Step 6 - Conversation History"
step "Retrieving the full conversation history for this session..." \
     "GET ${BASE_URL}/api/v1/history/${SESSION_ID}"

RESPONSE=$(curl -s "${BASE_URL}/api/v1/history/${SESSION_ID}")
pretty_print "$RESPONSE"

# ===========================================================
# 7. Clear Conversation History
# ===========================================================
banner "Step 7 - Clear History"
step "Clearing the conversation history..." \
     "DELETE ${BASE_URL}/api/v1/history/${SESSION_ID}"

RESPONSE=$(curl -s -X DELETE "${BASE_URL}/api/v1/history/${SESSION_ID}")
pretty_print "$RESPONSE"

# ===========================================================
# 8. Verify History is Cleared
# ===========================================================
banner "Step 8 - Verify Cleared"
step "Confirming the history is empty..." \
     "GET ${BASE_URL}/api/v1/history/${SESSION_ID}"

RESPONSE=$(curl -s "${BASE_URL}/api/v1/history/${SESSION_ID}")
pretty_print "$RESPONSE"

# ===========================================================
# Done!
# ===========================================================
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}${BOLD}${GREEN}                    ✅ Demo Complete!                        ${NC}${CYAN}║${NC}"
echo -e "${CYAN}║${NC}                                                              ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  MLflow UI:    ${BLUE}http://localhost:5000${NC}                          ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  API Docs:     ${BLUE}http://localhost:8000/docs${NC}                     ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}  ML API Docs:  ${BLUE}http://localhost:8001/docs${NC}                     ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
