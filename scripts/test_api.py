#!/usr/bin/env python3
"""
BYOAI — End-to-End API Integration Test Suite
==============================================

Runs a comprehensive test sequence against the running BYOAI gateway
to verify all API endpoints work correctly.

Usage:
    python scripts/test_api.py
    python scripts/test_api.py --base-url http://localhost:8000
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import httpx


# ── Colors ────────────────────────────────────────────────────────────────────

class Colors:
    """ANSI color codes for terminal output."""
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def colored(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


def print_header(text: str) -> None:
    width = 60
    print()
    print(colored("━" * width, Colors.CYAN))
    print(colored(f"  {text}", Colors.BOLD + Colors.BLUE))
    print(colored("━" * width, Colors.CYAN))


def print_result(label: str, passed: bool) -> None:
    icon = colored("✅ PASS", Colors.GREEN) if passed else colored("❌ FAIL", Colors.RED)
    print(f"  {icon}  {label}")


def pretty_json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


# ── Test Runner ───────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class TestSuite:
    base_url: str
    session_id: str = "test-session-integration"
    results: list = field(default_factory=list)

    def run_all(self) -> bool:
        """Run all tests in sequence and return True if all passed."""
        print()
        print(colored("╔══════════════════════════════════════════════════════════════╗", Colors.CYAN))
        print(colored("║", Colors.CYAN) + colored("         🧪 BYOAI — Integration Test Suite                  ", Colors.BOLD) + colored("║", Colors.CYAN))
        print(colored("║", Colors.CYAN) + f"  Target: {colored(self.base_url, Colors.GREEN)}                            " + colored("║", Colors.CYAN))
        print(colored("╚══════════════════════════════════════════════════════════════╝", Colors.CYAN))

        self.test_health_check()
        self.test_readiness_check()
        self.test_chat_greeting()
        self.test_chat_complaint()
        self.test_chat_booking()
        self.test_conversation_history()
        self.test_chat_out_of_scope()
        self.test_clear_history()
        self.test_history_cleared()

        return self.print_summary()

    def _request(
        self,
        method: str,
        path: str,
        json_body: dict | None = None,
        timeout: float = 30.0,
    ) -> httpx.Response | None:
        """Make an HTTP request and return the response, or None on error."""
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.request(method, url, json=json_body)
            return response
        except httpx.ConnectError:
            print(colored(f"    ⚠ Connection refused: {url}", Colors.RED))
            print(colored("    → Is the server running? Try: docker compose up -d", Colors.DIM))
            return None
        except httpx.TimeoutException:
            print(colored(f"    ⚠ Request timed out: {url}", Colors.RED))
            return None
        except Exception as e:
            print(colored(f"    ⚠ Unexpected error: {e}", Colors.RED))
            return None

    def _add_result(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append(TestResult(name=name, passed=passed, detail=detail))
        print_result(name, passed)
        if detail and not passed:
            print(colored(f"       {detail}", Colors.DIM))

    # ── Individual Tests ──────────────────────────────────────────────────

    def test_health_check(self) -> None:
        """Test 1: Gateway health endpoint."""
        print_header("Test 1 — Gateway Health Check")
        resp = self._request("GET", "/health")
        if resp is None:
            self._add_result("Gateway /health returns 200", False, "Connection failed")
            return

        print(colored(f"    {pretty_json(resp.json())}", Colors.DIM))
        passed = resp.status_code == 200 and resp.json().get("status") == "healthy"
        self._add_result("Gateway /health returns 200", passed)

    def test_readiness_check(self) -> None:
        """Test 2: Gateway readiness (ML service connectivity)."""
        print_header("Test 2 — Readiness Check (ML Connectivity)")
        resp = self._request("GET", "/health/ready")
        if resp is None:
            self._add_result("Gateway /health/ready returns 200", False, "Connection failed")
            return

        print(colored(f"    {pretty_json(resp.json())}", Colors.DIM))
        passed = resp.status_code == 200
        self._add_result("Gateway /health/ready returns 200", passed)

    def test_chat_greeting(self) -> None:
        """Test 3: Send a greeting, expect intent=greeting."""
        print_header("Test 3 — Chat: Greeting → intent=greeting")
        resp = self._request("POST", "/api/v1/chat", json_body={
            "message": "Hello! How are you doing today?",
            "session_id": self.session_id,
        })
        if resp is None:
            self._add_result("Greeting classified correctly", False, "Connection failed")
            return

        data = resp.json()
        print(colored(f"    {pretty_json(data)}", Colors.DIM))
        intent = data.get("intent", "").lower()
        passed = resp.status_code == 200 and intent == "greeting"
        detail = f"Got intent='{intent}'" if not passed else ""
        self._add_result("Greeting classified correctly", passed, detail)

    def test_chat_complaint(self) -> None:
        """Test 4: Send a complaint, expect intent=complaint."""
        print_header("Test 4 — Chat: Complaint → intent=complaint")
        resp = self._request("POST", "/api/v1/chat", json_body={
            "message": "I am very unhappy with your service. My order arrived completely damaged and broken.",
            "session_id": self.session_id,
        })
        if resp is None:
            self._add_result("Complaint classified correctly", False, "Connection failed")
            return

        data = resp.json()
        print(colored(f"    {pretty_json(data)}", Colors.DIM))
        intent = data.get("intent", "").lower()
        passed = resp.status_code == 200 and intent == "complaint"
        detail = f"Got intent='{intent}'" if not passed else ""
        self._add_result("Complaint classified correctly", passed, detail)

    def test_chat_booking(self) -> None:
        """Test 5: Send a booking request, expect intent=booking."""
        print_header("Test 5 — Chat: Booking → intent=booking")
        resp = self._request("POST", "/api/v1/chat", json_body={
            "message": "I would like to book an appointment for next Tuesday at 3 PM.",
            "session_id": self.session_id,
        })
        if resp is None:
            self._add_result("Booking classified correctly", False, "Connection failed")
            return

        data = resp.json()
        print(colored(f"    {pretty_json(data)}", Colors.DIM))
        intent = data.get("intent", "").lower()
        passed = resp.status_code == 200 and intent == "booking"
        detail = f"Got intent='{intent}'" if not passed else ""
        self._add_result("Booking classified correctly", passed, detail)

    def test_conversation_history(self) -> None:
        """Test 6: Check conversation history contains previous messages."""
        print_header("Test 6 — Conversation History")
        resp = self._request("GET", f"/api/v1/history/{self.session_id}")
        if resp is None:
            self._add_result("History returns previous messages", False, "Connection failed")
            return

        data = resp.json()
        print(colored(f"    {pretty_json(data)}", Colors.DIM))

        # History should contain at least 3 messages (greeting, complaint, booking)
        history = data.get("history", data.get("messages", []))
        passed = resp.status_code == 200 and len(history) >= 3
        detail = f"Got {len(history)} messages, expected ≥3" if not passed else ""
        self._add_result("History returns previous messages", passed, detail)

    def test_chat_out_of_scope(self) -> None:
        """Test 7: Send an out-of-scope message."""
        print_header("Test 7 — Chat: Out-of-Scope Message")
        resp = self._request("POST", "/api/v1/chat", json_body={
            "message": "What is the square root of negative one in a parallel universe?",
            "session_id": self.session_id,
        })
        if resp is None:
            self._add_result("Out-of-scope handled gracefully", False, "Connection failed")
            return

        data = resp.json()
        print(colored(f"    {pretty_json(data)}", Colors.DIM))
        # Should return 200 regardless — the system handles everything gracefully
        passed = resp.status_code == 200 and "intent" in data
        self._add_result("Out-of-scope handled gracefully", passed)

    def test_clear_history(self) -> None:
        """Test 8: Clear conversation history."""
        print_header("Test 8 — Clear Conversation History")
        resp = self._request("DELETE", f"/api/v1/history/{self.session_id}")
        if resp is None:
            self._add_result("History cleared successfully", False, "Connection failed")
            return

        data = resp.json()
        print(colored(f"    {pretty_json(data)}", Colors.DIM))
        passed = resp.status_code == 200
        self._add_result("History cleared successfully", passed)

    def test_history_cleared(self) -> None:
        """Test 9: Verify history is empty after clearing."""
        print_header("Test 9 — Verify History is Empty")
        resp = self._request("GET", f"/api/v1/history/{self.session_id}")
        if resp is None:
            self._add_result("History is empty after clear", False, "Connection failed")
            return

        data = resp.json()
        print(colored(f"    {pretty_json(data)}", Colors.DIM))
        history = data.get("history", data.get("messages", []))
        passed = resp.status_code == 200 and len(history) == 0
        detail = f"Got {len(history)} messages, expected 0" if not passed else ""
        self._add_result("History is empty after clear", passed, detail)

    # ── Summary ───────────────────────────────────────────────────────────

    def print_summary(self) -> bool:
        """Print test summary and return True if all passed."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        print()
        print(colored("═" * 60, Colors.CYAN))
        print(colored("  📊 Test Summary", Colors.BOLD))
        print(colored("═" * 60, Colors.CYAN))
        print()

        for r in self.results:
            icon = colored("✅", Colors.GREEN) if r.passed else colored("❌", Colors.RED)
            print(f"  {icon}  {r.name}")

        print()
        if failed == 0:
            print(colored(f"  🎉 All {total}/{total} tests passed!", Colors.GREEN + Colors.BOLD))
        else:
            print(colored(f"  ⚠ {passed}/{total} tests passed, {failed} failed", Colors.YELLOW + Colors.BOLD))

        print(colored("═" * 60, Colors.CYAN))
        print()
        return failed == 0


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BYOAI — End-to-End API Integration Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_api.py
  python scripts/test_api.py --base-url http://localhost:8000
  python scripts/test_api.py --base-url http://staging.example.com:8000
        """,
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the BYOAI gateway (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    suite = TestSuite(base_url=args.base_url.rstrip("/"))
    all_passed = suite.run_all()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
