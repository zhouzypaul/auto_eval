#!/usr/bin/env python3
"""
Command-line tool to take robots offline or bring them back online.

Usage:
    python robot_control.py offline widowx_drawer --message "Under maintenance until tomorrow"
    python robot_control.py offline widowx_sink --message "Hardware issue"
    python robot_control.py offline all --message "System maintenance"
    python robot_control.py online widowx_drawer
    python robot_control.py online widowx_sink
    python robot_control.py status
"""

import argparse
import os
import sys
from typing import Optional

import requests


def read_api_key() -> str:
    """Read the API key from the file."""
    api_key_file = "admin_api_key.txt"
    if not os.path.exists(api_key_file):
        print(f"Error: API key file '{api_key_file}' not found.")
        print(
            "Make sure the server has been started at least once to generate the API key."
        )
        sys.exit(1)

    with open(api_key_file, "r") as f:
        return f.read().strip()


def get_robot_status(base_url: str = "http://localhost:8080") -> None:
    """Get and display the status of all robots."""
    try:
        response = requests.get(f"{base_url}/robot_status/")
        response.raise_for_status()
        statuses = response.json()

        print("\nRobot Status:")
        print("=" * 50)
        for status in statuses:
            status_text = "ONLINE" if status["is_online"] else "OFFLINE"
            status_color = (
                "\033[92m" if status["is_online"] else "\033[91m"
            )  # Green for online, red for offline
            reset_color = "\033[0m"

            print(f"Robot: {status['robot']}")
            print(f"Status: {status_color}{status_text}{reset_color}")
            if not status["is_online"] and status["offline_message"]:
                print(f"Message: {status['offline_message']}")
            print("-" * 50)

    except requests.RequestException as e:
        print(f"Error: Failed to get robot status. {str(e)}")
        sys.exit(1)


def take_robot_offline(
    robot: str, message: Optional[str] = None, base_url: str = "http://localhost:8080"
) -> None:
    """Take a robot offline."""
    api_key = read_api_key()

    endpoint = f"{base_url}/robot_status/{robot}/offline"
    if robot == "all":
        endpoint = f"{base_url}/robot_status/all/offline"

    headers = {"X-API-Key": api_key}
    params = {}
    if message:
        params["message"] = message

    try:
        response = requests.post(endpoint, headers=headers, params=params)
        response.raise_for_status()
        result = response.json()
        print(f"Success: {result['message']}")
    except requests.RequestException as e:
        if hasattr(e, "response") and e.response is not None:
            try:
                error_detail = e.response.json().get("detail", str(e))
                print(f"Error: {error_detail}")
            except:
                print(f"Error: {str(e)}")
        else:
            print(f"Error: {str(e)}")
        sys.exit(1)


def bring_robot_online(robot: str, base_url: str = "http://localhost:8080") -> None:
    """Bring a robot back online."""
    api_key = read_api_key()

    headers = {"X-API-Key": api_key}

    try:
        response = requests.post(
            f"{base_url}/robot_status/{robot}/online", headers=headers
        )
        response.raise_for_status()
        result = response.json()
        print(f"Success: {result['message']}")
    except requests.RequestException as e:
        if hasattr(e, "response") and e.response is not None:
            try:
                error_detail = e.response.json().get("detail", str(e))
                print(f"Error: {error_detail}")
            except:
                print(f"Error: {str(e)}")
        else:
            print(f"Error: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Control robot online/offline status")
    parser.add_argument(
        "--url", default="http://localhost:8080", help="Base URL of the server"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Status command
    status_parser = subparsers.add_parser("status", help="Get status of all robots")

    # Offline command
    offline_parser = subparsers.add_parser("offline", help="Take a robot offline")
    offline_parser.add_argument(
        "robot",
        choices=["widowx_drawer", "widowx_sink", "all"],
        help="Robot to take offline or 'all' for all robots",
    )
    offline_parser.add_argument("--message", "-m", help="Message to display to users")

    # Online command
    online_parser = subparsers.add_parser("online", help="Bring a robot back online")
    online_parser.add_argument(
        "robot",
        choices=["widowx_drawer", "widowx_sink"],
        help="Robot to bring back online",
    )

    args = parser.parse_args()

    if args.command == "status":
        get_robot_status(args.url)
    elif args.command == "offline":
        take_robot_offline(args.robot, args.message, args.url)
        get_robot_status(args.url)  # Show updated status
    elif args.command == "online":
        bring_robot_online(args.robot, args.url)
        get_robot_status(args.url)  # Show updated status
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
