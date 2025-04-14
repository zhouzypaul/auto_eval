"""
To run the server:
    uvicorn job_scheduler:app --reload

You can submit and check jobs status with the web interface under /page

To submit a job with CLI:

curl -X 'POST' \
  'http://localhost:8080/jobs/' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "description": "Job description",
    "robot": "WidowX Drawer",
    "policy_server_ip": "localhost",
    "policy_server_port": "8000"
}'


To check job status with CLI:

curl 'http://localhost:8080/jobs/<job_id>'  | jq
curl 'http://localhost:8080/jobs/'  | jq


To change job status with CLI: (untested)

curl -X 'PATCH' \
  'http://localhost:8080/jobs/<job_id>' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "status": "FAILED"
}'
"""
import copy
import json
import logging
import os
import secrets
import socket
import time
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import libtmux
import numpy as np
import portalocker
import requests
from fastapi import Depends, FastAPI, File, Header, HTTPException, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine, select

from auto_eval.robot.robot_commands import (
    create_interface,
    sleep_and_torque_off,
    torque_on,
)
from auto_eval.utils.info import print_red
from auto_eval.web_ui.launcher import RobotIPs

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

logging.getLogger("urllib3").setLevel(logging.WARNING)

##############################################################################
# Configurations and DB setup
##############################################################################

# SQLite database setup
sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url, echo=False)

WEB_VIEWER_IP = "0.0.0.0"
WEB_VIEWER_PORT = 5000
IMAGE_TYPES = ["observation"]
UPLOAD_FOLDER = os.path.join(
    os.path.dirname(__file__), "uploads"
)  # also specified by auto_eval/visualization/web_viewer.py

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Add these helper functions before the API endpoints
def is_valid_image_type(image_type: str) -> bool:
    return image_type in IMAGE_TYPES


def create_feed():
    return {
        "observation": None,
        "status": {
            "language_instruction": "N/A",
            "episode": 0,
            "timestep": 0,
        },
    }


robot_video_feeds = {f"feed{i}": create_feed() for i in range(2)}

##############################################################################
# Job/Data Models
##############################################################################


class Status(str):
    QUEUED = "queued"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Job Model
class Job(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    description: str
    policy_name: Optional[str] = Field(default="", nullable=True)
    robot: str
    task: str
    policy_server_ip: str
    policy_server_port: int
    num_episodes: Optional[int] = Field(default=50)
    max_steps: Optional[int] = Field(default=70)
    type: str = "policy_server"
    status: str = Status.QUEUED
    submitted_at: float = Field(default_factory=time.time)
    executed_at: Optional[float] = Field(default=None, nullable=True)
    completed_at: Optional[float] = Field(default=None, nullable=True)
    tmux_output: Optional[str] = Field(default=None, nullable=True)
    wandb_url: Optional[str] = Field(default=None, nullable=True)
    success_rate: Optional[float] = Field(default=None, nullable=True)
    submitter_id: Optional[str] = Field(default=None, nullable=True)


# Queue Model
class JobQueue(SQLModel, table=True):
    job_id: str = Field(default=uuid4, primary_key=True)
    robot: str = Field(default="", index=True)


# Robot Status Model
# this keeps track of whether each robot is being taken offline
class RobotStatus(SQLModel, table=True):
    robot: str = Field(primary_key=True)
    is_online: bool = Field(default=True)
    offline_message: Optional[str] = Field(default=None, nullable=True)


##############################################################################
# Database Setup
##############################################################################

# Create the database tables
SQLModel.metadata.create_all(engine)
# Initialize robot statuses if they don't exist
with Session(engine) as session:
    for robot in ["widowx_drawer", "widowx_sink"]:
        robot_status = session.exec(
            select(RobotStatus).where(RobotStatus.robot == robot)
        ).first()
        if not robot_status:
            robot_status = RobotStatus(robot=robot, is_online=True)
            session.add(robot_status)
    session.commit()

##############################################################################
# Admin Key for security
##############################################################################

# Generate API key if it doesn't exist
API_KEY_FILE = "admin_api_key.txt"
if not os.path.exists(API_KEY_FILE):
    with open(API_KEY_FILE, "w") as f:
        api_key = secrets.token_hex(32)
        f.write(api_key)
    print(f"Generated new admin API key and saved to {API_KEY_FILE}")

# Read API key
with open(API_KEY_FILE, "r") as f:
    ADMIN_API_KEY = f.read().strip()

# API key security
api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key. You are not authorized to perform this action.",
        )
    return api_key


##############################################################################
# REST API Endpoints
##############################################################################

# Serve the static files
app.mount("/page", StaticFiles(directory="static", html=True), name="static")


# Show the tmux output of a job
@app.get("/jobs/{job_id}/output")
def get_job_output(job_id: str):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job.tmux_output:
            return {"output": "No output available"}
        return {"output": job.tmux_output}


# Test policy server connection
class PolicyServerTestRequest(BaseModel):
    policy_server_ip: str
    policy_server_port: int


def is_port_open(host: str, port: int, timeout: float = 2.0) -> Tuple[bool, str]:
    """Check if a port is open on a host."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            return True, "Port is open"
        else:
            return False, f"Port is closed or unreachable (error code: {result})"
    except socket.gaierror:
        return False, f"Could not resolve hostname: {host}"
    except socket.error as e:
        return False, f"Socket error: {str(e)}"


# Endpoint to test policy server connection
@app.post("/test_policy_server/")
def test_policy_server(request: PolicyServerTestRequest):
    try:
        # First check if the port is open
        port_open, port_message = is_port_open(
            request.policy_server_ip, request.policy_server_port
        )
        if not port_open:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Could not connect to policy server at {request.policy_server_ip}:{request.policy_server_port}. {port_message}",
                    "error_type": "port_closed",
                },
            )

        # Create a dummy observation with a blank image
        dummy_image = np.zeros((256, 256, 3), dtype=np.uint8)
        dummy_proprio = np.zeros((8,), dtype=np.float32)
        dummy_instruction = "test connection"

        # Send request to the policy server
        try:
            # we need to able to serialize numpy arrays to json to send over the network
            import json_numpy

            json_numpy.patch()

            raw_response = requests.post(
                f"http://{request.policy_server_ip}:{request.policy_server_port}/act",
                json={
                    "image": dummy_image,
                    "proprio": dummy_proprio,
                    "instruction": dummy_instruction,
                },
                timeout=8,  # 8 second timeout - client side has a 10 second timeout
            )

            # Check if the response is valid JSON
            try:
                response = raw_response.json()
            except ValueError:
                # Not a valid JSON response
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": f"Policy server did not return valid JSON. Response: {raw_response.text[:100]}...",
                    },
                )
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ) as e:
            error_type = "connection_error"
            if isinstance(e, requests.exceptions.Timeout):
                error_type = "timeout_error"
                message = f"Connection to policy server at {request.policy_server_ip}:{request.policy_server_port} timed out. The server might be busy or unreachable."
            elif isinstance(e, requests.exceptions.ConnectionError):
                message = f"Could not connect to policy server at {request.policy_server_ip}:{request.policy_server_port}. Please check the IP and port."
            else:
                error_type = "request_error"
                message = f"Error connecting to policy server at {request.policy_server_ip}:{request.policy_server_port}: {str(e)}"

            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": message,
                    "error_type": error_type,
                },
            )

        # Check if response is a list or numpy array
        if isinstance(response, list):
            action = np.array(response)
        else:
            action = response

        # Check if action has the expected shape
        if hasattr(action, "shape") and action.shape == (7,):
            return {
                "status": "success",
                "message": "Policy server returned an action with shape (7,)",
                "shape": "(7,)",
            }
        else:
            # Get detailed information about the shape
            if hasattr(action, "shape"):
                shape_str = str(action.shape)
            else:
                shape_str = f"type: {type(action).__name__} with value: {action}"

            return {
                "status": "error",
                "message": f"Policy server returned an action with unexpected shape: {shape_str}",
                "shape": shape_str,
            }

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"Failed to connect to policy server: {str(e)}",
            },
        )


# Endpoint to submit a new job
@app.post("/jobs/")
def submit_job(job: Job):
    with Session(engine) as session:
        # Check if the robot is online
        robot_status = session.exec(
            select(RobotStatus).where(RobotStatus.robot == job.robot)
        ).first()
        if robot_status and not robot_status.is_online:
            message = (
                robot_status.offline_message
                or "This robot is currently offline and not accepting jobs."
            )
            raise HTTPException(status_code=400, detail=message)

        # Add the job to the database
        session.add(job)
        session.commit()
        job_id = job.id
        # Add job to the queue table with robot information
        queue_entry = JobQueue(job_id=job_id, robot=job.robot)
        session.add(queue_entry)
        session.commit()
    return {"job_id": job_id, "status": "Job submitted and queued"}


# Endpoint to check job status
@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    with Session(engine) as session:
        job = session.exec(select(Job).where(Job.id == job_id)).first()
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job.id, "status": job.status}


# Endpoint to cancel a job
@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, submitter_id: str = None):
    if not submitter_id:
        raise HTTPException(status_code=400, detail="Submitter ID is required")

    try:
        with Session(engine) as session:
            # Start a transaction
            job = session.get(
                Job, job_id
            )  # Using get instead of exec for primary key lookup
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")

            # Check if the submitter_id matches
            if job.submitter_id != submitter_id:
                raise HTTPException(
                    status_code=403,
                    detail="You can only cancel jobs that you submitted",
                )

            # Only allow cancelling jobs that are queued or evaluating
            if job.status == Status.CANCELLED:
                raise HTTPException(status_code=400, detail="Job is already cancelled")
            elif job.status not in [Status.QUEUED, Status.EVALUATING]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot cancel job with status {job.status}",
                )

            # If job is evaluating, kill the tmux session
            tmux_killed = False
            if job.status == Status.EVALUATING:
                try:
                    server = libtmux.Server()
                    tmux_session_name = f"launcher_{job_id}"
                    tmux_session = server.find_where(
                        {"session_name": tmux_session_name}
                    )
                    if tmux_session:
                        print(f"Killing tmux session {tmux_session_name}")
                        tmux_session.kill_session()
                        tmux_killed = True
                except Exception as e:
                    print(f"Error killing tmux session: {e}")
                    # Continue with cancellation even if tmux kill fails

            # Update job status to cancelled
            job.status = Status.CANCELLED
            job.completed_at = time.time()
            session.add(job)

            # Remove from queue if it's there
            queue_item = session.exec(
                select(JobQueue).where(JobQueue.job_id == job_id)
            ).first()
            if queue_item:
                # Handle case where robot column might not exist in older database versions
                try:
                    robot_info = f" from the {queue_item.robot} queue"
                except AttributeError:
                    robot_info = ""
                print(f"Removing job {job_id}{robot_info}")
                session.delete(queue_item)

            # Commit all changes in a single transaction
            session.commit()

            return {
                "status": "success",
                "message": "Job cancelled successfully"
                + (", evaluation tmux session killed" if tmux_killed else ""),
                "job_status": job.status,
            }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error cancelling job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Admin endpoint to cancel any job
@app.post("/admin/jobs/{job_id}/cancel")
def admin_cancel_job(job_id: str, api_key: str = Depends(verify_api_key)):
    """
    Admin endpoint to cancel any job regardless of submitter.
    Requires admin API key for authentication.
    """
    try:
        with Session(engine) as session:
            # Start a transaction
            job = session.get(Job, job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")

            # Only allow cancelling jobs that are queued or evaluating
            if job.status == Status.CANCELLED:
                raise HTTPException(status_code=400, detail="Job is already cancelled")
            elif job.status not in [Status.QUEUED, Status.EVALUATING]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot cancel job with status {job.status}",
                )

            # If job is evaluating, kill the tmux session
            tmux_killed = False
            if job.status == Status.EVALUATING:
                try:
                    server = libtmux.Server()
                    tmux_session_name = f"launcher_{job_id}"
                    tmux_session = server.find_where(
                        {"session_name": tmux_session_name}
                    )
                    if tmux_session:
                        print(f"[ADMIN] Killing tmux session {tmux_session_name}")
                        tmux_session.kill_session()
                        tmux_killed = True
                except Exception as e:
                    print(f"[ADMIN] Error killing tmux session: {e}")
                    # Continue with cancellation even if tmux kill fails

            # Update job status to cancelled
            job.status = Status.CANCELLED
            job.completed_at = time.time()
            session.add(job)

            # Remove from queue if it's there
            queue_item = session.exec(
                select(JobQueue).where(JobQueue.job_id == job_id)
            ).first()
            if queue_item:
                # Handle case where robot column might not exist in older database versions
                try:
                    robot_info = f" from the {queue_item.robot} queue"
                except AttributeError:
                    robot_info = ""
                print(f"[ADMIN] Removing job {job_id}{robot_info}")
                session.delete(queue_item)

            # Commit all changes in a single transaction
            session.commit()

            return {
                "status": "success",
                "message": f"[ADMIN] Job {job_id} cancelled successfully"
                + (", evaluation tmux session killed" if tmux_killed else ""),
                "job_status": job.status,
            }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ADMIN] Error cancelling job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to get a list of all jobs
@app.get("/jobs/")
def get_jobs():
    with Session(engine) as session:
        jobs = session.exec(select(Job).order_by(Job.submitted_at.desc())).all()
        return jobs


# Endpoint to get robot status
@app.get("/robot_status/")
def get_robot_status():
    with Session(engine) as session:
        statuses = session.exec(select(RobotStatus)).all()
        return statuses


# Endpoint to take a robot offline (admin only)
@app.post("/robot_status/{robot}/offline")
def take_robot_offline(
    robot: str, message: Optional[str] = None, api_key: str = Depends(verify_api_key)
):
    with Session(engine) as session:
        robot_status = session.exec(
            select(RobotStatus).where(RobotStatus.robot == robot)
        ).first()
        if not robot_status:
            raise HTTPException(status_code=404, detail=f"Robot '{robot}' not found")

        robot_status.is_online = False
        robot_status.offline_message = (
            message or "This robot is currently offline and not accepting jobs."
        )
        session.add(robot_status)
        session.commit()

    return {"status": "success", "message": f"Robot '{robot}' has been taken offline"}


# Endpoint to bring a robot back online (admin only)
@app.post("/robot_status/{robot}/online")
def bring_robot_online(robot: str, api_key: str = Depends(verify_api_key)):
    with Session(engine) as session:
        robot_status = session.exec(
            select(RobotStatus).where(RobotStatus.robot == robot)
        ).first()
        if not robot_status:
            raise HTTPException(status_code=404, detail=f"Robot '{robot}' not found")

        robot_status.is_online = True
        robot_status.offline_message = None
        session.add(robot_status)
        session.commit()

    return {
        "status": "success",
        "message": f"Robot '{robot}' has been brought back online",
    }


# Endpoint to take all robots offline (admin only)
@app.post("/robot_status/all/offline")
def take_all_robots_offline(
    message: Optional[str] = None, api_key: str = Depends(verify_api_key)
):
    with Session(engine) as session:
        statuses = session.exec(select(RobotStatus)).all()
        for status in statuses:
            status.is_online = False
            status.offline_message = (
                message or "All robots are currently offline and not accepting jobs."
            )
            session.add(status)
        session.commit()

    return {"status": "success", "message": "All robots have been taken offline"}


##############################################################################
# Job handler
##############################################################################


# Constants for robot rest periods
# need to reset periodically so that the motors do not overheat
ROBOT_MAX_CONTINUOUS_OPERATION_HOURS = 6  # Maximum continuous operation time in hours
ROBOT_REST_PERIOD_MINUTES = 20  # Rest period in minutes

# Worker to process jobs
def job_worker():
    # Dictionary to track which robots are currently executing jobs
    # separate queues for different robots
    robots_running_now = {}

    # Dictionary to track robot operation times
    # Structure: {robot_name: {"start_time": timestamp, "resting_until": timestamp}}
    robot_operation_times = {}

    def _init_operation_time_tracker():
        return {"start_time": None, "resting_until": None}

    while True:
        with Session(engine) as session:
            # Get all unique robots from the queue
            robots_in_queue = session.exec(select(JobQueue.robot).distinct()).all()

            for robot in robots_in_queue:
                # Skip if this robot is already executing a job
                if robots_running_now.get(robot, False):
                    continue

                # Initialize robot tracking if not already tracked
                if robot not in robot_operation_times:
                    robot_operation_times[robot] = _init_operation_time_tracker()

                # Check if robot is in a rest period
                if robot_operation_times[robot]["resting_until"] is not None:
                    current_time = time.time()
                    if current_time < robot_operation_times[robot]["resting_until"]:
                        # Robot is still resting, skip this robot
                        rest_remaining = (
                            robot_operation_times[robot]["resting_until"] - current_time
                        )
                        print(
                            f"Robot {robot} is resting. {rest_remaining:.1f} seconds remaining."
                        )
                        continue
                    else:
                        # Rest period is over, reset operation time
                        print(
                            f"Robot {robot} has completed its rest period. Resuming operations."
                        )
                        robot_operation_times[robot] = _init_operation_time_tracker()

                        # Turn torque back on for the robot
                        try:
                            robot_ip = getattr(RobotIPs, robot.upper())
                            print(
                                f"Turning torque ON for robot {robot} at IP {robot_ip}"
                            )
                            interface = create_interface(robot_ip)
                            torque_on(interface)
                            print(f"Successfully turned torque ON for robot {robot}")
                        except Exception as e:
                            print_red(f"Error turning torque ON for robot {robot}: {e}")
                        time.sleep(5)  # wait before opening tmux

                # Select the oldest job for this specific robot
                queue_entry = session.exec(
                    select(JobQueue)
                    .where(JobQueue.robot == robot)
                    .order_by(JobQueue.job_id)
                ).first()

                if queue_entry:
                    job_id = queue_entry.job_id
                    job = session.get(Job, job_id)

                    # Skip cancelled jobs
                    if job and job.status == Status.CANCELLED:
                        session.delete(queue_entry)
                        session.commit()
                        continue

                    # Mark this robot as executing
                    robots_running_now[robot] = True

                    # Update operation tracking - start tracking time if this is the first job after a rest
                    if robot_operation_times[robot]["start_time"] is None:
                        robot_operation_times[robot]["start_time"] = time.time()

                    # Process the job in a separate thread to allow concurrent execution
                    def execute_and_cleanup(job_id, robot):
                        try:
                            executing_job(job_id)
                        except Exception as e:
                            print(f"Error executing job {job_id}: {e}")
                        finally:
                            # make the robot rest if the robot has been operating for too long
                            if (
                                robot_operation_times[robot]["start_time"] is not None
                                and (
                                    time.time()
                                    - robot_operation_times[robot]["start_time"]
                                )
                                >= ROBOT_MAX_CONTINUOUS_OPERATION_HOURS * 3600
                            ):
                                # robot needs a rest period
                                robot_operation_times[robot]["resting_until"] = (
                                    time.time() + ROBOT_REST_PERIOD_MINUTES * 60
                                )
                                print(
                                    f"Robot {robot} has been operating for over {ROBOT_MAX_CONTINUOUS_OPERATION_HOURS} hours. "
                                )
                                print(
                                    f"Starting {ROBOT_REST_PERIOD_MINUTES} minute rest period."
                                )

                                # Put the robot to sleep and turn off torque
                                try:
                                    robot_ip = getattr(RobotIPs, robot.upper())
                                    print(
                                        f"Putting robot {robot} to sleep and turning torque OFF at IP {robot_ip}"
                                    )
                                    interface = create_interface(robot_ip)
                                    sleep_and_torque_off(interface)
                                    print(
                                        f"Successfully put robot {robot} to sleep and turned torque OFF"
                                    )
                                except Exception as e:
                                    print_red(
                                        f"Error putting robot {robot} to sleep: {e}"
                                    )

                            # Remove job from queue after processing or if it failed
                            with Session(engine) as cleanup_session:
                                queue_entry = cleanup_session.exec(
                                    select(JobQueue).where(JobQueue.job_id == job_id)
                                ).first()
                                if queue_entry:
                                    cleanup_session.delete(queue_entry)
                                    cleanup_session.commit()
                            # Mark robot as available again
                            robots_running_now[robot] = False

                    # Start job execution in a separate thread
                    job_thread = Thread(
                        target=execute_and_cleanup, args=(job_id, robot)
                    )
                    job_thread.daemon = True
                    job_thread.start()

        time.sleep(2)


# Synchronous job processing
def executing_job(job_id: str):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        session.refresh(job)  # Refresh to get the latest state

        # Check if job was cancelled
        if job.status == Status.CANCELLED:
            print(f"Job {job_id} was cancelled, skipping execution")
            return

        job.status = Status.EVALUATING
        job.executed_at = time.time()
        session.add(job)
        session.commit()

        # copy the job config to pass to the launcher
        config = {
            "job_id": job_id,
            "robot": job.robot,
            "task": job.task,
            "policy_server_ip": job.policy_server_ip,
            "policy_server_port": job.policy_server_port,
            "num_episodes": job.num_episodes,
            "max_steps": job.max_steps,
            "description": job.description,
        }
        config = copy.deepcopy(config)

    # Create a new tmux session for this job
    tmux_session_name = f"launcher_{job_id}"
    server = libtmux.Server()

    output_file = f"/tmp/tmux_output_{job_id}.txt"
    result_file = f"/tmp/launcher_result_{job_id}.txt"
    try:
        # Create a new session
        session = server.new_session(
            session_name=tmux_session_name, kill_session=True, attach=False
        )
        if not session:
            raise Exception("Failed to create tmux session")

        window = session.attached_window
        pane = window.attached_pane

        # Check if conda environment exists and activate it
        pane.send_keys("conda env list | grep autoeval", enter=True)
        time.sleep(1)  # Wait for the command to complete
        output = pane.capture_pane()
        if "autoeval" not in "\n".join(output):
            raise Exception("Conda environment 'autoeval' not found")

        pane.send_keys("conda activate autoeval", enter=True)
        time.sleep(1)  # Wait for activation

        # Set up output capture using tmux's built-in pipe-pane command
        pane.cmd("pipe-pane", f"cat > {output_file}")

        # Import and run the launcher directly in the tmux pane
        setup_commands = [
            "import json",
            "from auto_eval.web_ui.launcher import Launcher",
            f"config = {repr(config)}",
            "launcher = Launcher(config)",
            f"result = launcher.run(); print('RESULT_START'); print(json.dumps(result)); print('RESULT_END'); f = open('{result_file}', 'w'); json.dump(result, f); f.close(); exit(0)",
        ]

        # Send Python commands to the pane
        pane.send_keys("python3", enter=True)
        time.sleep(1.0)  # Wait for Python to start
        for cmd in setup_commands:
            pane.send_keys(cmd, enter=True)
            time.sleep(0.1)

        # Wait for the result file to appear and read it
        status = Status.FAILED  # default status
        result = None
        while True:
            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    try:
                        result = json.loads(f.read().strip())
                    except json.JSONDecodeError as e:
                        print(f"Error decoding result JSON: {e}")
                        result = {
                            "status": Status.FAILED,
                            "wandb_url": None,
                            "success_rate": None,
                        }
                os.remove(result_file)  # Clean up
                break

            # Check if job was cancelled
            with Session(engine) as check_session:
                current_job = check_session.get(Job, job_id)
                if current_job and current_job.status == Status.CANCELLED:
                    print(
                        f"Job {job_id} was cancelled while waiting for result file, exiting"
                    )
                    return

            time.sleep(1)

    except Exception as e:
        error_msg = f"Error in executing job {job_id}: {str(e)}"
        print(error_msg)

        # Get the current pane output for debugging
        try:
            if "pane" in locals():
                current_output = pane.capture_pane()
                debug_info = "\n".join(current_output)
            else:
                debug_info = "No pane available"
        except:
            debug_info = "Failed to capture pane output"

        # Write both error and debug info to output file
        with open(output_file, "a") as f:
            f.write(f"\nError: {error_msg}\n")
            f.write(f"\nDebug Info:\n{debug_info}")

        status = Status.FAILED

    finally:
        # Clean up the tmux session
        try:
            server.kill_session(tmux_session_name)
        except Exception as e:
            print(f"Error cleaning up tmux session: {str(e)}")

    with Session(engine) as session:
        job = session.get(Job, job_id)
        # Use the status from the result if available, otherwise use the status from the status file
        if isinstance(result, dict) and "status" in result:
            job.status = result["status"]
        else:
            job.status = status
        job.completed_at = time.time()
        if isinstance(result, dict):
            job.wandb_url = result.get("wandb_url")
            job.success_rate = result.get("success_rate")
        # Read the output file content and store it in the job
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                job.tmux_output = f.read()
            # Clean up the output file after reading its content
            os.remove(output_file)
        else:
            job.tmux_output = "No output file was generated"
        session.add(job)
        session.commit()


# Start the job worker thread
worker_thread = Thread(target=job_worker, daemon=True)
worker_thread.start()


##############################################################################
# Image Upload
##############################################################################


@app.post("/upload/{robot_idx}")
async def upload_image(robot_idx: str, type: str, file: UploadFile = File(...)):
    robot_idx = int(robot_idx)
    if not is_valid_image_type(type):
        raise HTTPException(status_code=400, detail="Invalid image type")

    if not file:
        raise HTTPException(status_code=400, detail="No file part")

    filename = f"{type}_{robot_idx}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            f.write(contents)
            portalocker.unlock(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    return {
        "message": "Image uploaded successfully",
        "type": type,
        "robot_index": robot_idx,
    }


@app.get("/images/{robot_idx}/{image_type}")
async def get_latest_image(robot_idx: str, image_type: str):
    robot_idx = int(robot_idx)
    if not is_valid_image_type(image_type):
        print(f"Invalid image type: {image_type}")
        raise HTTPException(status_code=400, detail="Invalid image type")

    image_path = os.path.join(UPLOAD_FOLDER, f"{image_type}_{robot_idx}.jpg")
    if os.path.exists(image_path):
        return FileResponse(image_path)
    else:
        print(f"Image not found at path: {image_path}")
        raise HTTPException(status_code=404, detail="Image not found")


@app.post("/update_status/{robot_idx}")
async def update_status(robot_idx: str, data: dict):
    # Get current status or initialize if not exists
    current_status = robot_video_feeds[f"feed{robot_idx}"].get("status", {})

    # Update status fields that are present in the data
    if "language_instruction" in data:
        current_status["language_instruction"] = data.get("language_instruction")
    if "episode" in data:
        current_status["episode"] = data.get("episode")
    if "timestep" in data:
        current_status["timestep"] = data.get("timestep")

    # Update waiting_for_human flag if present
    if "waiting_for_human" in data:
        current_status["waiting_for_human"] = data.get("waiting_for_human")

    # Save the updated status
    robot_video_feeds[f"feed{robot_idx}"]["status"] = current_status

    return {"message": "Status updated successfully"}


@app.get("/get_status_data/{robot_idx}")
async def get_status_data(robot_idx: str):
    return robot_video_feeds[f"feed{robot_idx}"]["status"]
