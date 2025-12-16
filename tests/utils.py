import logging
import random
import subprocess
import time

import httpx

logger = logging.getLogger(__name__)


def run_openmockllm(**kwargs) -> subprocess.Popen:
    """Run the openmockllm process and return the process object."""

    port = random.randint(40000, 41000)

    # Kill any process listening on the specified port
    try:
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True, check=False)
        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                subprocess.run(["kill", "-9", pid], check=False)
            time.sleep(0.5)  # Give time for the port to be released
    except Exception:
        pass  # Ignore errors if lsof is not available or port is already free

    command = ["openmockllm", "--port", str(port)]
    for key, value in kwargs.items():
        command.append(f"--{key}")
        command.append(str(value))

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    url = f"http://localhost:{port}"
    process.url = url
    process.model_name = kwargs.get("model_name", "openmockllm")

    # Wait for the server to be ready with health check
    max_retries = 30  # 30 seconds max wait time
    retry_interval = 1  # Check every second
    for attempt in range(max_retries):
        # Check if process has terminated unexpectedly
        returncode = process.poll()
        if returncode is not None:
            # Process has terminated, try to read stderr for error message
            error_msg = "Unknown error"
            try:
                # Use wait with timeout to avoid blocking indefinitely
                process.wait(timeout=0.1)
                # Process has finished, try to read stderr
                if process.stderr:
                    stderr_data = process.stderr.read()
                    if stderr_data:
                        error_msg = stderr_data.decode(errors="replace")
            except (subprocess.TimeoutExpired, AttributeError):
                # stderr might not be readable or process already finished
                pass
            raise RuntimeError(f"openmockllm process failed to start. Process exited with code {returncode}. Error: {error_msg}")

        try:
            # Check if the server is responding by calling /v1/models endpoint
            response = httpx.get(f"{url}/v1/models", timeout=2)
            if response.status_code == 200:
                logger.info(f"openmockllm server is ready at {url} (attempt {attempt + 1})")
                return process
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError):
            # Server not ready yet, wait and retry
            if attempt < max_retries - 1:
                time.sleep(retry_interval)
            else:
                # Final attempt failed, check process status
                returncode = process.poll()
                if returncode is not None:
                    error_msg = "Unknown error"
                    try:
                        process.wait(timeout=0.1)
                        if process.stderr:
                            stderr_data = process.stderr.read()
                            if stderr_data:
                                error_msg = stderr_data.decode(errors="replace")
                    except (subprocess.TimeoutExpired, AttributeError):
                        pass
                    raise RuntimeError(
                        f"openmockllm process failed to start after {max_retries} attempts. Process exited with code {returncode}. Error: {error_msg}"
                    )
                else:
                    raise RuntimeError(
                        f"openmockllm server at {url} did not become ready after {max_retries} attempts. Process is still running but not responding."
                    )

    return process


def kill_openmockllm(process: subprocess.Popen) -> None:
    process.terminate()
    logger.info(f"openmockllm model - terminated ({process.url} - {process.model_name})")
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
