from datetime import datetime
import os
import signal
import sys
import threading
from typing import Optional, Any
import requests
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

from env import CONFIG_PATH, DOCUMENTS_PATH

MAX_WORKERS: int = 5

# Global flag to signal thread termination
exit_flag: threading.Event = threading.Event()

# Lock for synchronizing print statements
print_lock: threading.Lock = threading.Lock()

def signal_handler(signum: int, frame: Any) -> None:
    """
    Signal handler for interrupting the program.

    Args:
        signum (int): The signal number.
        frame (Any): Current stack frame.
    """
    with print_lock:
        print("\nInterrupt received, stopping downloads...")
    exit_flag.set()

def check_for_exit() -> None:
    """
    Continuously check for user input to exit the program.
    """
    while not exit_flag.is_set():
        if input() == "q":
            with print_lock:
                print("Exiting...")
            exit_flag.set()
            break

def load_json_file(file_path: str) -> Optional[dict[str, Any]]:
    """
    Load and parse a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Optional[dict[str, Any]]: Parsed JSON data as a dictionary, or None if an error occurs.
    """
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        with print_lock:
            print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError as e:
        with print_lock:
            print(f"Error: The file '{file_path}' contains invalid JSON.")
            print(f"JSON decode error: {str(e)}")
    return None

def download_file(url: str, file_path: str, desc: str) -> None:
    """
    Download a file from a given URL and save it to the specified path.

    Args:
        url (str): URL of the file to download.
        file_path (str): Path where the file will be saved.
        desc (str): Description for the progress bar.
    """
    with requests.get(url, stream=True) as r:
        total_size_in_bytes: int = int(r.headers.get("Content-Length", 0))
        with open(file_path, "wb") as f, tqdm(
            desc=desc,
            total=total_size_in_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
            leave=False
        ) as progress_bar:
            for data in r.iter_content(chunk_size=8192):
                if exit_flag.is_set():
                    f.close()
                    os.remove(file_path)  # Remove partially downloaded file
                    return
                size: int = f.write(data)
                progress_bar.update(size)

def is_remote_file_updated(file_path: str, url: str) -> bool:
    """
    Check if the remote file has been updated compared to the local file.

    Args:
        file_path (str): Path to the local file.
        url (str): URL of the remote file.

    Returns:
        bool: True if the remote file is newer or different in size, False otherwise.
    """
    if not os.path.exists(file_path):
        return True

    local_mtime: float = os.path.getmtime(file_path)
    local_file_last_modified_date: datetime = datetime.fromtimestamp(local_mtime)

    response: requests.Response = requests.head(url, allow_redirects=True)
    last_modified: Optional[str] = response.headers.get("Last-Modified")
    size: Optional[str] = response.headers.get("Content-Length")

    if last_modified:
        remote_file_last_modified_date: datetime = datetime.strptime(last_modified, "%a, %d %b %Y %H:%M:%S GMT")
        if local_file_last_modified_date <= remote_file_last_modified_date:
            return True

    if size:
        remote_file_size: int = int(size)
        local_file_size: int = os.path.getsize(file_path)
        if local_file_size / remote_file_size <= 0.9:
            return True

    return False

def download_single_pdf(pdf: dict[str, str], download_path: str, desc: str) -> None:
    """
    Download a single PDF file if it needs updating.

    Args:
        pdf (dict[str, str]): Dictionary containing 'url' and 'filename' of the PDF.
        download_path (str): Path where the PDF will be saved.
        desc (str): Description for the progress bar.
    """
    if exit_flag.is_set():
        return

    url: str = pdf["url"]
    filename: str = pdf["filename"]
    file_path: str = os.path.join(download_path, filename)

    if is_remote_file_updated(file_path, url):
        download_file(url, file_path, desc)

def download_docs(config: dict[str, list[dict[str, Any]]], download_path: str) -> None:
    """
    Download multiple documents based on the provided configuration.

    Args:
        config (dict[str, list[dict[str, Any]]]): Configuration dictionary containing document information.
        download_path (str): Path where the documents will be saved.
    """
    exit_thread: threading.Thread = threading.Thread(target=check_for_exit)
    exit_thread.daemon = True
    exit_thread.start()

    with print_lock:
        print(f"Queuing up to {MAX_WORKERS} downloads in parallel. Press q and then ENTER to cancel\n")

    max_name_length: int = max(len(doc["name"]) for doc in config["documents"])

    with print_lock:
        print("Following documents are downloaded:")
        for doc in config["documents"]:
            desc: str = f"{doc['name']:<{max_name_length}}"
            print(f" - {desc}\t{len(doc['pdfs'])} document{'s' if len(doc['pdfs']) != 1 else ''}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures: list[Future] = []
        for doc in config["documents"]:
            if exit_flag.is_set():
                break
            desc: str = f"{doc['name']:<{max_name_length}}"
            for pdf in doc["pdfs"]:
                if exit_flag.is_set():
                    break
                future: Future = executor.submit(download_single_pdf, pdf, download_path, desc)
                futures.append(future)

        total_downloads: int = len(futures)
        with tqdm(total=total_downloads, desc="Total", unit="file", unit_scale=True, bar_format="{l_bar}{bar}| {n}/{total} [{rate_fmt}]") as overall_progress:
            for future in as_completed(futures):
                if exit_flag.is_set():
                    break
                try:
                    future.result()
                except Exception as e:
                    with print_lock:
                        print(f"An error occurred: {str(e)}")
                overall_progress.update(1)

        if exit_flag.is_set():
            with print_lock:
                print("\nCancelling all ongoing downloads...")
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False)
            with print_lock:
                print("\nAll downloads stopped.\n")
        else:
            with print_lock:
                print("\nDocuments are now available")

def main(file_path: str, download_path: str) -> None:
    """
    Main function to start the document download process.

    Args:
        file_path (str): Path to the JSON configuration file.
        download_path (str): Path where the documents will be saved.
    """

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    config: Optional[dict[str, Any]] = load_json_file(file_path)

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    if config:
        try:
            download_docs(config, download_path)
        except KeyboardInterrupt:
            with print_lock:
                print("\nKeyboard interrupt received, exiting...")
        finally:
            exit_flag.set()
            threading.Event().wait(1)
            sys.exit(0)
    else:
        with print_lock:
            print("Failed to load JSON file.")

if __name__ == "__main__":
    main(CONFIG_PATH, DOCUMENTS_PATH)
