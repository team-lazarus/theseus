import socket
import threading
from rich.console import Console
from rich.logging import RichHandler
import logging

import constants as c

# Setup rich logging
console = Console()
logging.basicConfig(
    level=logging.INFO, format="%(message)s", handlers=[RichHandler()]
)
logger = logging.getLogger("tcp-server")

def handle_client(client_socket, address):
    logger.info(f"[+] New connection from {address}")
    try:
        while True:
            message = client_socket.recv(1024).decode("utf-8")
            if not message:
                break
            logger.info(f"[Received] {address}: {message}")
            reply("[PONG] "+message, client_socket)
    except ConnectionResetError:
        logger.warning(f"[!] Connection lost from {address}")
    finally:
        client_socket.close()
        logger.info(f"[-] Disconnected: {address}")

def reply(message, client):
    logger.info(f"[Broadcast] {message}")
    client.send(message.encode("utf-8"))


def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((c.SERVER_IP, c.PORT))
    server.listen(5)
    logger.info(f"[*] Server listening on {c.SERVER_IP}:{c.PORT}")
    
    while True:
        client_socket, address = server.accept()
        thread = threading.Thread(target=handle_client, args=(client_socket, address))
        thread.start()

if __name__ == "__main__":
    start_server()
