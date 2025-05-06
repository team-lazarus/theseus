import socket
import json
import logging

logger = logging.getLogger("tcp_client")

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class TCPClient(metaclass=Singleton):
    def __init__(self, host="0.0.0.0", port=4200):
        """Initialize TCP client with server host and port."""
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        self.logger = logging.getLogger("tcp-client")

    def connect(self):
        """Establish connection to the server."""
        logger.debug(f"connecting")
        try:
            self.socket.connect((self.host, self.port))
            self.connected = True
            self.logger.info(f"[INFO] Connected to server at {self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to connect: {e}")
            raise Exception("The TCP Client failed to connect to the server")

    def write(self, data):
        """
        Send data to the server.

        Args:
            data: Data to send (will be converted to JSON if it's a dict)
        """
        logger.debug(f"writing {data=}")
        if not self.connected:
            self.logger.error("[ERROR] Not connected to server")
            raise Exception("The TCP Client is not connected to the server")

        try:
            # Convert dict to JSON string if necessary
            if isinstance(data, dict):
                data = json.dumps(data)

            # Send the data encoded as UTF-8
            self.socket.sendall(data.encode("utf-8"))
            return True
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to send data: {e}")
            self.connected = False
            raise Exception("The TCP Client failed to send data to the server")

    def read(self):
        """
        Read data from the server.

        Returns:
            Decoded message as string or dict (if valid JSON), None if error
        """
        logger.debug(f"reading")
        if not self.connected:
            self.logger.error("[ERROR] Not connected to server")
            raise Exception("The TCP Client is not connected to the server")

        try:
            # Buffer for receiving data
            data = b""

            # Keep reading until we get all the data
            while True:
                chunk = self.socket.recv(4096)
                if not chunk:
                    break
                data += chunk

                # Try to decode and see if we have a complete message
                try:
                    decoded = data.decode("utf-8")
                    # Try to parse as JSON
                    try:
                        return json.loads(decoded)
                    except json.JSONDecodeError:
                        # If not valid JSON, return as string
                        return decoded
                except UnicodeDecodeError:
                    # If we can't decode yet, we need more data
                    continue

        except Exception as e:
            self.logger.error(f"[ERROR] Failed to read data: {e}")
            raise Exception("The TCP Client failed to read data")

    def close(self):
        """Close the connection to the server."""
        self.socket.close()
        self.connected = False


if __name__ == "__main__":
    # Create client instance
    client = TCPClient()

    # Connect to server
    client.connect()

    # Read server response (game state)
    game_state = client.read()

    # Send action
    client.write("1")

    if game_state:
        print("Received game state:", game_state)

    # Close connection when done
    client.close()
