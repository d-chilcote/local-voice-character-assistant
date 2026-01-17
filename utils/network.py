import socket

def get_local_ip() -> str:
    """
    Returns the local IP address of the machine on the network.
    Attempts to connect to a public IP to determine which interface is used.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('8.8.8.8', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP
