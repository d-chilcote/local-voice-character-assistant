import os

class stderr_suppressor(object):
    """
    Context manager to suppress stderr (file descriptor 2) at the C-level.
    This effectively silences noises from llama.cpp's Metal initialization.
    """
    def __init__(self):
        self._null_fds = [os.open(os.devnull, os.O_RDWR)]
        self._save_fds = [os.dup(2)]

    def __enter__(self):
        os.dup2(self._null_fds[0], 2)

    def __exit__(self, *_):
        os.dup2(self._save_fds[0], 2)
        for fd in self._null_fds + self._save_fds:
            os.close(fd)
