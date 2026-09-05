"""Hardware-in-the-loop (HIL) transport abstractions."""

from __future__ import annotations

import collections
import socket
from abc import ABC, abstractmethod
from typing import Any


class Transport(ABC):
    """Abstract base class for HIL communication transports."""

    @abstractmethod
    def send(self, data: bytes) -> int:
        """Send raw bytes across the transport. Return number of bytes sent."""

    @abstractmethod
    def recv(self, timeout: float | None = None) -> bytes | None:
        """Receive raw bytes from the transport. Return None on timeout."""

    @abstractmethod
    def close(self) -> None:
        """Close the transport and release system resources."""

    @abstractmethod
    def reset(self) -> None:
        """Flush buffers and restore transport to initial state."""


class InProcessTransport(Transport):
    """In-memory transport for fast deterministic testing and simulated HIL."""

    def __init__(self, delay_steps: int = 0, drop_rate: float = 0.0) -> None:
        self.delay_steps = int(delay_steps)
        self.drop_rate = float(drop_rate)
        self._tx_queue: collections.deque[bytes] = collections.deque()
        self._rx_queue: collections.deque[bytes] = collections.deque()
        self._delay_buf: list[tuple[int, bytes]] = []
        self._step_count = 0
        self._is_open = True

    def send(self, data: bytes) -> int:
        if not self._is_open:
            raise RuntimeError("Cannot send on closed InProcessTransport")
        b = bytes(data)
        if self.delay_steps <= 0:
            self._rx_queue.append(b)
        else:
            self._delay_buf.append((self._step_count + self.delay_steps, b))
        return len(b)

    def tick(self) -> None:
        """Advance internal delay buffer by one simulation step."""
        self._step_count += 1
        ready = []
        remaining = []
        for delivery_step, data in self._delay_buf:
            if self._step_count >= delivery_step:
                ready.append(data)
            else:
                remaining.append((delivery_step, data))
        self._delay_buf = remaining
        self._rx_queue.extend(ready)

    def recv(self, timeout: float | None = None) -> bytes | None:
        if not self._is_open:
            raise RuntimeError("Cannot recv on closed InProcessTransport")
        if self._rx_queue:
            return self._rx_queue.popleft()
        return None

    def reset(self) -> None:
        self._tx_queue.clear()
        self._rx_queue.clear()
        self._delay_buf.clear()
        self._step_count = 0

    def close(self) -> None:
        self.reset()
        self._is_open = False


class UDPTransport(Transport):
    """UDP socket transport for networked HIL nodes."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        peer_host: str = "127.0.0.1",
        peer_port: int | None = None,
        buffer_size: int = 4096,
    ) -> None:
        self.host = host
        self.port = port
        self.peer_host = peer_host
        self.peer_port = peer_port
        self.buffer_size = buffer_size
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self.host, self.port))
        self.port = self._sock.getsockname()[1]
        self._sock.setblocking(False)
        self._is_open = True

    def send(self, data: bytes) -> int:
        if not self._is_open:
            raise RuntimeError("Cannot send on closed UDPTransport")
        if self.peer_port is None:
            raise RuntimeError("peer_port must be specified to send data via UDPTransport")
        return self._sock.sendto(data, (self.peer_host, self.peer_port))

    def recv(self, timeout: float | None = None) -> bytes | None:
        if not self._is_open:
            raise RuntimeError("Cannot recv on closed UDPTransport")
        self._sock.settimeout(timeout)
        try:
            data, _ = self._sock.recvfrom(self.buffer_size)
            return data
        except (socket.timeout, BlockingIOError, TimeoutError):
            return None

    def reset(self) -> None:
        if self._is_open:
            self._sock.setblocking(False)
            while True:
                try:
                    self._sock.recvfrom(self.buffer_size)
                except (BlockingIOError, socket.timeout, TimeoutError):
                    break

    def close(self) -> None:
        if self._is_open:
            self._sock.close()
            self._is_open = False


class SerialTransport(Transport):
    """Serial transport for microcontroller hardware nodes."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.01,
    ) -> None:
        try:
            import serial
        except ImportError as err:
            raise ImportError(
                "SerialTransport requires the pyserial package. "
                "Install it with: pip install pyserial"
            ) from err
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        self._is_open = True

    def send(self, data: bytes) -> int:
        if not self._is_open:
            raise RuntimeError("Cannot send on closed SerialTransport")
        return self._serial.write(data)

    def recv(self, timeout: float | None = None) -> bytes | None:
        if not self._is_open:
            raise RuntimeError("Cannot recv on closed SerialTransport")
        if timeout is not None:
            self._serial.timeout = timeout
        data = self._serial.read(self._serial.in_waiting or 1)
        return data if len(data) > 0 else None

    def reset(self) -> None:
        if self._is_open:
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()

    def close(self) -> None:
        if self._is_open:
            self._serial.close()
            self._is_open = False
