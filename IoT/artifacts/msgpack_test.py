"""Call the guide's MCU LED handler through the local Arduino Router."""

import argparse
import socket

import msgpack


def set_led(client, state):
    # shortcut: one request per connection; use Arduino Bridge for bidirectional apps.
    client.sendall(msgpack.packb([0, 1, "set_led_state", [state]], use_bin_type=True))
    unpacker = msgpack.Unpacker(raw=False, max_buffer_size=65536)
    while True:
        data = client.recv(4096)
        if not data:
            raise ConnectionError("Router disconnected before replying")
        unpacker.feed(data)
        for response in unpacker:
            if (
                not isinstance(response, list)
                or len(response) != 4
                or type(response[0]) is not int
                or response[0] != 1
                or type(response[1]) is not int
                or response[1] != 1
            ):
                raise ValueError(f"Unexpected RPC response: {response!r}")
            if response[2] is not None:
                raise RuntimeError(f"RPC failed: {response[2]!r}")
            return response[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", choices=("0", "1"), nargs="?", default="1")
    args = parser.parse_args()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(5)
            client.connect("/var/run/arduino-router.sock")
            result = set_led(client, args.state == "1")
    except (OSError, ValueError, RuntimeError, msgpack.UnpackException) as exc:
        parser.exit(1, f"LED request failed: {exc}\n")
    print(f"LED request acknowledged: {result!r}")


if __name__ == "__main__":
    main()
