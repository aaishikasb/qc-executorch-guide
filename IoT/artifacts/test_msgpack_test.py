"""Protocol regression checks without a board or Router service."""

import socket
import unittest
from unittest.mock import Mock

import msgpack

from msgpack_test import set_led


class LedClientTests(unittest.TestCase):
    def test_fragmented_reply_and_request_encoding(self):
        reply = msgpack.packb([1, 1, None, None])
        client = Mock()
        client.recv.side_effect = [bytes([byte]) for byte in reply]
        self.assertIsNone(set_led(client, True))
        request = msgpack.unpackb(client.sendall.call_args.args[0], raw=False)
        self.assertEqual(request, [0, 1, "set_led_state", [True]])

    def test_rpc_errors_including_falsey_errors(self):
        for error in ("no handler", "", 0, False):
            with self.subTest(error=error):
                client = Mock()
                client.recv.return_value = msgpack.packb([1, 1, error, None])
                with self.assertRaises(RuntimeError):
                    set_led(client, False)

    def test_rejects_unexpected_responses(self):
        for response in (None, [], [1, 1, None], [1, 2, None, None],
                         [0, 1, None, None], [True, 1, None, None]):
            with self.subTest(response=response):
                client = Mock()
                client.recv.return_value = msgpack.packb(response)
                with self.assertRaises(ValueError):
                    set_led(client, True)

    def test_disconnect_mid_reply(self):
        client = Mock()
        client.recv.side_effect = [b"\x94\x01", b""]
        with self.assertRaises(ConnectionError):
            set_led(client, True)

    def test_timeout_propagates(self):
        client = Mock()
        client.recv.side_effect = socket.timeout("timed out")
        with self.assertRaises(socket.timeout):
            set_led(client, True)


if __name__ == "__main__":
    unittest.main()
