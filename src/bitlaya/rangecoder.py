"""Deterministic binary range coder (LZMA-style carry handling), integer-only.

Probabilities are integers ``p0`` in [1, 2**16 - 1]: P(bit = 0) = p0 / 2**16.
State is Python integers only (32-bit range, 33-bit low), so encoder and decoder
agree exactly on every platform. The LZMA encoder always emits a leading zero
byte; it is omitted from the stream and restored implicitly by the decoder.

Byte accounting: encoded length = (normalization shifts) + 4. The decoder reads
4 bytes to start plus one per normalization shift, so a valid stream is consumed
exactly; leftover bytes are rejected as excess payload.
"""

PROB_BITS = 16
PROB_ONE = 1 << PROB_BITS
TOP = 1 << 24
MASK32 = 0xFFFFFFFF


class RangeEncoder:
    def __init__(self):
        self.low, self.range = 0, MASK32
        self.cache, self.cache_size = 0, 1
        self.output = bytearray()

    def _shift_low(self):
        if self.low < 0xFF000000 or self.low > MASK32:
            carry = self.low >> 32
            byte = self.cache
            while True:
                self.output.append((byte + carry) & 0xFF)
                byte = 0xFF
                self.cache_size -= 1
                if not self.cache_size:
                    break
            self.cache = (self.low >> 24) & 0xFF
        self.cache_size += 1
        self.low = (self.low & 0x00FFFFFF) << 8

    def encode(self, bit: int, p0: int):
        bound = (self.range >> PROB_BITS) * p0
        if bit:
            self.low += bound
            self.range -= bound
        else:
            self.range = bound
        while self.range < TOP:
            self.range = (self.range << 8) & MASK32
            self._shift_low()

    def finish(self) -> bytes:
        for _ in range(5):
            self._shift_low()
        if self.output[0] != 0:
            raise AssertionError("range coder invariant violated: leading byte must be zero")
        return bytes(self.output[1:])


class RangeDecoder:
    def __init__(self, data: bytes):
        if len(data) < 4:
            raise ValueError("truncated range-coded payload")
        self.data, self.position = data, 4
        self.code = int.from_bytes(data[:4], "big")
        self.range = MASK32

    def decode(self, p0: int) -> int:
        bound = (self.range >> PROB_BITS) * p0
        if self.code < bound:
            self.range = bound
            bit = 0
        else:
            self.code -= bound
            self.range -= bound
            bit = 1
        while self.range < TOP:
            if self.position >= len(self.data):
                raise ValueError("range-coded payload exhausted; stream is corrupt")
            self.range = (self.range << 8) & MASK32
            self.code = ((self.code << 8) | self.data[self.position]) & MASK32
            self.position += 1
        return bit

    def finish(self):
        if self.position != len(self.data):
            raise ValueError("excess range-coded payload bytes")
