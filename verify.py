def hex_to_num(s: str, base: int):
    n = len(s)
    if n % l != 0:
        raise ValueError("Input string length must be a multiple of 3")
    result = 0
    power = 0
    for i in range(0, n, l):
        part = s[i : i + l]
        result += int(part, 16) * (base**power)
        power += 1
    return result


def num_to_hex(n: int, base: int) -> str:
    if n < 0:
        raise ValueError("Input number must be non-negative")
    result = []
    while n > 0:
        result.append(f"{n % base:03x}")
        n //= base
    return "".join(result)


if __name__ == "__main__":
    l = 3
    base = 16**l
    a = "e46464031b29564586fedd9c6d751f4c2107525c48420fa2"
    b = "cc4ba835aceb10b52054114954ff8b0f10f86cd330bf1513"
    c = "5987e33a724eafd49cc40d179a5efddde79c4052b61715f0dc56d2d8097ff87d9528f4814522d571c291a1c5f90004f6"
    a = hex_to_num(a, base)
    b = hex_to_num(b, base)
    c = hex_to_num(c, base)
    print("A: ", a)
    print("B: ", b)
    print("Result: ", c)
    print("Answer: ", a * b)
    print("Verification: ", c == a * b)
