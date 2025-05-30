def hex_to_num(s: str):
    n = len(s)
    if n % 3 != 0:
        raise ValueError("Input string length must be a multiple of 3")
    result = 0
    power = 0
    for i in range(0, n, 3):
        part = s[i : i + 3]
        result += int(part, 16) * (4096**power)
        power += 1
    return result


def num_to_hex(n: int):
    if n < 0:
        raise ValueError("Input number must be non-negative")
    result = []
    while n > 0:
        result.append(f"{n % 4096:03x}")
        n //= 4096
    return "".join(result)


a = "e46464031b29564586fedd9c6d751f4c2107525c48420fa2"
b = "cc4ba835aceb10b52054114954ff8b0f10f86cd330bf1513"
c = "5987e33a724eafd49cc40d179a5efddde79c4052b61715f0dc56d2d8097ff87d9528f4814522d571c291a1c5f90004f6"
a = hex_to_num(a)
b = hex_to_num(b)
c = hex_to_num(c)
print(a)
print(b)
print(c)
print(a * b)
# print(num_to_hex(a * b))
# print(a * b == c)
