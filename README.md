# My integer sequences. #

These are my integer sequences that at the day of this publishing are not in the OEIS database.

They are lincensed under the [Creative Commons Attribution Share-Alike 4.0 license (CC-BY-SA-4.0).](https://creativecommons.org/licenses/by-sa/4.0/)




## Number of distinct values of f(x,y) = x*y mod x+y for x,y in the range [1,n]. ##

### DATA ###
`1, 3, 4, 6, 7, 9, 10, 10, 13, 15, 16, 18, 21, 21, 23, 24, 26, 28, 30, 31, 32, 35, 37, 38, 42, 43, 44, 46, 47, 48, 51, 52, 54, 57, 58, 61, 64, 65, 67, 68, 70, 72, 75, 76, 77, 78, 79, 81, 82, 85, 86, 90, 92, 93, 95, 97, 98, 100, 102, 104, 106, 107, 109, 111, 113`

### OFFSET ###
1

### COMMENTS ###
```
a(n) is simetric because f(x,y) = f(y,x).
a(n) grows ∼ n + floor(n/2).
a(n) <= 2*n-1.
```

### CODE ###
```
(Python)
def a(n):
  D = set()
  for x in range(1,n+1):
    for y in range(x,n+1):
      D.add((x*y) % (x+y))
  return len(D)
  print([a(n) for n in range(1,66)])
```



## a(n) is the order of the Hadamard Matrix not constructible with Sylvester's or Paley's methods alone. ###

### DATA ###
`40, 52, 56, 88, 92, 96, 100, 112, 116, 120, 136, 144, 156, 160, 172, 176, 184, 188, 208, 216, 232, 236, 244, 248, 260, 268, 280, 288, 292, 296, 304, 320, 324, 328, 336, 340, 344, 352, 356, 372, 376, 392, 400, 404, 408, 412, 416, 424, 428, 436, 448, 452, 456, 472`

### OFFSET ###
1

### LINKS ###
```
Eric Weisstein's World of Mathematics, <a href="https://mathworld.wolfram.com/HadamardMatrix.html">Hadamard Matrix</a>.
Wikipedia, <a href="https://en.m.wikipedia.org/wiki/Hadamard_matrix">Hadamard Matrix</a>.
```

### XREF ###
Cf. A003432.

## Sum of distinct values of the quadratic discriminant D=b^2-4ac, for a,b,c in the range [-n,n]. ##

### DATA ###
`3, 27, 121, 271, 785, 1497, 3102, 4646, 7191, 11040, 17042, 22212, 33547, 46227, 60289, 72515, 102447, 120770, 154047, 180511, 222252, 268785, 349006, 386829, 461894, 531422, 620106, 703939, 836929, 941338, 1134204, 1239196, 1400083, 1632960, 1818962, 1936320, 2270656`

### OFFSET ###
1

### COMMENTS ###
Conversely the count of distinct values of the quadratic discriminant D=b^2-4ac, for a,b,c in the range [-n,n] is A384666.

### CODE ###
```
(Python)
def a(n):
    D, ac = {0}, {0}
    SQ = [i*i for i in range(0, n+1)]
    for i in range(1, n+1):
        ac.add(i)
        if (s:= SQ[i]) > n:
            ac.add(s)
    if n > 2:
        for a_ in range(2, n):
            for c in range(a_ + 1, n + 1):
                ac.add(a_ * c)
    for b in range(n + 1):
        b2 = SQ[b]
        for v in ac:
            ac4 = v << 2
            D.add(b2 + ac4)
            if b2 < ac4:
                D.add(b2 - ac4)
    return sum(D)
print([a(n) for n in range(1, 38)])
```

### XREF ###
Cf. A384666.



## The totient of the product of the totients of the unitary divisors of n. ##

### DATA ###
`1, 1, 1, 1, 2, 2, 2, 2, 2, 8, 4, 8, 4, 12, 32, 4, 8, 12, 6, 32, 48, 40, 10, 32, 8, 48, 6, 48, 12, 2048, 8, 8, 160, 128, 192, 48, 12, 108, 192, 128, 16, 6912, 12, 160, 192, 220, 22, 128, 12, 160, 512, 192, 24, 108, 640, 192, 432, 336, 28, 32768, 16, 240, 432, 16`

### OFFSET ###
1

### FORMULA ###
```
a(p) = totient(Product{d|n} totient(d) if gcd(d, n/d)=1).
a(n) = totient(totient(n)^(2^(omega(n)-1))).
a(n) = A000010(A384763(n)).
a(p) = A008330(p) for p prime.
```

### CODE ###
```
(Python)
from sympy import totient, divisors, gcd
def a(n):
   prod = 1
   for d in divisors(n):
      if gcd(d, n//d) == 1:
          prod *= totient(d)
   return totient(prod)
print([a(n) for n in range(1, 65)])
(Python)
from sympy import factorint, divisors, gcd, totient, prod
def a(n):
    if n == 1: return 1
    pe = {}
    for d in divisors(n):
        if gcd(d, n // d) == 1:
            for p, e in factorint(totient(d)).items():
                pe[p] = pe.get(p, 0) + e
    return prod(p**(e - 1) * (p - 1) for p,e in pe.items())
print([a(n) for n in range(1, 65)])
(Python)
from sympy import factorint, divisors, gcd, totient, prod
def a(n):
    if n == 1: return 1
    pe = []
    for d in divisors(n):
        if gcd(d, n // d) == 1:
            pe.extend(factorint(totient(d)).items())
    primes = sorted(set(p for p, _ in pe))
    return prod(p**(sum(e for q, e in pe if q == p) - 1) * (p - 1) for p in primes)
print([a(n) for n in range(1, 65)])
```

### XREF ###
Cf. A000010, A008330, A384763.



## The integer representation of the reversal of the Reed-Muller code of size 2^(n+1)-1. ##

### DATA ###
`1, 14, 3820, 4006538480, 1127740325610919595933440, 5855562549912621432400532814181205703033719227392014090240, 678027821314169029533837277126308108243817843666549070645730770517828410950207716447345344965940166970542012394294840655177503788236800`

### COMMENTS ###
```
Reed-Muller codes are created such that H(0) is 1 and H(n) is a concatenation equal to: 2^(n-1) zeros plus 2^(n-1) ones plus two copies of H(n-1).
Typically, these codes contain leading zeros. To avoid ignoring them and loss of general information, we reverse the code, since they always end in ones.
A self-similar structure can easily be observed in the binary expansion of a(n).
The bitsize of a(n) is n*2^(n-1).
```

### OFFSET ###
0

### FORMULA ###
`a(n+1) mod a(n) = A111403(n) for n >= 1.`

### EXAMPLE ###
```
The Reed-muller codes are:
 n | H(n)                                | reversed                         | a(n)
---+-------------------------------------+----------------------------------+------
 0 | 1                                   | 1                                | 1
 1 | 0 1 11                              | 1110                             | 14
 2 | 00 11 0111 0111                     | 111011101100                     | 3820
 3 | 0000 1111 001101110111 001101110111 | 11101110110011101110110011110000 | 4006538480
```

### LINK ###
Youtube, <a href="https://www.youtube.com/watch?v=CtOCqKpti7s">Reed-Muller Code (64 Shades of Grey pt2) - Computerphile</a>

### CODE ###
```
(Python)
from functools import cache
@cache
def H(n):
  if n == 0: return "1"
  m =  (1 << (n-1))
  prev = H(n-1)
  return m * "0" + m * "1" + 2*prev
a = lambda n: int(H(n)[::-1],2)
print([a(n).bit_length() for n in range(7)])
```

### KEYWORD ###
base

### XREF ###
Cf. A000225, A001787, A036289.



## Number of distinct subsets S of [1..n] such that for all 1 <= k <= n, there exists two elements x,y in S (not necessarily distinct) such that x*y = k^2. ##

### DATA ###
`0, 1, 1, 1, 2, 2, 2, 2, 3, 10, 10, 10, 11, 11, 11, 11, 37, 37, 40, 40, 80, 80, 80, 80, 80, 592, 592, 1076`

### OFFSET ###
0

### EXAMPLE ###
```
For n=5 a(5) = 2, because there are three sets that matches the said condition:
{1, 3, 4, 5} and {1, 2, 3, 4, 5}
```

### CODE ###
```
(Python)
def a(n):
    t = set(k*k  for k in range(1, n+1))
    c = 0
    for i in range(1, (1 << n)+1, 2):
        s = [j+1 for j in range(n) if (i >> j) & 1]
        if len(s) == 0 or s[0] != 1 or s[-1] != n: continue
        ss = set(x * y for x in s for y in s)
        if t.issubset(ss):
            c += 1
    return c
```

### KEYWORD ###
nonn,more

### XREFS ###
Cf. A383968.



## Number of remainders n mod p greater than zero, over all primes p < n. ##

### DATA ###
`1, 1, 2, 2, 3, 2, 4, 4, 4, 3, 5, 4, 6, 5, 5, 6, 7, 6, 8, 7, 7, 7, 9, 8, 9, 8, 9, 8, 10, 8, 11, 11, 10, 10, 10, 10, 12, 11, 11, 11, 13, 11, 14, 13, 13, 13, 15, 14, 15, 14, 14, 14, 16, 15, 15, 15, 15, 15, 17, 15, 18, 17, 17, 18, 17, 16, 19, 18, 18, 17, 20, 19, 21`

### OFFSET ###
1

### FORMULA ###
`a(A000040(n)) = n.`

### CODE ###
```
(Python)
from sympy import primerange
def a(n):
    s = 1
    for p in primerange(0, n):
        if p > (n >> 1): s += 1
        elif (n % p) > 0: s += 1
    return s
print([a(n) for n in range(1,74)])
```

### XREF ###
Cf. A000040, A383752.


## Apply Rule 110 as an encoding to the binary expansion of n. ##

### DATA ###
`0, 0, 2, 0, 6, 2, 6, 0, 6, 6, 14, 2, 14, 6, 6, 0, 6, 6, 14, 6, 30, 14, 22, 2, 22, 14, 30, 6, 30, 6, 6, 0, 6, 6, 14, 6, 30, 14, 22, 6, 54, 30, 62, 14, 62, 22, 38, 2, 38, 22, 46, 14, 62, 30, 54, 6, 54, 30, 62, 6, 30, 6, 6, 0, 6, 6, 14, 6, 30, 14, 22, 6, 54, 30`

### COMMENTS ###
Leading zeros are omitted.

### LINKS ###
Wikipedia, <a href="https://en.wikipedia.org/wiki/Rule_110">Rule 110</a>

### EXAMPLE ###
```
For n = 52, a(52) = 14 because:
52 = 110100_2 and
Aplying the Rule 110 we get:
 Current Pattern | new pattern
-----------------+-------------
 110             | 1
  101            | 1
   010           | 1
    100          | 0
     00          | 0
      0          | 0
and 111000_2 = 14.
```

### CODE ###
```
(Python)
def a(n):
    m = int(bin(n)[2:][::-1],2)
    R110 = {0:0,1:1,2:1,3:1,4:0,5:1,6:1,7:0}
    e = 0
    mask = 0b111
    while m:
        m >>= 1
        e |= R110[m & mask]
        e <<= 1
    return e >> 1
print([a(n) for n in range(1,75)])
```

### KEYWORD ###
base


## Number of remainders n mod p equal zero, over all primes p < n. ##

### DATA ###
`1, 1, 1, 2, 1, 3, 1, 2, 2, 3, 1, 3, 1, 3, 3, 2, 1, 3, 1, 3, 3, 3, 1, 3, 2, 3, 2, 3, 1, 4, 1, 2, 3, 3, 3, 3, 1, 3, 3, 3, 1, 4, 1, 3, 3, 3, 1, 3, 2, 3, 3, 3, 1, 3, 3, 3, 3, 3, 1, 4, 1, 3, 3, 2, 3, 4, 1, 3, 3, 4, 1, 3, 1, 3, 3, 3, 3, 4, 1, 3, 2, 3, 1, 4, 3, 3, 3`

### OFFSET ###
1

### FORMULA ###
`a(p) = 2 if p is an odd prime.`

### CODE ###
```
(Python)
from sympy import primerange
def a(n):
    s = 1
    for p in primerange(0, (n >> 1) + 1):
        if (n % p) == 0: s += 1
    return s
print([a(n) for n in range(1,88)])
```

### XREF ###
Cf. A000040, A383752.


## The bit position of the bit flip to obtain the longest run of 1s in the binary expansion of n, 1-indexed and 0 if no bit flip was possible. ##

### DATA ###
`1, 0, 1, 0, 1, 2, 1, 0, 1, 2, 3, 3, 1, 2, 1, 0, 1, 2, 3, 3, 4, 2, 4, 4, 1, 2, 3, 3, 1, 2, 1, 0, 1, 2, 3, 3, 4, 2, 4, 4, 5, 5, 3, 3, 5, 5, 5, 5, 1, 2, 3, 3, 4, 2, 4, 4, 1, 2, 3, 3, 1, 2, 1, 0, 1, 2, 3, 3, 4, 2, 4, 4, 5, 2, 3, 3, 5, 2, 5, 5, 6, 6, 6, 3, 4, 2, 4, 4`

### COMMENTS ###
The longest run of 1s in the binary expansion of n is given by A383270(n).

### OFFSET ###
0

### FORMULA ###
```
a(2^k-1) = 0 if k > 0.
a(2^k) = 1
a(2^k+1) = 2 if k > 1.
```

### CODE ###
```
(Python)
def a(n):
    if n == 0: return 1
    if n.bit_length() == n.bit_count(): return 0
    b = c = i = p = m = 0
    while n:
        if n & 1: c += 1
        else:
            p = c * ((n & 2) > 0)
            if (pc := p + c) > m:
                m,b = pc,i
            c = 0
        n >>= 1
        i += 1
    return b+1
print([a(n) for n in range(0,88)])
```

### KEYWORD ###
base

### XREF ###
Cf. A383270.

## Number of distinct values of j^2-i^2 for i,j in range [1,n]. ##

### DATA ###
`1, 2, 4, 7, 11, 16, 21, 26, 32, 41, 47, 56, 65, 74, 86, 97, 107, 118, 130, 144, 158, 173, 187, 204, 221, 238, 255, 272, 288, 309, 328, 347, 366, 389, 411, 434, 456, 479, 504, 530, 553, 581, 605, 633, 662, 689, 717, 747, 774, 804, 834, 868, 896, 931, 968, 1001`

### OFFSET ###
1

### CODE ###
```
(Python)
def a(n):
    s = set()
    for i in range(1,n+1):
      for j in range(i,n+1):
          s.add(j*j-i*i)
    return len(s)
print([a(n) for n in range(1,57)])
```

## Number of solutions wining the Tchoukaillon game with 2n seeds and n pits. ##

### DATA ###
`1, 0, 0, 1, 9, 71, 531, 3836, 27073, 187959, 1289718, 8775209, 59342609, 399533919, 2681325612, 17953216130, 120009760270, 801276639051, 5345587080397, 35642710395824, 237571467879718, 1583179263631879, 10549354995548345, 70293849142393155`

### OFFSET ###
0

### COMMENTS ###
a(n) is the number of permutations of [n+1] with n*(n+1)/2 inversions.

### FORMULA ###
```
a(n) = T(n,2n) with T(i,r) = Sum_{v=0..min(i,r)} T(i-1, r-v) and T(0,r) = 1 if r = 0 else 0.
a(n) = A008302(n+1, A000217(n)) for n >= 2.
```

### LINKS ###
Mancala World, <a href="https://mancala.fandom.com/wiki/Tchoukaillon">Tchoukaillon</a>.

### CODE ###
```
(Python)
from functools import lru_cache
def a(n):
    @lru_cache(maxsize=None)
    def T(i, r):
        if i == 0:
            return 1 if r == 0 else 0
        return sum(T(i - 1, r - v) for v in range(min(i, r) + 1))
    return T(n, n*2)
print([a(n) for n in range(0,24)])
```

### XREF ###
Cf. A000707, A383454.

## Number of solutions wining the Tchoukaillon game with n seeds and n^2 pits. ##

### DATA ###
`1, 1, 9, 155, 3723, 115480, 4405035, 199766491, 10508057625, 629280966619, 42282286220836, 3150585380260000, 257864665508695118, 22998694581983709355, 2220257469063898905802, 230669987024626328456534, 25662670635977625719048303, 3043998217222850740624118838, 383488586060201709909994560725`

### COMMENTS ###
a(n) is the number of permutations of [A098749(n)] with n+1 inversions.

### OFFSET ###
0

### FORMULA ###
```
a(n) = T(n*n,n) with T(x,y) = Sum_{v=0..min(x,y)} T(x-1, y-v) and T(0,y) = 1 if y = 0 else 0.
a(n) = A008302(A098749(n), n+1)
```

### CODE ###
```
(Python)
from functools import lru_cache
def a(n):
    @lru_cache(maxsize=None)
    def T(i, r):
        if i == 0:
            return 1 if r == 0 else 0
        return sum(T(i - 1, r - v) for v in range(min(i, r) + 1) if r - v >= 0)
    return T(n*n, n)
print([a(n) for n in range(0,19)])
```

### XREF ###
Cf. A008302, A098749.

## Concatenation of the interim digits in the Michael Damm error detecting algorithm applied to n. ##

### DATA ###
`0, 3, 1, 7, 5, 9, 8, 6, 4, 2, 31, 37, 35, 30, 39, 38, 33, 34, 32, 36, 17, 10, 19, 12, 11, 15, 14, 18, 16, 13, 78, 79, 74, 75, 73, 76, 72, 70, 71, 77, 53, 56, 57, 54, 52, 50, 59, 55, 58, 51, 92, 95, 98, 91, 94, 93, 96, 97, 99, 90, 89, 84, 83, 88, 86, 81, 87, 82`

### OFFSET ###
0

### COMMENTS ###
Last digit in a(n) is A375584(m).

### LINKS ###
H. Michael Damm, <a href="https://doi.org/10.1016/j.disc.2006.05.033">Totally anti-symmetric quasigroups for all orders n not equal to 2 or 6</a>, Discrete Math., 307:6 (2007), 715-729.
Wikipedia, <a href="https://en.wikipedia.org/wiki/Damm_algorithm">Damm algorithm</a>.

### CODE ###
```
(Python)
t = [
    [0, 3, 1, 7, 5, 9, 8, 6, 4, 2],
    [7, 0, 9, 2, 1, 5, 4, 8, 6, 3],
    [4, 2, 0, 6, 8, 7, 1, 3, 5, 9],
    [1, 7, 5, 0, 9, 8, 3, 4, 2, 6],
    [6, 1, 2, 3, 0, 4, 5, 9, 7, 8],
    [3, 6, 7, 4, 2, 0, 9, 5, 8, 1],
    [5, 8, 6, 9, 7, 2, 0, 1, 3, 4],
    [8, 9, 4, 5, 3, 6, 2, 0, 1, 7],
    [9, 4, 3, 8, 6, 1, 7, 2, 0, 5],
    [2, 5, 8, 1, 4, 3, 6, 7, 9, 0]
]
def a(n):
    i,r,s = 0,0,str(n)
    x = len(s)-1
    for d in s:
        i = t[i][int(d)]
        r += i * (10 ** x)
        x -= 1
    return r
print([a(n) for n in range(0, 68)])
```

### XREF ###
Cf. A375584.

### KEYWORD ###
base

## In the non-adjacent form of n, increment each digit by one then convert to base 10 from base 3. ##

### DATA ###
`0, 1, 7, 21, 22, 23, 64, 66, 67, 68, 70, 192, 193, 194, 199, 201, 202, 203, 205, 210, 211, 212, 577, 579, 580, 581, 583, 597, 598, 599, 604, 606, 607, 608, 610, 615, 616, 617, 631, 633, 634, 635, 637, 1731, 1732, 1733, 1738, 1740, 1741, 1742, 1744, 1749, 1750, 1751, 1792, 1794, 1795, 1796, 1798, 1812, 1813, 1814, 1819, 1821, 1822, 1823, 1825, 1830`

### OFFSET ###
0

### LINK ###
Wikipedia, <a href="https://en.wikipedia.org/wiki/Non-adjacent_form">Non-adjacent form</a>.

### CODE ###
```
(Python)
def g(n):
    if n < 2: return n
    E, Z, tmp = n, [], ""
    while E:
        Zi = 0
        if E & 1:
            Zi = 2 - (E & 3)
            E -= Zi
        tmp = str(Zi+1) + tmp
        E >>= 1
    return int(tmp,3)
print([g(n) for n in range(0,68)])
```

### KEYWORD ###
base

### XREF ###
Cf. A030190, A379015.


## Concatenation of remainders n mod p, over all primes p < n in largest prime to smallest prime order for n > 2 else 0. ##

### DATA ###
`0, 0, 10, 100, 210, 1000, 2110, 13200, 24010, 30100, 41210, 152000, 263110, 1304200, 2410010, 3521100, 4632210, 15743000, 26854110, 137960200, 2481001010, 359012100, 4610123210, 15711234000, 26812340110, 3790451200, 48101562010, 59112603100, 610123714210, 1711134820000`

### OFFSET ###
1

### CODE ###
```
(Python)
from sympy import primerange
def a(n):
  s = "0"
  for p in primerange(0, n):
    s = str(n % p) + s
  return int(s)
print([a(n) for n in range(3,31)])
```

### XREF ###
Cf. A024934.

## Integer encoding of the Huffman-reverse-binary of digit frequency codes from a string concatenated 0 through n-1 in hex. ##

### DATA ###
`0, 2, 28, 228, 4004, 64196, 1027176, 16434824, 534431368, 17103505032, 547312453256, 17513998550664, 560447953628296, 17934334516106504, 573898704515408272, 18364758544493064720, 15905192667998458110, 32091890336705087591, 294840779328134333229, 294840779309540717289`

### OFFSET ###
1

### COMMENTS ###
```
The huffman resulting codes are agnostic to the order of concatenation, It could be 0..(n-1) or (n-1)..0.
Concatenate the hex digits of all numbers from 0 to n-1 into a string, compute the digit frequencies, construct a Huffman code using these frequencies, reverse the binary codes for each digit (in order of increasing digit), concatenate these reversed codes, and interpret the result as a binary number.
```

### LINKS ###
Wikipedia, <a href="https://en.wikipedia.org/wiki/Huffman_coding">Huffman coding</a>

### EXAMPLE ###
```
For n = 5, a(5) = 4004 because:
'01234' has a the following Huffman coding: {'2':'00','3':'01','4':'10','0':'110','1':'111'},
and the reversed and concatenated codes: '111110100100_2 = 4004.
```

### CODE ###
```
(Python)
from heapq import heappush, heappop, heapify
from collections import defaultdict
def encode(S):
    if len(S) < 2: return [(s, '0') for s in S]
    h = [[w, [s, ""]] for s, w in S.items()]
    heapify(h)
    while len(h) > 1:
        lo, hi = heappop(h), heappop(h)
        for p in lo[1:]: p[1] = '0' + p[1]
        for p in hi[1:]: p[1] = '1' + p[1]
        heappush(h, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heappop(h)[1:], key=lambda p: (len(p[-1]), p))
def a(n):
    t = "".join([hex(x)[2:] for x in range(0,n)])
    s = defaultdict(int)
    for c in t: s[c] += 1
    return int("".join([p[1] for p in encode(s)][::-1]),2)
print([a(n) for n in range(1,21)])
```

### KEYWORDS ###
base

## The Hamming(7,4) error-correcting code encoding of n. ##

### DATA ###
`0, 7, 28, 15, 56, 45, 30, 11, 112, 25, 90, 51, 60, 85, 22, 127, 448, 195, 100, 359, 360, 107, 204, 463, 240, 499, 340, 87, 88, 347, 508, 255, 896, 645, 390, 131, 200, 461, 718, 971, 720, 981, 214, 467, 408, 157, 926, 667, 480, 229, 998, 739, 680, 941, 174, 427, 176, 437, 694`

### COMMENTS ###
```
This is essentialy Hamming(7,4) with leading zeros omitted.
a(n) always has an even number of bits set.
```

### LINKS ###
Wikipedia, <a href="https://en.wikipedia.org/wiki/Hamming_code">Hamming code</a>

### OFFSET ###
0

### EXAMPLE ###
```
For n = 11, a(11) = 51 because 11 = 1011_2 and
floor(log2(11)) + 1 = 4 and
p = 3 such that 2^3 >= 4 + 3 + 1, p = A320065(n+4), and codeword = 3+4=7.
Positions:   1 2 3 4 5 6 7
Data:        ? ? 1 ? 0 1 1
bits set:  3,6,7 and 3 XOR 6 XOR 7 = 2 or 010_2 and
fill parity bits 1,2,4 with Ones
Positions:   1 2 3 4 5 6 7
Parity bits: 0 1 1 0 0 1 1 (total XOR is 0) and
0110011_2 is 51.
```

### FORMULA ###
```
a(2^k) 7*(2^A324540(k+1)) NO!!!!
a(2^k) = 0 (mod 7).
```

### CODE ###
```
(Python)
def a(n):
    if n == 0: return 0
    p, l = 0, n.bit_length()
    while (1 << p) < (l + p + 1): p += 1
    tl, k, ix, cw = l+p, l-1, 0, 0
    for i in range(1, tl + 1):
        if (i & (i - 1)):
            v = (n & (1 << k)) >> k
            k -= 1
            cw += (v << (tl-i))
            if v == 1: ix ^= i
    for i in range(p):
        cw += ((ix >> i) & 1) << (tl-(1 << i))
    return cw
print([a(n) for n in range(0, 59)])
```

### KEYWORD ###
base

### XREF ###
Cf. A000079, A070939, A320065.

## Number of calls to Karatsuba's multiplication algorithm K(x,y) when recursively calculating K(Fibonacci(n),Fibonacci(n+1)) in binary digits. ##

### DATA ###
`1, 1, 1, 1, 1, 1, 4, 4, 4, 7, 4, 4, 10, 10, 10, 13, 13, 13, 16, 22, 22, 28, 22, 25, 25, 25, 34, 37, 34, 37, 37, 40, 43, 40, 49, 49, 52, 61, 58, 64, 70, 64, 67, 67, 70, 76, 79, 85, 73, 100, 88, 88, 97, 94, 91, 106, 100, 106, 106, 115, 112, 112, 97, 118, 142`

### OFFSET ###
1

### COMMENTS ###
```
When x and y are both 2 or more digits and the larger is L digits long, base b = 2^floor(L/2) is chosen as a split point for those digits with x = xhi*b + xlo and y = yhi*b + ylo.
K(x,y) then makes 3 recursive calls to K(xhi,yhi), K(xlo,ylo) and K(xhi+xlo,yhi+ylo) (and those results can be assembled to make product x*y).
The initial K call and all further recursive calls are counted in a(n).
1 initial call and then 3 recursive calls each time means a(n) == 1 (mod 3).
```

### LINKS ###
Wikipedia, <a href="https://en.wikipedia.org/wiki/Karatsuba_algorithm">Karatsuba algorithm</a>

### CODE ###
```
(Python) 
counter = 0
def K(x: int, y: int) -> int:
    global counter; counter += 1
    if x < 10 or y < 10: return x * y
    digits = max(len(bin(x)[2:]), len(bin(y)[2:])) >> 1
    base = 1 << digits
    b1 = base-1
    a,b = x >> digits, x & b1
    c,d = y >> digits, y & b1
    x = K(b, d)
    y = K(a + b, c + d)
    z = K(a, c)
    return z * (1 << (digits << 1)) + (y - z - x) * base + x
def a(n: int) -> int:
    from sympy import fibonacci
    global counter; counter = 0
    K(fibonacci(n), fibonacci(n + 1))
    return counter
print([a(n) for n in range(1, 66)])
```

### XREF ###
Cf. A379740.

### KEYWORD ###
base

