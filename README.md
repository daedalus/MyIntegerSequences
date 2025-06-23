# My integer sequences. #

These are my integer sequences that at the day of this publishing are not in the [OEIS](https://oeis.org) database.

They are lincensed under the [Creative Commons Attribution Share-Alike 4.0 license (CC-BY-SA-4.0).](https://creativecommons.org/licenses/by-sa/4.0/)

For those the they are already, there is this [repo](https://github.com/daedalus/MyOEIS).

For an explanation of why these sequences are here and not in the OEIS, see [Issue #1](https://github.com/daedalus/MyIntegerSequences/issues/1).

## Triangle T(n,k) where each k-th element is the k size MacMahon integer plane partition of n. ##

### DATA ###
`1, 3, 0, 4, 1, 0, 7, 3, 0, 0, 6, 9, 0, 0, 0, 12, 15, 1, 0, 0, 0, 8, 30, 3, 0, 0, 0, 0, 15, 45, 9, 0, 0, 0, 0, 0, 13, 67, 22, 0, 0, 0, 0, 0, 0, 18, 99, 42, 1, 0, 0, 0, 0, 0, 0, 12, 135, 81, 3, 0, 0, 0, 0, 0, 0, 0, 28, 175, 140, 9, 0, 0, 0, 0, 0, 0, 0, 0`

### COMMENTS ###
M(j,n) is the MacMahon plane partition function and is defined as the sum over all partitions of n into exactly j distinct parts, each such partition contributing the product of multiplicities of the parts.

### OFFSET ###
1

### LINK ###
Wikipedia, <a href="https://en.wikipedia.org/wiki/Plane_partition">Plane partition</a>

### FORMULA ###
```
M(1,n) = A000203(n)
M(2,n) = A002127(n) for n>= 3 else 0.
M(3,n) = A002128(n) for n>= 6 else 0.
```

### PROG ###
```
(Python)
def M(j, n):
    dp = [ [0]*(n+1) for _ in range(j+1) ]
    dp[0][0] = 1
    for s in range(1, n+1):
        for l in range(j-1, -1, -1):
            for t in range(n - s + 1):
                if (val := dp[l][t]):
                    for m in range(1, (n - t) // s + 1):
                        dp[l+1][t + m * s] += val * m
    return dp[j][n]
row = lambda n: [M(k,n) for k in range (1,n+1)]
```

### KEYWORD ###
tabl

### XREF ###
Cf. A000203, A002127, A002128.



## a(n) = (n^2-3n+2)*M(1,n) - 8*M(2,n). ##

### DATA ###
`0, 0, 0, 18, 0, 120, 0, 270, 192, 504, 0, 1680, 0, 1296, 1536, 2790, 0, 5160, 0, 6804, 3840, 4680, 0, 16800, 2880, 7560, 7680, 17280, 0, 30960, 0, 25110, 13440, 16416, 13824, 59490, 0, 22680, 21504, 66780, 0, 78720, 0, 61776, 59136, 39600, 0, 148800, 16128, 86184`

### COMMENTS ###
M(j,n) is defined as the sum over all partitions of n into exactly j distinct parts, each such partition contributing the product of multiplicities of the parts.

### FORMULA ###
```
a(n) = 0 iff n is prime.
M(1,n) = A000203(n)
M(2,n) = A002127(n) for n>= 3 else 0.
a(n) = A279019(n-4)*M(1,n) - 8*M(2,n).
```

### OFFSET ###
1

### LINK ###
William Craig, Jan-Willem van Ittersum, Ken Ono, <a href="https://arxiv.org/abs/2405.06451">Integer partitions detect the primes</a>

### PROG ###
```
(Python)
def M(j, n):
    dp = [ [0]*(n+1) for _ in range(j+1) ]
    dp[0][0] = 1
    for s in range(1, n+1):
        for l in range(j-1, -1, -1):
            for t in range(n - s + 1):
                if (val := dp[l][t]):
                    for m in range(1, (n - t) // s + 1):
                        dp[l+1][t + m * s] += val * m
    return dp[j][n]
def a(n):
    return (n**2-3*n+2)*M(1,n) - 8*M(2,n)
print([a(n) for n in range(1,51)])
```

### XREF ###
Cf. A000203, A002127, A002128, A279019.



## a(n) = (3n^3 - 13n^2 + 18n - 8) * M(1,n) + (12n^2 -120n + 212)* M(2,n) - 960*M(3,n) with M(j,n) the MacMahon partition function. ##

### DATA ###
`0, 0, 0, 108, 0, 1260, 0, 4860, 2592, 14364, 0, 51660, 0, 75816, 43776, 169020, 0, 367380, 0, 551124, 213120, 723060, 0, 1745100, 108000, 1666980, 725760, 2854440, 0, 5059800, 0, 5525820, 1955520, 6377616, 808704, 13324320, 0, 11124540, 4483584, 18193140, 0`

### COMMENTS ###
M(j,n) is defined as the sum over all partitions of n into exactly j distinct parts, each such partition contributing the product of multiplicities of the parts.

### FORMULA ###
```
a(n) = 0 iff n is prime.
M(1,n) = A000203(n)
M(2,n) = A002127(n) for n>= 3 else 0.
M(3,n) = A002128(n) for n>= 6 else 0.
```

### OFFSET ###
1

### LINK ###
William Craig, Jan-Willem van Ittersum, Ken Ono, <a href="https://arxiv.org/abs/2405.06451">Integer partitions detect the primes</a>

### PROG ###
```
(Python)
def M(j, n):
    dp = [ [0]*(n+1) for _ in range(j+1) ]
    dp[0][0] = 1
    for s in range(1, n+1):
        for l in range(j-1, -1, -1):
            for t in range(n - s + 1):
                if (val := dp[l][t]):
                    for m in range(1, (n - t) // s + 1):
                        dp[l+1][t + m * s] += val * m
    return dp[j][n]
def a(n):
    term1 = M(1, n) * (3 * n**3 - 13 * n**2 + 18 * n - 8)
    term2 = M(2, n) * (12 * n**2 - 120 * n + 212)
    term3 = M(3, n) * 960
    return term1 + term2 - term3
print([a(n) for n in range(1,42)])
````

### XREF ###
Cf. A000203, A002127, A002128.



## a(n) is the sum of pairs x+y such that (x^2+y^2)/(xy+1) is square for x,y in [0, n-1]. ##

### DATA ###
`0, 4, 8, 14, 22, 32, 44, 58, 94, 112, 132, 154, 178, 204, 232, 262, 294, 328, 364, 402, 442, 484, 528, 574, 622, 672, 724, 838, 894, 952, 1088, 1150, 1214, 1280, 1348, 1418, 1490, 1564, 1640, 1718, 1798, 1880, 1964, 2050, 2138, 2228, 2320, 2414, 2510, 2608, 2708`

### COMMENT ###
The function is symmetric in x,y because (x,y) = (y,x).

### FORMULA ###
`a(n) = Sum_{x=0..n-1} Sum_{y=0..n-1} x+y iif (x^2 + y^2)/(xy+1) is square.`

### LINKS ###
Numberphile, <a href="https://www.youtube.com/watch?v=NcaYEaVTA4g">The notorious question six (solved by induction)</a>.

### PROG ###
```
(Python)
from sympy.ntheory.primetest import is_square
def a(n):
  if n == 1: return 0
  c = 0
  for x in range(1,n):
    x2 = x*x
    for y in range(x+1,n):
      q,r = divmod(x2 + y*y, x*y + 1) 
      if r == 0 and is_square(q):
        c += (x+y)
  return c*2+2+n*(n-1)
print([a(n) for n in range(1,52)])
```


## Number of solutions that satisfy x^2 + y^2 + w^2 + z^2 = xywz for x,y,w,z in [1,n]. ##

### DATA ###
`0, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17`

### LINKS ###
Numberphile, <a href="https://www.youtube.com/watch?v=a7BVL1MOCl4">A simple equation that behaves strangely</a>.
Wikipedia, <a href="https://en.wikipedia.org/wiki/Vieta_jumping">Vieta jumping</a>.

### FORMULA ###
`a(f(n)+c) = g(n), for c < f(n+1) with f=2*A0752762 and g remaining to be defined.`

### PROG ###
```
(Python)
from itertools import combinations_with_replacement
from math import factorial
from collections import Counter
def a(n):
    if n == 1: return 0
    if n < 6: return 1
    c,f4 = 0, 4*3*2
    F={}
    for x, y, w, z in combinations_with_replacement(range(1, n+1), 4):
        if x**2 + y**2 + w**2 + z**2 == x*y*w*z:
            m = Counter([x, y, w, z])  
            perms = f4
            for v in m.values():
                if v in F:
                    perms //= F[v]
                else:
                    F[v] = factorial(v)
                    perms //= factorial(v)
            c += perms
    return c
print([a(n) for n in range(1,71)])
```

### XREF ###
Cf. A075276.



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

### PROG ###
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
Eric Weisstein's World of Mathematics, <a href="https://mathworld.wolfram.com/HadamardMatrix.html">Hadamard Matrix</a>.
Wikipedia, <a href="https://en.m.wikipedia.org/wiki/Hadamard_matrix">Hadamard Matrix</a>.

### XREF ###
Cf. A003432.



## Sum of distinct values of the quadratic discriminant D=b^2-4ac, for a,b,c in the range [-n,n]. ##

### DATA ###
`3, 27, 121, 271, 785, 1497, 3102, 4646, 7191, 11040, 17042, 22212, 33547, 46227, 60289, 72515, 102447, 120770, 154047, 180511, 222252, 268785, 349006, 386829, 461894, 531422, 620106, 703939, 836929, 941338, 1134204, 1239196, 1400083, 1632960, 1818962, 1936320, 2270656`

### OFFSET ###
1

### COMMENTS ###
Conversely the count of distinct values of the quadratic discriminant D=b^2-4ac, for a,b,c in the range [-n,n] is A384666.

### PROG ###
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

### PROG ###
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



## The integer representation of the reversal of the Reed-Muller PROG of size 2^(n+1)-1. ##

### DATA ###
`1, 14, 3820, 4006538480, 1127740325610919595933440, 5855562549912621432400532814181205703033719227392014090240, 678027821314169029533837277126308108243817843666549070645730770517828410950207716447345344965940166970542012394294840655177503788236800`

### COMMENTS ###
```
Reed-Muller PROGs are created such that H(0) is 1 and H(n) is a concatenation equal to: 2^(n-1) zeros plus 2^(n-1) ones plus two copies of H(n-1).
Typically, these PROGs contain leading zeros. To avoid ignoring them and loss of general information, we reverse the PROG, since they always end in ones.
A self-similar structure can easily be observed in the binary expansion of a(n).
The bitsize of a(n) is n*2^(n-1).
```

### OFFSET ###
0

### FORMULA ###
`a(n+1) mod a(n) = A111403(n) for n >= 1.`

### EXAMPLE ###
```
The Reed-muller PROGs are:
 n | H(n)                                | reversed                         | a(n)
---+-------------------------------------+----------------------------------+------
 0 | 1                                   | 1                                | 1
 1 | 0 1 11                              | 1110                             | 14
 2 | 00 11 0111 0111                     | 111011101100                     | 3820
 3 | 0000 1111 001101110111 001101110111 | 11101110110011101110110011110000 | 4006538480
```

### LINK ###
Youtube, <a href="https://www.youtube.com/watch?v=CtOCqKpti7s">Reed-Muller PROG (64 Shades of Grey pt2) - Computerphile</a>

### PROG ###
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

### PROG ###
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

### PROG ###
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

### PROG ###
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

### PROG ###
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

### PROG ###
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

### PROG ###
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

### PROG ###
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

### PROG ###
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

### PROG ###
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

### PROG ###
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

### PROG ###
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



## Integer encoding of the Huffman-reverse-binary of digit frequency PROGs from a string concatenated 0 through n-1 in hex. ##

### DATA ###
`0, 2, 28, 228, 4004, 64196, 1027176, 16434824, 534431368, 17103505032, 547312453256, 17513998550664, 560447953628296, 17934334516106504, 573898704515408272, 18364758544493064720, 15905192667998458110, 32091890336705087591, 294840779328134333229, 294840779309540717289`

### OFFSET ###
1

### COMMENTS ###
```
The huffman resulting PROGs are agnostic to the order of concatenation, It could be 0..(n-1) or (n-1)..0.
Concatenate the hex digits of all numbers from 0 to n-1 into a string, compute the digit frequencies, construct a Huffman PROG using these frequencies, reverse the binary PROGs for each digit (in order of increasing digit), concatenate these reversed PROGs, and interpret the result as a binary number.
```

### LINKS ###
Wikipedia, <a href="https://en.wikipedia.org/wiki/Huffman_coding">Huffman coding</a>

### EXAMPLE ###
```
For n = 5, a(5) = 4004 because:
'01234' has a the following Huffman coding: {'2':'00','3':'01','4':'10','0':'110','1':'111'},
and the reversed and concatenated PROGs: '111110100100_2 = 4004.
```

### PROG ###
```
(Python)
from heapq import heappush, heappop, heapify
from collections import defaultdict
def enPROG(S):
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
    return int("".join([p[1] for p in enPROG(s)][::-1]),2)
print([a(n) for n in range(1,21)])
```

### KEYWORD ###
base



## The Hamming(7,4) error-correcting PROG encoding of n. ##

### DATA ###
`0, 7, 28, 15, 56, 45, 30, 11, 112, 25, 90, 51, 60, 85, 22, 127, 448, 195, 100, 359, 360, 107, 204, 463, 240, 499, 340, 87, 88, 347, 508, 255, 896, 645, 390, 131, 200, 461, 718, 971, 720, 981, 214, 467, 408, 157, 926, 667, 480, 229, 998, 739, 680, 941, 174, 427, 176, 437, 694`

### COMMENTS ###
```
This is essentialy Hamming(7,4) with leading zeros omitted.
a(n) always has an even number of bits set.
```

### LINKS ###
Wikipedia, <a href="https://en.wikipedia.org/wiki/Hamming_PROG">Hamming PROG</a>

### OFFSET ###
0

### EXAMPLE ###
```
For n = 11, a(11) = 51 because 11 = 1011_2 and
floor(log2(11)) + 1 = 4 and
p = 3 such that 2^3 >= 4 + 3 + 1, p = A320065(n+4), and PROGword = 3+4=7.
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

### PROG ###
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

### PROG ###
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



## Integer encoding of the Huffman-reverse-binary of digit frequency PROGs from a string concatenated 0 through n-1. ##

### DATA ###
`0, 2, 28, 228, 4004, 64196, 1027176, 16434824, 534431368, 17103505032, 17103430188, 34206888202, 25044430919, 25044431395, 22753794595, 20463342115, 548981675858, 549180765488, 532537767216, 549180765488, 25044430919, 25044430919, 45507564723, 40926268323, 36345002643`

### OFFSET ###
1

### COMMENTS ###
```
The huffman resulting PROGs are agnostic to the order of concatenation, It could be 0..(n-1) or (n-1)..0.
Concatenate the digits of all numbers from 0 to n-1 into a string, compute the digit frequencies, construct a Huffman PROG using these frequencies, reverse the binary PROGs for each digit (in order of increasing digit), concatenate these reversed PROGs, and interpret the result as a binary number.
```

### LINKS ###
Wikipedia, <a href="https://en.wikipedia.org/wiki/Huffman_coding">Huffman coding</a>

### EXAMPLE ###
```
For n = 5, a(5) = 4004 because:
'01234' has a the following Huffman coding: {'2':'00','3':'01','4':'10','0':'110','1':'111'},
and the reversed and concatenated PROGs: '111110100100_2 = 4004.
```

### PROG ###
```
(Python)
from heapq import heappush, heappop, heapify
from collections import defaultdict
def enPROG(S):
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
    t = "".join([str(x) for x in range(0,n)])
    s = defaultdict(int)
    for c in t: s[c] += 1
    return int("".join([p[1] for p in enPROG(s)][::-1]),2)
print([a(n) for n in range(1,26)])
```

### KEYWORD ###
base



## Partial products of A217793. ##

### DATA ###
`91, 368016, 7704998400, 154254553560616860000, 81099490326469519642214400, 123904409109840398901842327944396800000, 312980442261030492019371810265757601600000000, 7086288143652192493789225352443309285162175220940800000000, 14310712231229003211358115874216631351811959497046688833146971817246720000000000`

### COMMENTS ###
```
In October of 1941 Paul Erdős and Pál Turán found that a Golomb ruler could be constructed for every odd prime p.
Such a ruler has the property that the mark or notches are defined by: notch(k) = 2pk + (k^2 mod p) for k in {0..p-1}, with p=A000040(n).
```

### FORMULA ###
`a(n) = Product_{k=1..p-1} (2*k*p + (k^2 mod p)), where p is the n-th prime.`

### OFFSET ###
2

### PROG ###
```
(Python)
from sympy import prod, prime
def a(n):
  p = prime(n)
  return prod(2*p*k + pow(k,2,p) for k in range(1, p))
print([a(n) for n in range(2, 11)])
```

### XREF ###
Cf. A000040, A217793, A380790.



## Numbers that can be written in only one way in the form (j+2k)^2-(j+k)^2-j^2 with j,k>0. ##

### DATA ###
3, 4, 7, 11, 12, 16, 19, 20, 23, 28, 31, 43, 44, 47, 48, 52, 59, 67, 68, 71, 76, 79, 80, 83, 92, 103, 107, 112, 116, 124, 127, 131, 139, 148, 151, 163, 164, 167, 172, 176, 179, 188, 191, 199, 208, 211, 212, 223, 227, 236, 239, 244, 251, 263, 268, 271, 272, 283

### OFFSET ###
1

### LINK ###
Project Euler, <a href="https://projecteuler.net/problem=135">Problem 135: Same Differences</a>.

### COMMENTS ###
```
These numbers have a pair of divisors p,q that sum to a multiple of 4.
Numbers congruent {0, 3, 4, 7, 11, 12, 15} mod 16.
Also numbers that can be written in only one way in the form (j+k)*(3k-j) for j,k>0.
```

### PROG ###
```
(Python)
from sympy import divisors
def isok(n):
  s = 0
  for d in divisors(n):
    t = n // d + d
    if ((q:=t >> 2) << 2) == t and q < d:
      s += 1
  return s == 1
print([n for n in range(1, 284) if isok(n)])
```

### XREF ###
Cf. A364168, A383252.



## Numbers that can be written in the form (j+2k)^2-(j+k)^2-j^2 with j,k>0. ##

### DATA ###
`3, 4, 7, 11, 12, 15, 16, 19, 20, 23, 27, 28, 31, 32, 35, 36, 39, 43, 44, 47, 48, 51, 52, 55, 59, 60, 63, 64, 67, 68, 71, 75, 76, 79, 80, 83, 84, 87, 91, 92, 95, 96, 99, 100, 103, 107, 108, 111, 112, 115, 116, 119, 123, 124, 127, 128, 131, 132, 135, 139, 140`

### OFFSET ###
1

### LINK ###
Project Euler, <a href="https://projecteuler.net/problem=135">Problem 135: Same Differences</a>.

### COMMENTS ###
```
These numbers have a pair of divisors p,q that sum to a multiple of 4.
Numbers congruent {0, 3, 4, 7, 11, 12, 15} mod 16.
Also numbers that can be written in the form (j+k)*(3k-j) for j,k>0.
```

### PROG ###
```
(Python)
from sympy import divisors
def isok(n):
  D = divisors(n)
  L = len(D)
  for i in range((L >> 1) + 1):
    p,q = D[i], D[L-i-1]
    if ((p+q) & 3 == 0) and (p <= q):
      return True
  return False
print([n for n in range(1,141) if isok(n)])
```

### XREF ###
Cf. A364168, A383252.



## a(n) is the minimum bucket size in a bucket sort algorithm with input {0, 1, ..., n-1} and floor(sqrt(n)) buckets. ##

### DATA ###
`1, 2, 3, 2, 2, 3, 3, 4, 3, 2, 3, 4, 3, 4, 5, 4, 2, 3, 4, 5, 3, 4, 5, 6, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 6, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8, 7, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 9, 8, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10, 9, 2, 3, 4`

### COMMENTS ###
The maximum and minimum bucket size are equal when n is in A006446.

### FORMULA ###
`a(n) = min(floor((n-1)/sqrt(n))+1, n mod (floor((n-1)/sqrt(n))+1)).`

### OFFSET ###
1

### EXAMPLE ###
```
For n = 10 a(10) = 4 because:
Input array: [0,1,2,3,4,5,6,7,8,9] and floor(sqrt(10)) = 3.
Resulting 3 buckets of [0, 1, 2, 3], [4, 5, 6, 7], [8, 9] and the length of the buckets [4,4,2].
The minimum bucket size is 2.
```

### PROG ###
```
(Python)
from sympy.core.intfunc import isqrt
def a(n):
    bc = isqrt(n)
    bs = ((n-1) // bc) + 1
    fb,r = divmod(n,bs)
    return min(bs, r) if r > 0 else bs
print([a(n) for n in range(1,85)])
```

### XREF ###
Cf. A000079, A000196, A006446.



## The lexicographic rank of the permutation obtained by recording the swaps needed to sort the Eytzinger permutation of [0, 1, ..., n-1] with the bitonic sorter algorithm. ##

### DATA ###
`1, 20, 37610, 20246977580570, 258952989957427698229458143957804570, 125887757413908728356528535566203146374133193857422387130710461384133774303059413717804570, 384108221355416548242103320084870428383288373093396696247459149225011268451060632674249034983367221167680047201563521138868562742195457949673151148273338742934440997616360245085791817113232421743299551059413717804570`

### OFFSET ###
1

### COMMENTS ###
```
The Eytzinger array layout (A375825) arranges elements so that a binary search can be performed starting at element k=1 and at a given k step to 2*k or 2*k+1 according as the target is smaller or larger than the element at k.
The lexicographic rank of a permutation of n elements is its position in the ordered list of all possible permutations of n elements, and here taking the first permutation as rank 0.
```

### LINKS ###
geeksforgeeks.org, <a href="https://www.geeksforgeeks.org/lexicographic-rank-string-duplicate-characters">Lexicographic rank of a String</a>
Sergey Slotin, <a href="https://algorithmica.org/en/eytzinger">Eytzinger binary search</a>
sympy.org, <a href="https://docs.sympy.org/latest/modules/combinatorics/permutations.html#sympy.combinatorics.permutations.Permutation.rank">Permutation rank</a>
Wikipedia, <a href="https://en.wikipedia.org/wiki/bitonic_sorter">bitonic sort</a>

### PROG ###
```
(Python)
from sympy.combinatorics import Permutation
def s(arr):
    n = len(arr)
    R = list(range(n))
    k = 2
    while k <= n:
        j = k >> 1
        while j > 0:
            for i in range(n):
                if i & j == 0:
                    l = i ^ j
                #if l > i:
                    if ((i & k) == 0 and arr[i] > arr[l]) or ((i & k) != 0 and arr[i] < arr[l]):
                        arr[i], arr[l] = arr[l], arr[i]
                        R[i], R[l] = R[l], R[i]
            j >>= 1
        k <<= 1
    return R
def eytzinger(t, k=1, i=0):
    if (k < len(t)):
        i = eytzinger(t, k * 2, i)
        t[k] = i
        i += 1
        i = eytzinger(t, k * 2 + 1, i)
    return i
def a(n):
    def eytzinger(t, k=1, i=0):
        if (k < len(t)):
            i = eytzinger(t, k * 2, i)
            t[k] = i
            i += 1
            i = eytzinger(t, k * 2 + 1, i)
        return i
    t = [0] * ((1 << n) + 1 )
    eytzinger(t)
    return Permutation(s(t[1:])).rank()
print([a(n) for n in range(1,8)])
```

### XREF ###
Cf. A030298, A369802, A370006, A375825, A368783.



## The lexicographic rank of the permutation obtained by recording the swaps needed to sort the Eytzinger permutation of [0, 1, ..., n-1] with the Bubble sort algorithm. ##

### DATA ###
`0, 1, 2, 20, 82, 397, 2330, 37610, 301850, 2692730, 26741138, 292548740, 3495111922, 45271195597, 631862060570, 20246977580570, 324237678994970, 5500423810911770, 98823436151007770, 1874553112933484570, 37436027019862950170, 785121450483596287130, 17252158693640677392410, 396372452178749756086250`

### OFFSET ###
1

### COMMENTS ###
```
The Eytzinger array layout (A375825) arranges elements so that a binary search can be performed starting at element k=1 and at a given k step to 2*k or 2*k+1 according as the target is smaller or larger than the element at k.
The lexicographic rank of a permutation of n elements is its position in the ordered list of all possible permutations of n elements, and here taking the first permutation as rank 0.
```

### LINKS ###
geeksforgeeks.org, <a href="https://www.geeksforgeeks.org/lexicographic-rank-string-duplicate-characters">Lexicographic rank of a String</a>
Sergey Slotin, <a href="https://algorithmica.org/en/eytzinger">Eytzinger binary search</a>
sympy.org, <a href="https://docs.sympy.org/latest/modules/combinatorics/permutations.html#sympy.combinatorics.permutations.Permutation.rank">Permutation rank</a>
Wikipedia, <a href="https://en.wikipedia.org/wiki/Bubble_sort">Bubble sort</a>

### PROG ###
```
(Python)
from sympy.combinatorics import Permutation
def s(arr):
    n = len(arr)
    R = list(range(n))
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                R[j], R[j+1] = R[j+1], R[j]
    return R
def eytzinger(t, k=1, i=0):
    if (k < len(t)):
        i = eytzinger(t, k * 2, i)
        t[k] = i
        i += 1
        i = eytzinger(t, k * 2 + 1, i)
    return i
def a(n):
    t = [0] * (n+1)
    eytzinger(t)
    return Permutation(s(t[1:])).rank()
print([a(n) for n in range(1,25)])
```

### XREF ###
Cf. A030298, A369802, A370006, A375825, A368783.



## The binary expansion of a(n) tracks where the swaps occur to sort the binary expansion of n. ##

### DATA ###
`0, 2, 0, 10, 8, 4, 0, 74, 72, 68, 64, 36, 32, 16, 0, 1098, 1096, 1092, 1088, 1060, 1056, 1040, 1024, 548, 544, 528, 512, 272, 256, 128, 0, 33866, 33864, 33860, 33856, 33828, 33824, 33808, 33792, 33316, 33312, 33296, 33280, 33040, 33024, 32896, 32768, 16932, 16928`

### COMMENTS ###
```
Leading zeros are ommitted in the resulting encoding.
The number of left shifts for a(n) is A000217(floor(log_2(n))+1).
```

### EXAMPLE ###
```
For n = 22, a(22) = 1040 because:
22 is 10110_2 and
 i | j | B[i] | B[j] | Encoding
---+---+------+------+----------
 0 | 1 | 1    | 0    | 1
 0 | 2 | 0    | 1    | 0
 0 | 3 | 0    | 1    | 0
 0 | 4 | 0    | 0    | 0
 1 | 2 | 1    | 1    | 0
 1 | 3 | 1    | 1    | 0
 1 | 4 | 1    | 0    | 1
 2 | 3 | 1    | 1    | 0
 2 | 4 | 1    | 1    | 0
 3 | 4 | 1    | 1    | 0
10110_2 sorted is 00111_2.
And a(n) = 1000 001 00 0 = 1040.
       i =    0   1  2 3
```

### FORMULA ###
`a(2^k) = Sum_{j=1..k-1} 2^((j^2 + 3j - 4)/2 + 3) + 2.`

### PROG ###
```
(Python)
def a(n):
    c, B, lb = 0, list(map(int, bin(n)[2:])), n.bit_length()
    for i in range(lb):
        for j in range(i+1, lb):
            if B[i] > B[j]:
                B[i],B[j] = B[j],B[i]
                c |=1
            c <<= 1
    return c
print([a(n) for n in range(1,50)])
```

### XREF ###
Cf. A000079, A000217, A006125, A070939, A380145.

### KEYWORD ###
base



## Binary left shift XOR sum of n. ##

### DATA ###
`1, 6, 5, 28, 27, 18, 21, 120, 119, 102, 105, 68, 75, 90, 85, 496, 495, 462, 465, 396, 403, 434, 429, 264, 279, 310, 297, 372, 363, 330, 341, 2016, 2015, 1950, 1953, 1820, 1827, 1890, 1885, 1560, 1575, 1638, 1625, 1764, 1755, 1690, 1701, 1040, 1071, 1134`

### FORMULA ###
```
a(2^k+1) = a(2^k) - 1 for k>2.
a(2^k) = A006516(k-1).
```

### EXAMPLE ###
```
for n = 6 a(6)= 18 because 6 in base 2 is 110
and:
  110
 110
110
------
10010
and 10010 in base 10 is 18
```

### PROG ###
```
(Python)
def a(n):
    if (n > 2) and (n - 1) & (n - 2) == 0: return a(n-1)-1
    m = n
    r = n
    for i in range(0, n.bit_length()-1):
        m <<= 1
        r ^= m
    return r
print([a(n) for n in range(1, 51)])
```

### keyword ###
base

### XREF ###
Cf. A000051, A000079, A006516, A378299.



## Lexicographic rank of the bit-reversal permutation of elements {0, 1, ..., 2^n - 1}. ##

### DATA ###
`0, 0, 2, 2354, 633303178034, 4047127158317611833545968021642034, 983558374988244870572855228078991302744595248608705829863205162000316468367968661642034`

### COMMENTS ###
Also the inversion count of the bit reversal permutation of elements {0, 1, ..., 2^n - 1} is A100575(n).

### EXAMPLE ###
```
| n | 2^n | Sequence                                              | rank
+---+-----+-------------------------------------------------------+--------------
| 0 | 1   | 0                                                     | 0
| 1 | 2   | 0, 1                                                  | 0
| 2 | 4   | 0, 2, 1, 3                                            | 2
| 3 | 8   | 0, 4, 2, 6, 1, 5, 3, 7                                | 2354
| 4 | 16  | 0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15  | 633303178034
```

### LINK ###
Wikipedia, <a href="https://en.m.wikipedia.org/wiki/Bit-reversal_permutation">Bit-reversal_permutation</a>.

### PROG ###
```
(Python)
from sympy.combinatorics import Permutation
def a(n):
    p = [0]
    for _ in range(n):
        p = [x << 1 for x in p] + [(x << 1) + 1 for x in p]
    return Permutation(p).rank()
print([a(n) for n in range(0,8)])
```

### XREF ###
Cf. A000079, A100575.

## GCR(0,2) RLL encoding of n. ##

### DATA ###
`25, 27, 18, 19, 889, 891, 882, 883, 601, 603, 594, 595, 633, 635, 626, 627, 28473, 28475, 28466, 28467, 28537, 28539, 28530, 28531, 28249, 28251, 28242, 28243, 28281, 28283, 28274, 28275, 19257, 19259, 19250, 19251, 19321, 19323, 19314, 19315, 19033, 19035, 19026, 19027, 19065, 19067, 19058, 19059, 20281, 20283, 20274, 20275, 20345, 20347, 20338, 20339, 20057, 20059, 20050, 20051, 20089, 20091, 20082`

### COMMENTS ###
GCR(0,2) RLL encoding of n but with leading zeros ignored.

### OFFSET ###
0

### FORMULA ###
```
a(n) = a(n-1) + 2 for n = 1 mod 4.
a(n) = a(n-1) + 1 for n = 3 mod 4.
```

### LINK ###
Wikipedia, <a href="https://en.wikipedia.org/wiki/Run-length_limited">Run-length limited</a>

### PROG ###
```
(Python)
def tobase4(n):
    if n == 0: return [0]
    d = []
    while n > 0:
        d.append(n & 3)
        n >>= 2
    return d[::-1]
def a(n):
    if n & 1:
        if n & 3 == 1: return a(n-1) + 2
        if n & 3 == 3: return a(n-1) + 1
    enc_map = {
         '0': '11001',
         '1': '11011',
         '2': '10010',
         '3': '10011',
        '10': '11101',
        '11': '10101',
        '12': '10110',
        '13': '10111',
        '20': '11010',
        '21': '01001',
        '22': '01010',
        '23': '01011',
        '30': '11110',
        '31': '01101',
        '32': '01110',
        '33': '01111'
    }
    return int("".join(enc_map[str(b)] for b in tobase4(n)),2)
print([a(n) for n in range(0, 63)])
```

### KEYWORD ###
base, easy



## Determinant for a Matrix M where M[i,j] = i*j for n >= i > j >= 1 and 1 for 1 <= i < j <= n. ##

### DATA ###
`1, 2, 15, 220, 5225, 181830, 8697535, 546702200, 43667838225, 4318264002250, 517759853869775, 73992590025753300, 12424589075157741625, 2421839132034593636750, 542318977066317932229375, 138255184553439984856342000, 39808852202356125639572974625, 12855917564172654691838566511250`

### OFFSET ###
1

### COMMENTS ###
Conversely the determinant for a Matrix M where M[i,j] = i*j for n >= i < j >= 1 and 1 for 1 <= i > j <= n is A130031.

### PROG ###
```
(Python)
from sympy import Matrix
def a(n):
  M=[]
  for i in range(1,n+1):
    row = []
    for j in range(1, i + 1):
      row.append(i*j)
    for j in range(i+1, n+1):
      row.append(1)
    M.append(row)
  return Matrix(M).det()
print([a(n) for n in range(1,19)])
```

### XREF ###
Cf. A130031.



## a(n) is the binary representation of all partitions of n concatenated together and then converted back to an integer. ##

### DATA ### 
`1, 14, 247, 257724, 1065025357, 35885265650137438, 38609324441197878878632815, 2784857543866383669141335397168626591038200, 3289398245348065727050918124067161877654368252646454686822432025, 34167438679741495089595242316096683785231545331043085651352870101508863522845099586771564396397882`

### OFFSET ### 
1

### PROG ###
```
(Python)
def partitions(n):
    s = [(n, n, [])]
    while s:
        r, m, c = s.pop()
        if r == 0:
            yield c
            continue
        for i in range(min(r, m), 0, -1):
            s.append((r - i, i, c + [i]))
def a(n):
    tmp = ""
    for p in partitions(n):
        tmp += "".join([bin(x)[2:] for x in p])
    return int(tmp, 2)
print([a(n) for n in range(1,11)])
```

### KEYWORD ###
base



## a(0) = 0 and a(n) is binomial(n, a(n - 1)) + 1 ##

### DATA ###
`0, 2, 2, 4, 2, 11, 1, 8, 2, 37, 1, 12, 2, 79, 1, 16, 2, 137, 1, 20, 2, 211, 1, 24, 2, 301, 1, 28, 2, 407, 1, 32, 2, 529, 1, 36, 2, 667, 1, 40, 2, 821, 1, 44, 2, 991, 1, 48, 2, 1177, 1, 52, 2, 1379, 1, 56, 2, 1597, 1, 60, 2, 1831, 1, 64, 2, 2081, 1, 68, 2, 2347, 1, 72, 2, 2629, 1, 76`


### FORMULA ###
```
a(A016825(n)) = 1.
a(A019442(n)) = 2.
a(A000225(n)) = 2^n.
a(A016813(n)) = A188135(floor(n/4)) for n > 0.
a(A016754(n)) = 1 + 2n + 10n^2 + 16n^3 + 8n^4 if n > 0.
a(n) = (gamma(n+1) / gamma(n-a(n-1)+1) * gamma(a(n-1)+1)) + 1.
```

### OFFSET ###
0

### PROG ###
```
(Python)
from sympy import binomial
def a(n):
  if n == 0: return 0
  return binomial(n, a(n - 1)) + 1
print([a(n) for n in range(0,76)])
```

### XREF ### 
Cf. A000142, A000225, A016825, A019442, A188135, A016813.



## a(n) is the product of antidivisors of the totient of n. ##

### DATA ###
`1, 1, 3, 1, 4, 3, 4, 3, 84, 3, 40, 4, 15, 15, 33, 4, 1680, 15, 40, 84, 8100, 15, 312, 40, 1680, 40, 25080, 15, 960, 33, 312, 33, 112, 40, 192, 1680, 112, 33, 11664, 40, 114240, 312, 112, 8100, 33852, 33, 114240, 312, 257985, 112, 9261000, 1680, 11664, 112, 192, 25080, 6296940, 33, 10053120, 960, 192, 257985, 3040, 312, 280896, 257985, 696, 112, 315840, 112, 15924480, 192, 11664`

### OFFSET ###
3

### FORMULA ###
```
a(n) = Product_{d in antidivisors(totient(n))} d.
a(n) = A091507(A000010(n)).
```

### CODE ###
```
(Python)
from sympy.ntheory.factor_ import antidivisors
from sympy import prod, totient
a = lambda n: prod(antidivisors(totient(n)))
print([a(n) for n in range(3, 76)])
```

### KEYWORD ###
easy

### XREF ###
Cf. A000010, A091507.



## Product of the totients of the antidivisors of n. ##
 
### DATA ###
`1, 1, 1, 2, 2, 2, 8, 8, 2, 24, 12, 16, 48, 24, 8, 20, 480, 192, 24, 96, 12, 768, 384, 48, 768, 64, 480, 5760, 36, 64, 864, 41472, 960, 88, 1056, 32, 1280, 153600, 1440, 1728, 216, 6144, 3584, 224, 27648, 8640, 4320, 1152, 14400, 38400, 32, 442368, 110592, 96, 2880, 576, 3168, 608256, 331776, 491520, 800, 12800, 69120, 84, 4032, 17280, 17915904, 663552, 44, 17664, 11040, 1720320, 677376, 12096, 1280`

### OFFSET ###
1

### FORMULA ###
`a(n) = Product_{d in antidivisors(n)} totient(d).`

### PROG ###
```
(Python)
from sympy.ntheory.factor_ import antidivisors
from sympy import prod, totient
a = lambda n: prod(totient(d) for d in antidivisors(n))
print([a(n) for n in range(1, 76)])
```

### XREF ###
Cf. A000010, A091507.



## a(n) is the cumulative sum of the multiplicative cost of merge in the optimal file merge pattern like algorithm applied to the list comprising 1 to n. ##

### DATA ###
`0, 2, 8, 32, 148, 784, 5244, 41160, 365196, 3634770, 39939046, 479102742, 6227559458, 87181950308, 1307698387400, 20922911172308, 355687947758520, 6402375473783654, 121645105512067786, 2432902028391283234, 51090942270685982446, 1124000728416448387006, 25852016743979951090832`

### COMMENTS ###
In the original version of the optimal file merge pattern the cost is calculated with a sum.

### OFFSET ###
1

### LINK ###
geeksforgeeks.org, <a href="https://www.geeksforgeeks.org/optimal-file-merge-patterns/">Optimal merge patterns</a>

### EXAMPLE ###
```
For n = 5:
Starting with f=[1,2,3,4,5]:
len(f) | f               | t   | c
      5 | [1, 2, 3, 4, 5] | 2   | 2
      4 | [2, 3, 4, 5]    | 6   | 8
      3 | [4, 5, 6]       | 20  | 28
      2 | [6, 20]         | 120 |148
 a(n) = 2+6+20+120 = 148
```

### PROG ###
```
(Python)
import heapq
def Omp(f):
    c = 0
    heapq.heapify(f)
    while len(f) > 1:
        a = heapq.heappop(f)
        b = heapq.heappop(f)
        m = a * b
        c += m
        heapq.heappush(f, m)
    return c
a = lambda n: Omp(list(range(1, n+1)))
print([a(n) for n in range(1, 24)])
```



## a(n) is the multiplicative cost of merge in the optimal merge pattern like algorithm applied to the list comprising 1 to n. ##

### DATA ###
`1, 2, 12, 288, 28800, 6220800, 6096384000, 14046068736000, 63712967786496000, 573416710078464000000, 15264352822288711680000000, 949564860368936176189440000000, 116826863900910955628939182080000000, 28851562308968969602122820406476800000000, 12853371008645675957745716491085414400000000000`

### COMMENTS ###
```
In the original optimal file merge pattern algorithm: the counter variable c accumulates a sum of each value of t, while in this algorithm is c*= t.
Conversely when c accmuluates a sum of eatch t the resulting sequence is A328950.
All the t numbers are congruent to 0 or 1 (mod 3) (A032766).
a(n) is divisible by n!.
```

### OFFSET ###
1

### LINK ###
geeksforgeeks.org, <a href="https://www.geeksforgeeks.org/optimal-file-merge-patterns/">Optimal merge patterns</a>

### EXAMPLE ###
```
 For n = 5:
 Starting with f=[1,2,3,4,5]:
 len(f) | f               | m   | c
      5 | [1, 2, 3, 4, 5] | 2   | 2
      4 | [2, 3, 4, 5]    | 6   | 12
      3 | [4, 5, 6]       | 20  | 240
      2 | [6, 20]         | 120 | 28800
 a(n) = 2*6*20*120 = 28800
```

### PROG ###
```
(Python)
def Omp(f):
    c = 1
    heapq.heapify(f) 
    while len(f) > 1:
        a = heapq.heappop(f)  
        b = heapq.heappop(f)  
        m = a * b            
        c *= m                
        heapq.heappush(f, m) 
    return c
a = lambda n: Omp(list(range(1, n+1)))
print([a(n) for n in range(1, 16)])
```

### XREF ###
Cf. A000120.


## How to cite

```
@misc{clavijo2025myintegersequences,
  author       = {Darío Clavijo},
  title        = {MyIntegerSequences: Personal collection of integer sequences},
  year         = {2025},
  howpublished = {\url{https://github.com/daedalus/MyIntegerSequences}},
  note         = {GitHub repository},
}
```
