import io
p = 'modules/mod-101-tokenization-and-text-foundations/exercise-03-unicode-segmentation-and-normalisation-audit/SOLUTION.md'
with io.open(p, 'r', encoding='utf-8') as f:
    s = f.read()

NBSP = chr(0xA0)
BOM = chr(0xFEFF)

old99 = "Python's `str.split()` will handle `" + NBSP + "` as whitespace"
new99 = "Python's `str.split()` will handle U+00A0 (NBSP) as whitespace"
assert old99 in s, 'line 99 match failed'
s = s.replace(old99, new99)

old140 = "often start with `" + BOM + "`"
new140 = "often start with U+FEFF (BOM)"
assert old140 in s, 'line 140 match failed'
s = s.replace(old140, new140)

with io.open(p, 'w', encoding='utf-8') as f:
    f.write(s)
print('ok')
