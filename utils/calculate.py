a = int(input('years'))
b = float(input('rate'))
c = int(input('model'))
if c == 1:
    print(3000 * (b ** a))
else:
    d = 3000
    for i in range(a):
        d *= b
        b = b + 0.005
        if b > 15:
            b = 15
    print(d)

