import random

MAX = 1000
MAX_STAGE = 10
print(MAX, MAX_STAGE)

stage = 1
answer = random.randint(1, MAX)

while stage <= MAX_STAGE:
    # print(stage, end='')
    n = int(input())

    if n < 1 or n > MAX:
        continue
    if n == answer:
        print(('正解', stage))
    elif n > answer:
        print('もっと小さい')
    else:
        print('もっと大きい')

    stage += 1

else:
    print('残念')