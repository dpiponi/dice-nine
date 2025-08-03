# https://rpg.stackexchange.com/questions/204051/i-need-a-standard-array-for-a-dd-like-homebrew-game-but-anydice-chokes-how
# Cf. https://highdiceroller.github.io/icepool/apps/icecup.html?p=eq&v=2.1.0&c=GYJw9gtgBAlgxgUwA5jAG1hFIAuUAiMCAUMQHYCuEA+gM5xggK1QC8UAjAEyl7uEIAFAG0uAGgDMksQBYxAVjEA2ALoBKYgEMARjDQwcATzZQJUAAJQcxFOhM69BwwDpbaQZRr1GzDcWCMsLBkUCCaZADmQp50DEy0agBcxFCpUGAUOEiZgm7CMOrEQA

def append(a, b):
  return concat([a, [b]], -1)

def my_sort(s, index):
  x = []
  for i in s:
    x = append(x, [i])

def my_top_k(s, index):
  x = []
  l = 0
  for i in s:
    x = append(x, i)
    l += 1
    x = sort(x)
    if l > index + 1:
      x = x[1:]
  return x[0]

def my_bot_k(s, index):
  x = []
  l = 0
  for i in s:
    x = append(x, i)
    l += 1
    x = sort(x)
    if l > index + 1:
      x = x[:-1]
  return x[-1]

def kth(s, index, n):
  if 2 * index < n:
    return my_bot_k(s, index)
  else:
    return my_top_k(s, n - 1 - index)

def f(rolls, index):
  return kth(rolls @ sum(3 @ d[2:4:1, 3:5:1, 5:7:1]), index, rolls)

plt.grid(True)
rolls = 12
for i in range(rolls):
  print(i)
  pmf = d9.run(f, rolls, i)
  plt.plot(pmf.keys(), pmf.values(), label=f"{rolls - i}th highest")
plt.legend()
plt.show()