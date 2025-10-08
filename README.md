INTRO
-----
`dice-nine` is a probabilistic DSL embedded in Python that allows computation (to machine precision) of
probabilities, typically from gaming scenarios, without using Monte-Carlo sampling techniques.

A very simple example is

```
def f():
    return d(6)
```

When called with `d9.run(f)` it returns a `dict` showing that each outcome has a probability of 1/6.

It can solve problems in a single line that are challenging with other libraries. For example this
code allows you to roll d10 `dice` times and summarise all of the probabilities of rolling combinations
like one pair, two pairs, a triple, a triple with a pair and so on.

```
def f(dice):
  return bincount(lazy_bincount(dice @ d(10), 11), 6)[2:]
```

You can simulate entire games within it. Here's a complete D&D (1e) fight:

```
def f():
    hp1 = sum(36 @ d(8))
    hp2 = sum(18 @ d(8))

    for i in range(14):
        print("round", i)
        if hp1 > 0 and d(20) > 1:
            hp2 = sum((-x for x in 5 @ d(4)), hp2)
            hp2 = max(0, hp2)

        if hp2 > 0: 
            if d(20) > 1:
                hp1 -= d(6)
            if d(20) > 1:
                hp1 -= d(6)
            if d(20) > 1:
                hp1 = sum((-x for x in 5 @ d(8)), hp1)
            hp1 = max(hp1, 0)

    win1 = hp2 == 0
    win2 = hp1 == 0

    return win1, win2
```

It is reasonably performant. Here'an "eliminative" dice pool example that runs in a few seconds
with 12 attack and 12 danger dice event though there are 6^24 = 4,738,381,338,321,616,896 ways
you can roll 24 dice.

```
def f(a, d):
  actions = lazy_bincount(a @ d(6), 7)

  for i in d @ d(6):
    actions[i] = max(actions[i] - 1, 0)

  if reduce_all(actions == 0):
    result = 0
  else:
    result = last(actions > 0)
  boons = max(actions[6] - 1, 0)
  return (result, boons)
```

Philosophy
----------
`dice-nine` is written entirely in Python and uses brute-force rather than domain specific
knowledge about probability theory. There are some small tricks to give it performance:

* If there are multiple ways to get to the same state it will combine them.
  This works best if your state doesn't contain non-essential info about the
  past or future. So with that in mind it:
  - quickly forgets the past
    (by analyzing the code to find where it can safely delete stuff)
  - delays thinking about the future
    (by using generators to compute stuff lazily)

The combination of these can give some surprisingly good results.

It can also target computation on GPUs via TensorFlow and PyTorch. TensorFlow generally slows
it down. PyTorch often accelerates it a modest amount. Nonetheless, TensorFlow did inform the
design more than the other libraries because I started thinking about the idea of writing a DSL
like this while working at [Google](https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/experimental/marginalize/marginalizable_test.py).
Compare also with [`tf.vectorized_map`](https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/python/ops/parallel_for/control_flow_ops.py#L452-L582).

By using Python we get to use libraries like `matplotlib` to present results attractively.

I want to make my goals modest. This isn't intended to be a fully general purpose programming language.

The Reality
-----------
This is just a toy spare time project and the code will fail in all kinds of ways with various
inputs. I am still working on catching at least the bad things that could result in incorrect results.
I've been using the [anydice](https://rpg.stackexchange.com/questions/tagged/anydice) tag on the
RPG stackexchange as a source of text cases and in every
case either dice-nine agrees with the results there or I disagree with how to interpret the question.
