import pandas as pd
import numpy as np
import streamlit as st
st.set_page_config(layout="wide")

#seminar.notes.dutc.io

#Notes
# “Rock·Paper·Scissors”
print("Let's take a look!")
# Problem
# Model a game of “rock·paper·scissors” between two players.

# Solution
# What is the simplest modeling?
player = 'rock'
challenger = 'scissors'

if player == 'rock' and challenger == 'rock':
    print("It's a tie!")
elif player == 'rock' and challenger == 'paper':
    print('Player loses!')
elif player == 'rock' and challenger == 'scissors':
    print('Player wins!')
elif player == 'paper' and challenger == 'rock':
    print('Player wins!')
elif player == 'paper' and challenger == 'paper':
    print("It's a tie!")
elif player == 'paper' and challenger == 'scissors':
    print('Player loses!')
elif player == 'scissors' and challenger == 'rock':
    print('Player loses!')
elif player == 'scissors' and challenger == 'paper':
    print('Player wins!')
elif player == 'scissors' and challenger == 'scissors':
    print("It's a tie!")
# How do I turn this into something useful?
from random import Random

rnd = Random()

player = rnd.choice('rock paper scissors'.split())
challenger = rnd.choice('rock paper scissors'.split())

print(f'{player     = }')
print(f'{challenger = }')
if player == 'rock' and challenger == 'rock':
    print("It's a tie!")
elif player == 'rock' and challenger == 'paper':
    print('Player loses!')
elif player == 'rock' and challenger == 'scissors':
    print('Player wins!')
elif player == 'paper' and challenger == 'rock':
    print('Player wins!')
elif player == 'paper' and challenger == 'paper':
    print("It's a tie!")
elif player == 'paper' and challenger == 'scissors':
    print('Player loses!')
elif player == 'scissors' and challenger == 'rock':
    print('Player loses!')
elif player == 'scissors' and challenger == 'paper':
    print('Player wins!')
elif player == 'scissors' and challenger == 'scissors':
    print("It's a tie!")
# How do I turn this into something reüsable or testable?
from random import Random

def game(player, challenger):
    print(f'{player     = }')
    print(f'{challenger = }')
    if player == 'rock' and challenger == 'rock':
        print("It's a tie!")
    elif player == 'rock' and challenger == 'paper':
        print('Challenger wins!')
    elif player == 'rock' and challenger == 'scissors':
        print('Player wins!')
    elif player == 'paper' and challenger == 'rock':
        print('Player wins!')
    elif player == 'paper' and challenger == 'paper':
        print("It's a tie!")
    elif player == 'paper' and challenger == 'scissors':
        print('Player loses!')
    elif player == 'scissors' and challenger == 'rock':
        print('Player loses!')
    elif player == 'scissors' and challenger == 'paper':
        print('Player wins!')
    elif player == 'scissors' and challenger == 'scissors':
        print("It's a tie!")

if __name__ == '__main__':
    rnd = Random()
    player = rnd.choice('rock paper scissors'.split())
    challenger = rnd.choice('rock paper scissors'.split())
    game(player, challenger)
# Should I print inside a function call?
from random import Random

def game(player, challenger):
    if player == 'rock' and challenger == 'rock':
        return "It's a tie!"
    elif player == 'rock' and challenger == 'paper':
        return 'Player loses!'
    elif player == 'rock' and challenger == 'scissors':
        return 'Player wins!'
    elif player == 'paper' and challenger == 'rock':
        return 'Player wins!'
    elif player == 'paper' and challenger == 'paper':
        return "It's a tie!"
    elif player == 'paper' and challenger == 'scissors':
        return 'Player loses!'
    elif player == 'scissors' and challenger == 'rock':
        return 'Player loses!'
    elif player == 'scissors' and challenger == 'paper':
        return 'Player wins!'
    elif player == 'scissors' and challenger == 'scissors':
        return "It's a tie!"

if __name__ == '__main__':
    rnd = Random(0)
    player = rnd.choice('rock paper scissors'.split())
    challenger = rnd.choice('rock paper scissors'.split())
    print(f'{player     = }')
    print(f'{challenger = }')
    print(f'{game(player, challenger) = }')
# Should I use simpler, guessable input and output values?
# What if I make a mistake?

from random import Random

def game(player, challenger):
    if player == 'rock' and challenger == 'rock':
        return 'tie'
    elif player == 'rock' and challenger == 'paper':
        return 'loss'
    elif player == 'rock' and challenger == 'scissors':
        return 'win'
    elif player == 'paper' and challenger == 'rock':
        return 'win'
    elif player == 'paper' and challenger == 'paper':
        return 'tie'
    elif player == 'paper' and challenger == 'scissors':
        return 'loss'
    elif player == 'scissors' and challenger == 'rock':
        return 'loss'
    elif player == 'scissors' and challenger == 'paper':
        return 'win'
    elif player == 'scissors' and challenger == 'scissors':
        return 'tie'

if __name__ == '__main__':
    rnd = Random()
    # player = rnd.choice('rock paper scissors'.split())
    player = 'Rock'
    challenger = rnd.choice('rock paper scissors'.split())
    print(f'Player plays {player}.')
    print(f'Challenger plays {challenger}.')
    if game(player, challenger) == 'wins':
        print(f'Player wins!!')
    elif game(player, challenger) == 'tie':
        print(f"It's a tie!")
    else:
    # elif game(player, challenger) == 'lose':
        print(f'Player loses!')

# How can I check for mistakes?
# python -m pip install mypy pyre
from random import Random
from typing import Literal, get_args

Shape = Literal['rock', 'paper', 'scissors']
Result = Literal['win', 'loss', 'tie']
def game(player : Shape, challenger : Shape) -> Result:
    if player == 'rock' and challenger == 'rock':
        return 'tie'
    elif player == 'rock' and challenger == 'paper':
        return 'loss'
    elif player == 'rock' and challenger == 'scissors':
        return 'win'
    elif player == 'paper' and challenger == 'rock':
        return 'win'
    elif player == 'paper' and challenger == 'paper':
        return 'tie'
    elif player == 'paper' and challenger == 'scissors':
        return 'loss'
    elif player == 'scissors' and challenger == 'rock':
        return 'loss'
    elif player == 'scissors' and challenger == 'paper':
        return 'win'
    elif player == 'scissors' and challenger == 'scissors':
        return 'tie'
    # return None

if __name__ == '__main__':
    rnd = Random(0)
    player = rnd.choice(get_args(Shape))
    challenger = rnd.choice(get_args(Shape))
    print(f'Player plays {player}.')
    print(f'Challenger plays {challenger}.')
    result : Result = game(player, challenger)
    if result == 'win':
        print(f'Player wins!')
    elif result == 'tie':
        print(f"It's a tie!")
    elif result == 'lose':
        print(f'Player loses!')
# How I use mypy properly?
from random import Random
from typing import Literal, get_args

Shape = Literal['rock', 'paper', 'scissors']
Result = Literal['win', 'loss', 'tie']
def game(player : Shape, challenger : Shape) -> Result:
    if player == 'rock' and challenger == 'rock':
        return 'tie'
    elif player == 'rock' and challenger == 'paper':
        return 'loss'
    elif player == 'rock' and challenger == 'scissors':
        return 'win'
    elif player == 'paper' and challenger == 'rock':
        return 'win'
    elif player == 'paper' and challenger == 'paper':
        return 'tie'
    elif player == 'paper' and challenger == 'scissors':
        return 'loss'
    elif player == 'scissors' and challenger == 'rock':
        return 'loss'
    elif player == 'scissors' and challenger == 'paper':
        return 'win'
    elif player == 'scissors' and challenger == 'scissors':
        return 'tie'
    return 'tie'

if __name__ == '__main__':
    rnd = Random(0)

    player = rnd.choice(get_args(Shape))
    challenger = rnd.choice(get_args(Shape))
    print(f'Player plays {player}.')
    print(f'Challenger plays {challenger}.')

    result = game(player, challenger)
    if result == 'win':
        print(f'Player wins!')
    elif result == 'tie':
        print(f"It's a tie!")
    elif result == 'lose':
        print(f'Player loses!')
# Would an enum.Enum work better?
from random import Random
from enum import Enum

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

def game(player : Shape, challenger : Shape) -> Result:
    if player is Shape.Rock and challenger is Shape.Rock:
        return Result.Tie
    elif player is Shape.Rock and challenger is Shape.Paper:
        return Result.Loss
    elif player is Shape.Rock and challenger is Shape.Scissors:
        return Result.Win
    elif player is Shape.Paper and challenger is Shape.Rock:
        return Result.Win
    elif player is Shape.Paper and challenger is Shape.Paper:
        return Result.Tie
    elif player is Shape.Paper and challenger is Shape.Scissors:
        return Result.Loss
    elif player is Shape.Scissors and challenger is Shape.Rock:
        return Result.Loss
    elif player is Shape.Scissors and challenger is Shape.Paper:
        return Result.Win
    elif player is Shape.Scissors and challenger is Shape.Scissors:
        return Result.Tie

if __name__ == '__main__':
    rnd = Random()

    player = rnd.choice([*Shape])
    challenger = rnd.choice([*Shape])
    print(f'Player plays {player.name.lower()}.')
    print(f'Challenger plays {challenger.name.lower()}.')

    result = game(player, challenger)
    if result is Result.Win:
        print(f'Player wins!')
    elif result is Result.Tie:
        print(f"It's a tie!")
    elif result is Result.Loss:
        print(f'Player loses!')
# How do I make this code less ‘noisy’?
from random import Random
from enum import Enum

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

def game(player : Shape, challenger : Shape) -> Result | None:
    if player is Shape.Rock     and challenger is Shape.Paper: return Result.Loss
    if player is Shape.Rock     and challenger is Shape.Scissors: return Result.Win
    if player is Shape.Paper    and challenger is Shape.Rock: return Result.Win
    if player is Shape.Paper    and challenger is Shape.Scissors: return Result.Loss
    if player is Shape.Scissors and challenger is Shape.Rock: return Result.Loss
    if player is Shape.Scissors and challenger is Shape.Paper: return Result.Win
    if player is challenger:
        return Result.Tie

if __name__ == '__main__':
    rnd = Random(0)

    player = rnd.choice([*Shape])
    challenger = rnd.choice([*Shape])
    print(f'Player plays {player.name.lower()}.')
    print(f'Challenger plays {challenger.name.lower()}.')

    result = game(player, challenger)
    if result is Result.Win:
        print(f'Player wins!')
    elif result is Result.Tie:
        print(f"It's a tie!")
    elif result is Result.Loss:
        print(f'Player loses!')
# Can I use match/case to make patterns more visible?
from random import Random
from enum import Enum

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

def game(player : Shape, challenger : Shape) -> Result | None:
    match player, challenger:
        case Shape.Rock, Shape.Rock:     return Result.Tie
        case Shape.Rock, Shape.Paper:    return Result.Loss
        case Shape.Rock, Shape.Scissors: return Result.Win

        case Shape.Paper, Shape.Rock:     return Result.Tie
        case Shape.Paper, Shape.Paper:    return Result.Win
        case Shape.Paper, Shape.Scissors: return Result.Loss

        case Shape.Scissors, Shape.Rock:     return Result.Loss
        case Shape.Scissors, Shape.Paper:    return Result.Win
        case Shape.Scissors, Shape.Scissors: return Result.Tie

if __name__ == '__main__':
    rnd = Random(0)

    player = rnd.choice([*Shape])
    challenger = rnd.choice([*Shape])
    print(f'Player plays {player.name.lower()}.')
    print(f'Challenger plays {challenger.name.lower()}.')

    result = game(player, challenger)
    if result is Result.Win:
        print(f'Player wins!')
    elif result is Result.Tie:
        print(f"It's a tie!")
    elif result is Result.Loss:
        print(f'Player loses!')
# Is this any better than using a dict?
from random import Random
from enum import Enum

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

rules = {
    (Shape.Rock, Shape.Rock):     Result.Tie,
    (Shape.Rock, Shape.Paper):    Result.Loss,
    (Shape.Rock, Shape.Scissors): Result.Win,

    (Shape.Paper, Shape.Rock):     Result.Win,
    (Shape.Paper, Shape.Paper):    Result.Tie,
    (Shape.Paper, Shape.Scissors): Result.Loss,

    (Shape.Scissors, Shape.Rock):     Result.Loss,
    (Shape.Scissors, Shape.Paper):    Result.Win,
    (Shape.Scissors, Shape.Scissors): Result.Tie,
}
assert len({(pl, res) for (pl, cl), res in rules.items()}) == 9
def game(player : Shape, challenger : Shape) -> Result | None:
    return rules.get((player, challenger))

if __name__ == '__main__':
    rnd = Random(0)

    player = rnd.choice([*Shape])
    challenger = rnd.choice([*Shape])
    print(f'Player plays {player.name.lower()}.')
    print(f'Challenger plays {challenger.name.lower()}.')

    result = game(player, challenger)
    if result is Result.Win:
        print(f'Player wins!')
    elif result is Result.Tie:
        print(f"It's a tie!")
    elif result is Result.Loss:
        print(f'Player loses!')
# Can a dict and a data-driven approach encode fundamental ‘structure’ of my game?
from random import Random
from enum import Enum

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

def game(player : Shape, challenger : Shape) -> Result | None:
    rules = {
        (Shape.Rock, Shape.Scissors):  Result.Win,
        (Shape.Paper, Shape.Rock):     Result.Win,
        (Shape.Scissors, Shape.Paper): Result.Win,
    }
    rrules = {(ch, pl): Result.Loss for (pl, ch), res in rules.items()}
    return rules.get(
        (player, challenger),
        rrules.get(
            (player, challenger),
            Result.Tie,
        )
    )

if __name__ == '__main__':
    rnd = Random(0)

    player = rnd.choice([*Shape])
    challenger = rnd.choice([*Shape])
    print(f'Player plays {player.name.lower()}.')
    print(f'Challenger plays {challenger.name.lower()}.')

    result = game(player, challenger)
    if result is Result.Win:
        print(f'Player wins!')
    elif result is Result.Tie:
        print(f"It's a tie!")
    elif result is Result.Loss:
        print(f'Player loses!')
# How can I improve the dict modeling?
# ```python -O from random import Random from enum import Enum

Shape = Enum('Shape', 'Stone Paper Scissors') 
Result = Enum('Result', 'Win Loss Tie')

beats = { Shape.Stone: Shape.Scissors, Shape.Paper: Shape.Rock, Shape.Scissors: Shape.Paper, } 
beaten_by = {v: k for k, v in beats.items()}

# assert … if debug: pass

# def game(player : Shape, challenger : Shape) -> Result | None: if beats[player] is challenger: return Result.Win if beaten_by[player] is challenger: return Result.Loss return Result.Tie

# if name == ‘main’: rnd = Random(0)

player = rnd.choice([*Shape])
challenger = rnd.choice([*Shape])
print(f'Player plays {player.name.lower()}.')
print(f'Challenger plays {challenger.name.lower()}.')

result = game(player, challenger)
if result is Result.Win:
    print(f'Player wins!')
elif result is Result.Tie:
    print(f"It's a tie!")
elif result is Result.Loss:
    print(f'Player loses!')
# How can I introduce correctness checks?
from random import Random
from enum import Enum

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

beats = {
    Shape.Rock: Shape.Scissors,
    Shape.Paper: Shape.Paper,
    Shape.Scissors: Shape.Paper,
}
beaten_by = {v: k for k, v in beats.items()}

assert {*beats} == {*Shape}, 'Missing shape'
assert {*beats.values()} == {*Shape}, 'Missing shape'

def game(player : Shape, challenger : Shape) -> Result | None:
    if beats[player] is challenger:
        return Result.Win
    if beaten_by[player] is challenger:
        return Result.Loss
    return Result.Tie

if __name__ == '__main__':
    rnd = Random(0)

    player = rnd.choice([*Shape])
    challenger = rnd.choice([*Shape])
    print(f'Player plays {player.name.lower()}.')
    print(f'Challenger plays {challenger.name.lower()}.')

    result = game(player, challenger)
    if result is Result.Win:
        print(f'Player wins!')
    elif result is Result.Tie:
        print(f"It's a tie!")
    elif result is Result.Loss:
        print(f'Player loses!')
# How can I handle variations such as Rock paper scissors: Additional weapons?
# One popular five-weapon expansion is “rock paper scissors Spock lizard”, invented by Sam Kass and Karen Bryla,[59] which adds “Spock” and “lizard” to the standard three choices. “Spock” is signified with the Star Trek Vulcan salute, while “lizard” is shown by forming the hand into a sock-puppet-like mouth. Spock smashes scissors and vaporizes rock; he is poisoned by lizard and disproved by paper. Lizard poisons Spock and eats paper; it is crushed by rock and decapitated by scissors.

from random import Random
from enum import Enum
from itertools import chain
from collections import defaultdict, Counter

Shape = Enum('Shape', 'Rock Paper Scissors Spock Lizard')
Result = Enum('Result', 'Win Loss Tie')

beats : dict[Shape, set[Shape]] = {
    Shape.Rock:     {Shape.Scissors},
    Shape.Paper:    {Shape.Rock},
    Shape.Scissors: {Shape.Paper},
    # Shape.Spock:    {Shape.Scissors, Shape.Rock},
    # Shape.Lizard:   {Shape.Paper, Shape.Spock},
}
beaten_by = defaultdict(set)
for k, vs in beats.items():
    for v in vs:
        beaten_by[v].add(k)

assert {*beats} == {*Shape}, 'Missing shape'
assert Counter(chain.from_iterable(beats.values())) == {s: 2 for s in Shape}, 'Unbalanced shape'

def game(player : Shape, challenger : Shape) -> Result | None:
    if challenger in beats[player]:
        return Result.Win
    if challenger in beaten_by[player]:
        return Result.Loss
    return Result.Tie

if __name__ == '__main__':
    rnd = Random()

    player = rnd.choice([*Shape])
    challenger = rnd.choice([*Shape])
    print(f'Player plays {player.name.lower()}.')
    print(f'Challenger plays {challenger.name.lower()}.')

    result = game(player, challenger)
    if result is Result.Win:
        print(f'Player wins!')
    elif result is Result.Tie:
        print(f"It's a tie!")
    elif result is Result.Loss:
        print(f'Player loses!')
# What do I do with this code?
from random import Random
from enum import Enum
from collections import Counter

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

beats = {
    Shape.Rock: Shape.Scissors,
    Shape.Paper: Shape.Rock,
    Shape.Scissors: Shape.Paper,
}
beaten_by = {v: k for k, v in beats.items()}

assert {*beats} == {*Shape}, 'Missing shape'
assert {*beats.values()} == {*Shape}, 'Missing shape'

def game(player : Shape, challenger : Shape) -> Result | None:
    if beats[player] is challenger:
        return Result.Win
    if beaten_by[player] is challenger:
        return Result.Loss
    return Result.Tie

if __name__ == '__main__':
    rnd = Random(0)

    results = []
    for _ in range(10_000):
        player = rnd.choice([*Shape])
        challenger = rnd.choice([*Shape])
        results.append(game(player, challenger))

    for res, cnt in Counter(results).items():
        print(f'{res.name:<5} {cnt / len(results) * 100:.0f}%')
# How can I compare ‘strategies’?
from random import Random
from enum import Enum
from collections import Counter

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

beats = {
    Shape.Rock: Shape.Scissors,
    Shape.Paper: Shape.Rock,
    Shape.Scissors: Shape.Paper,
}
beaten_by = {v: k for k, v in beats.items()}

assert {*beats} == {*Shape}, 'Missing shape'
assert {*beats.values()} == {*Shape}, 'Missing shape'

def game(player : Shape, challenger : Shape) -> Result | None:
    if beats[player] is challenger:
        return Result.Win
    if beaten_by[player] is challenger:
        return Result.Loss
    return Result.Tie

def random_play():
    return rnd.choice([*Shape])

def always_rock():
    return Shape.Rock

if __name__ == '__main__':
    rnd = Random(0)

    results = []
    for _ in range(10_000):
        player = always_rock()
        challenger = always_rock()
        results.append(game(player, challenger))

    for res, cnt in Counter(results).items():
        print(f'{res.name:<5} {cnt / len(results) * 100:.0f}%')
# How do I structure my scenarios?
from random import Random
from enum import Enum
from collections import Counter

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

beats = {
    Shape.Rock: Shape.Scissors,
    Shape.Paper: Shape.Rock,
    Shape.Scissors: Shape.Paper,
}
beaten_by = {v: k for k, v in beats.items()}

assert {*beats} == {*Shape}, 'Missing shape'
assert {*beats.values()} == {*Shape}, 'Missing shape'

def game(player : Shape, challenger : Shape) -> Result | None:
    if beats[player] is challenger:
        return Result.Win
    if beaten_by[player] is challenger:
        return Result.Loss
    return Result.Tie

def random_play():
    return rnd.choice([*Shape])

def always_rock():
    return Shape.Rock

if __name__ == '__main__':
    rnd = Random(0)

    scenarios = {
        (random_play, random_play),
        (random_play, always_rock),
        (always_rock, always_rock),
    }

    results = {sc: [] for sc in sorted(scenarios, key=lambda pl_ch: (pl_ch[0].__name__, pl_ch[-1].__name__))}

    for (pl, ch), res in results.items():
        for _ in range(10_000):
            player = pl()
            challenger = ch()
            results[pl, ch].append(game(player, challenger))

    for (pl, ch), all_res in results.items():
        print(f'{pl.__name__} vs {ch.__name__}')
        for res, cnt in Counter(all_res).items():
            print(f'{res.name:<5} {cnt / len(all_res) * 100:.0f}%')
# Should I take care to avoid non-determinism?
from random import Random
from enum import Enum
from collections import Counter

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

beats = {
    Shape.Rock: Shape.Scissors,
    Shape.Paper: Shape.Rock,
    Shape.Scissors: Shape.Paper,
}
beaten_by = {v: k for k, v in beats.items()}

assert {*beats} == {*Shape}, 'Missing shape'
assert {*beats.values()} == {*Shape}, 'Missing shape'

def game(player : Shape, challenger : Shape) -> Result | None:
    if beats[player] is challenger:
        return Result.Win
    if beaten_by[player] is challenger:
        return Result.Loss
    return Result.Tie

def random_play():
    return rnd.choice([*Shape])

def zzz_always_rock():
    return Shape.Rock

if __name__ == '__main__':
    rnd = Random(0)

    scenarios = {
        (random_play, random_play),
        (random_play, zzz_always_rock),
        (zzz_always_rock, zzz_always_rock),
    }

    results = {sc: [] for sc in sorted(scenarios, key=lambda x_y: (x_y[0].__name__, x_y[-1].__name__))}

    for (pl, ch), res in results.items():
        for _ in range(10_000):
            player = pl()
            challenger = ch()
            results[pl, ch].append(game(player, challenger))

    for (pl, ch), all_res in results.items():
        print(f'{pl.__name__} vs {ch.__name__}')
        for res, cnt in Counter(all_res).items():
            print(f'{res.name:<5} {cnt / len(all_res) * 100:.0f}%')
# Can this code be improved with some light collections.namedtuple structuring?
from random import Random
from enum import Enum
from collections import Counter, namedtuple
from functools import cached_property

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

beats = {
    Shape.Rock: Shape.Scissors,
    Shape.Paper: Shape.Rock,
    Shape.Scissors: Shape.Paper,
}
beaten_by = {v: k for k, v in beats.items()}

assert {*beats} == {*Shape}, 'Missing shape'
assert {*beats.values()} == {*Shape}, 'Missing shape'

def game(player : Shape, challenger : Shape) -> Result | None:
    if beats[player] is challenger:
        return Result.Win
    if beaten_by[player] is challenger:
        return Result.Loss
    return Result.Tie

def random_play():
    return rnd.choice([*Shape])

def always_rock():
    return Shape.Rock

if __name__ == '__main__':
    class Scenario(namedtuple('ScenarioBase', 'player_strategy challenger_strategy')):
        @cached_property
        def name(self):
            return f'{self.player_strategy.__name__} vs {self.challenger_strategy.__name__}'

    scenarios = {
        Scenario(random_play, always_rock),
        Scenario(always_rock, always_rock),
        Scenario(random_play, random_play),
    }

    results = {sc: [] for sc in scenarios}

    for sc, res in results.items():
        rnd = Random(0)
        for _ in range(10_000):
            player = sc.player_strategy()
            challenger = sc.challenger_strategy()
            results[sc].append(game(player, challenger))

    for sc, all_res in sorted(results.items(), key=lambda sc_: sc_[0].name):
        print(f'{sc.name}')
        for res, cnt in Counter(all_res).items():
            print(f'{res.name:<5} {cnt / len(all_res) * 100:.0f}%')
# Can I determine the scenarios programmatically?
from random import Random
from enum import Enum
from collections import Counter, namedtuple
from functools import cached_property
from itertools import combinations_with_replacement

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

beats = {
    Shape.Rock: Shape.Scissors,
    Shape.Paper: Shape.Rock,
    Shape.Scissors: Shape.Paper,
}
beaten_by = {v: k for k, v in beats.items()}

assert {*beats} == {*Shape}, 'Missing shape'
assert {*beats.values()} == {*Shape}, 'Missing shape'

def game(player : Shape, challenger : Shape) -> Result | None:
    if beats[player] is challenger:
        return Result.Win
    if beaten_by[player] is challenger:
        return Result.Loss
    return Result.Tie

def random_play():
    return rnd.choice([*Shape])

def always_rock():
    return Shape.Rock

if __name__ == '__main__':
    class Scenario(namedtuple('ScenarioBase', 'player_strategy challenger_strategy')):
        @cached_property
        def name(self):
            return f'{self.player_strategy.__name__} vs {self.challenger_strategy.__name__}'

    scenarios = [
        Scenario(*strats)
        for strats in combinations_with_replacement([
            random_play,
            always_rock,
        ], r=2)
    ]
    results = {sc: [] for sc in scenarios}

    for sc, res in results.items():
        rnd = Random(0)
        for _ in range(10_000):
            player = sc.player_strategy()
            challenger = sc.challenger_strategy()
            results[sc].append(game(player, challenger))

    for sc, all_res in sorted(results.items(), key=lambda sc_: sc_[0].name):
        print(f'{sc.name}')
        for res, cnt in Counter(all_res).items():
            print(f'{res.name:<5} {cnt / len(all_res) * 100:.0f}%')
# What if I want to implement a strategy that has some historical knowledge?
from random import Random
from enum import Enum
from collections import Counter, namedtuple
from functools import cached_property
from itertools import combinations_with_replacement

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

beats = {
    Shape.Rock: Shape.Scissors,
    Shape.Paper: Shape.Rock,
    Shape.Scissors: Shape.Paper,
}
beaten_by = {v: k for k, v in beats.items()}

assert {*beats} == {*Shape}, 'Missing shape'
assert {*beats.values()} == {*Shape}, 'Missing shape'

def game(player : Shape, challenger : Shape) -> Result | None:
    if beats[player] is challenger:
        return Result.Win
    if beaten_by[player] is challenger:
        return Result.Loss
    return Result.Tie

def random_play(_):
    return rnd.choice([*Shape])

def always_rock(_):
    return Shape.Rock

def beats_last_play(last_play):
    if last_play is None:
        return random_play(last_play)
    return beaten_by[last_play]

if __name__ == '__main__':
    class Scenario(namedtuple('ScenarioBase', 'player_strategy challenger_strategy')):
        @cached_property
        def name(self):
            return f'{self.player_strategy.__name__} vs {self.challenger_strategy.__name__}'

    scenarios = [
        Scenario(*strats)
        for strats in combinations_with_replacement([
            random_play,
            always_rock,
            beats_last_play,
        ], r=2)
    ]
    results = {sc: [] for sc in scenarios}

    for sc, res in results.items():
        rnd = Random(1)
        last_player, last_challenger = None, None
        for _ in range(10_000):
            player = sc.player_strategy(last_challenger)
            challenger = sc.challenger_strategy(last_player)
            results[sc].append(game(player, challenger))
            last_player, last_challenger = player, challenger

    for sc, all_res in sorted(results.items(), key=lambda sc_: sc_[0].name):
        print(f'{sc.name}')
        for res, cnt in Counter(all_res).items():
            print(f'{res.name:<5} {cnt / len(all_res) * 100:.0f}%')
# What if my strategies are parameterized?
from random import Random
from enum import Enum
from collections import Counter, namedtuple
from functools import cached_property
from itertools import combinations_with_replacement

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

beats = {
    Shape.Rock: Shape.Scissors,
    Shape.Paper: Shape.Rock,
    Shape.Scissors: Shape.Paper,
}
beaten_by = {v: k for k, v in beats.items()}

assert {*beats} == {*Shape}, 'Missing shape'
assert {*beats.values()} == {*Shape}, 'Missing shape'

def game(player : Shape, challenger : Shape) -> Result | None:
    if beats[player] is challenger:
        return Result.Win
    if beaten_by[player] is challenger:
        return Result.Loss
    return Result.Tie

def random_play(_):
    return rnd.choice([*Shape])

def always_shape(shape):
    def strategy(_):
        return shape
    strategy.__name__ = f'always_{shape.name.lower()}'
    return strategy

def beats_last_play(last_play):
    if last_play is None:
        return random_play(last_play)
    return beaten_by[last_play]

if __name__ == '__main__':
    class Scenario(namedtuple('ScenarioBase', 'player_strategy challenger_strategy')):
        @cached_property
        def name(self):
            return f'{self.player_strategy.__name__} vs {self.challenger_strategy.__name__}'

    scenarios = [
        Scenario(*strats)
        for strats in combinations_with_replacement([
            random_play,
            always_shape(Shape.Rock),
            always_shape(Shape.Paper),
            always_shape(Shape.Scissors),
            beats_last_play,
        ], r=2)
    ]
    results = {sc: [] for sc in scenarios}

    for sc, res in results.items():
        rnd = Random(0)
        player, challenger = None, None
        for _ in range(10_000):
            player, challenger = sc.player_strategy(challenger), sc.challenger_strategy(player)
            results[sc].append(game(player, challenger))

    for sc, all_res in sorted(results.items(), key=lambda sc_: sc_[0].name):
        print(f'{sc.name}')
        for res, cnt in Counter(all_res).items():
            print(f'{res.name:<5} {cnt / len(all_res) * 100:.0f}%')
# Should these parameters be statically or dynamically determined?
from random import Random
from enum import Enum
from collections import Counter, namedtuple
from functools import cached_property
from itertools import combinations_with_replacement

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

beats = {
    Shape.Rock: Shape.Scissors,
    Shape.Paper: Shape.Rock,
    Shape.Scissors: Shape.Paper,
}
beaten_by = {v: k for k, v in beats.items()}

assert {*beats} == {*Shape}, 'Missing shape'
assert {*beats.values()} == {*Shape}, 'Missing shape'

def game(player : Shape, challenger : Shape) -> Result | None:
    if beats[player] is challenger:
        return Result.Win
    if beaten_by[player] is challenger:
        return Result.Loss
    return Result.Tie

def random_play(_):
    return rnd.choice([*Shape])

def _always_shape(shape):
    def strategy(_):
        return shape
    return strategy

always_rock = _always_shape(Shape.Rock)
always_scissors = _always_shape(Shape.Scissors)
always_paper = _always_shape(Shape.Paper)

always_rock.__name__ = 'always_rock'
always_paper.__name__ = 'always_paper'
always_scissors.__name__ = 'always_scissors'

def beats_last_play(last_play):
    if last_play is None:
        return random_play(last_play)
    return beaten_by[last_play]

if __name__ == '__main__':
    class Scenario(namedtuple('ScenarioBase', 'player_strategy challenger_strategy')):
        @cached_property
        def name(self):
            return f'{self.player_strategy.__name__} vs {self.challenger_strategy.__name__}'

    scenarios = [
        Scenario(*strats)
        for strats in combinations_with_replacement([
            random_play,
            always_rock,
            always_scissors,
            always_paper,
            beats_last_play,
        ], r=2)
    ]
    results = {sc: [] for sc in scenarios}

    for sc, res in results.items():
        rnd = Random(0)
        player, challenger = None, None
        for _ in range(10_000):
            player, challenger = sc.player_strategy(challenger), sc.challenger_strategy(player)
            results[sc].append(game(player, challenger))

    for sc, all_res in sorted(results.items(), key=lambda sc_: sc_[0].name):
        print(f'{sc.name}')
        for res, cnt in Counter(all_res).items():
            print(f'{res.name:<5} {cnt / len(all_res) * 100:.0f}%')
# What if I want to implement a strategy with other game-wide knowledge?
from random import Random
from enum import Enum
from collections import Counter, namedtuple
from functools import cached_property
from itertools import combinations_with_replacement

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

beats = {
    Shape.Rock: Shape.Scissors,
    Shape.Paper: Shape.Rock,
    Shape.Scissors: Shape.Paper,
}
beaten_by = {v: k for k, v in beats.items()}

assert {*beats} == {*Shape}, 'Missing shape'
assert {*beats.values()} == {*Shape}, 'Missing shape'

def game(player : Shape, challenger : Shape) -> Result | None:
    if beats[player] is challenger:
        return Result.Win
    if beaten_by[player] is challenger:
        return Result.Loss
    return Result.Tie

def random_play(_, __):
    return rnd.choice([*Shape])

def always_shape(shape):
    def strategy(_, __):
        return shape
    return strategy
always_rock = always_shape(Shape.Rock)
always_scissors = always_shape(Shape.Scissors)
always_paper = always_shape(Shape.Paper)
always_rock.__name__ = 'always_rock'
always_paper.__name__ = 'always_paper'
always_scissors.__name__ = 'always_scissors'

def beats_last_play(last_play, *_, **__):
    if last_play is None:
        return random_play(last_play, __)
    return beaten_by[last_play]

def patterned_play(pattern):
    def strategy(_, round_num):
        return pattern[round_num % len(pattern)]
    return strategy

if __name__ == '__main__':
    class Scenario(namedtuple('ScenarioBase', 'player_strategy challenger_strategy')):
        @cached_property
        def name(self):
            return f'{self.player_strategy.__name__} vs {self.challenger_strategy.__name__}'

    scenarios = [
        Scenario(*strats)
        for strats in combinations_with_replacement([
            # random_play,
            # always_rock,
            # always_scissors,
            # always_paper,
            patterned_play([Shape.Rock, Shape.Scissors, Shape.Paper]),
            beats_last_play,
        ], r=2)
    ]
    results = {sc: [] for sc in scenarios}

    for sc, res in results.items():
        rnd = Random(0)
        player, challenger = None, None
        for round_num in range(10_000):
            player, challenger = sc.player_strategy(challenger, round_num), sc.challenger_strategy(player, round_num)
            results[sc].append(game(player, challenger))

    for sc, all_res in sorted(results.items(), key=lambda sc_: sc_[0].name):
        print(f'{sc.name}')
        for res, cnt in Counter(all_res).items():
            print(f'{res.name:<5} {cnt / len(all_res) * 100:.0f}%')
# Can the same be accomplished without changing the signature of all strategies?
from random import Random
from enum import Enum
from collections import Counter, namedtuple
from dataclasses import dataclass
from collections.abc import Iterator
from functools import cached_property
from itertools import combinations_with_replacement, cycle

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

beats = {
    Shape.Rock: Shape.Scissors,
    Shape.Paper: Shape.Rock,
    Shape.Scissors: Shape.Paper,
}
beaten_by = {v: k for k, v in beats.items()}

assert {*beats} == {*Shape}, 'Missing shape'
assert {*beats.values()} == {*Shape}, 'Missing shape'

def game(player : Shape, challenger : Shape) -> Result | None:
    if beats[player] is challenger:
        return Result.Win
    if beaten_by[player] is challenger:
        return Result.Loss
    return Result.Tie

def random_play(_):
    return rnd.choice([*Shape])

def always_shape(shape):
    def strategy(_):
        return shape
    return strategy
always_rock = always_shape(Shape.Rock)
always_scissors = always_shape(Shape.Scissors)
always_paper = always_shape(Shape.Paper)
always_rock.__name__ = 'always_rock'
always_paper.__name__ = 'always_paper'
always_scissors.__name__ = 'always_scissors'

def beats_last_play(last_play):
    if last_play is None:
        return random_play(last_play)
    return beaten_by[last_play]

@dataclass(frozen=True)
class patterned_play:
    pattern : list[Shape]
    _iter   : Iterator = None

    def __post_init__(self):
        object.__setattr__(self, '_iter', cycle(self.pattern))

    def __call__(self, _):
        return next(self._iter)

    def __hash__(self):
        return hash(tuple(self.pattern))

    @cached_property
    def __name__(self):
        return type(self).__name__

if __name__ == '__main__':
    class Scenario(namedtuple('ScenarioBase', 'player_strategy challenger_strategy')):
        @cached_property
        def name(self):
            return f'{self.player_strategy.__name__} vs {self.challenger_strategy.__name__}'

    scenarios = [
        Scenario(*strats)
        for strats in combinations_with_replacement([
            # random_play,
            # always_rock,
            # always_scissors,
            # always_paper,
            # always_paper,
            # always_paper,
            # always_paper,
            # always_paper,
            # always_paper,
            # always_paper,
            patterned_play([Shape.Rock, Shape.Scissors, Shape.Paper]),
            beats_last_play,
        ], r=2)
    ]
    results = {sc: [] for sc in scenarios}

    for sc, res in results.items():
        rnd = Random(0)
        player, challenger = None, None
        for _ in range(10_000):
            player, challenger = sc.player_strategy(challenger), sc.challenger_strategy(player)
            results[sc].append(game(player, challenger))

    for sc, all_res in sorted(results.items(), key=lambda sc_: sc_[0].name):
        print(f'{sc.name}')
        for res, cnt in Counter(all_res).items():
            print(f'{res.name:<5} {cnt / len(all_res) * 100:.0f}%')
# How do I detect newly defined strategies programmatically?
from random import Random
from enum import Enum
from collections import Counter, namedtuple
from dataclasses import dataclass
from collections.abc import Iterator
from functools import cached_property
from itertools import combinations_with_replacement, cycle

Shape = Enum('Shape', 'Rock Paper Scissors')
Result = Enum('Result', 'Win Loss Tie')

beats = {
    Shape.Rock: Shape.Scissors,
    Shape.Paper: Shape.Rock,
    Shape.Scissors: Shape.Paper,
}
beaten_by = {v: k for k, v in beats.items()}

assert {*beats} == {*Shape}, 'Missing shape'
assert {*beats.values()} == {*Shape}, 'Missing shape'

def game(player : Shape, challenger : Shape) -> Result | None:
    if beats[player] is challenger:
        return Result.Win
    if beaten_by[player] is challenger:
        return Result.Loss
    return Result.Tie

class Strategy(namedtuple('Strategy', 'name wrapper func')):
    def __call__(self, *args, **kwargs):
        return self.wrapper(self.func, *args, **kwargs)

ALL_STRATEGIES = []
def strategy(variants):
    def dec(func):
        for name, wrapper in variants.items():
            ALL_STRATEGIES.append(Strategy(name, wrapper, func))
        return func
    return dec

@strategy(
    variants={
        # 'random play': lambda f, *_, **__: f(),
    }
)
def random_play():
    return rnd.choice([*Shape])

@strategy(
    variants={
        'always rock':     lambda f, *_, **__: f(shape=Shape.Rock),
        'always paper':    lambda f, *_, **__: f(shape=Shape.Paper),
        'always scissors': lambda f, *_, **__: f(shape=Shape.Scissors),
    }
)
def always_shape(shape):
    return shape

@strategy(
    variants={
        'beats last play': lambda f, *_, last_play: f(last_play=last_play),
    }
)
def beats_last_play(last_play):
    if last_play is None:
        return random_play()
    return beaten_by[last_play]

@strategy(
    variants={
        'patterned play (R→S→P)':
        lambda f, *_, pattern=cycle([Shape.Rock, Shape.Scissors, Shape.Paper]), **__: f(shape=next(pattern)),
        'patterned play (R→S→S→P)':
        lambda f, *_, pattern=cycle([Shape.Rock, Shape.Scissors, Shape.Scissors, Shape.Paper]), **__: f(shape=next(pattern)),
    }
)
def patterned_play(shape):
    return shape

@strategy(
    variants={}
)
def new_strategy():
    pass

if __name__ == '__main__':
    class Scenario(namedtuple('ScenarioBase', 'player_strategy challenger_strategy')):
        @cached_property
        def name(self):
            return f'{self.player_strategy.name} vs {self.challenger_strategy.name}'

    scenarios = [
        Scenario(*strats)
        for strats in combinations_with_replacement(ALL_STRATEGIES, r=2)
    ]
    results = {sc: [] for sc in scenarios}

    for sc, res in results.items():
        rnd = Random(0)
        player, challenger = None, None
        for _ in range(10_000):
            player, challenger = sc.player_strategy(last_play=challenger), sc.challenger_strategy(last_play=player)
            results[sc].append(game(player, challenger))

    for sc, all_res in sorted(results.items(), key=lambda sc_: sc_[0].name):
        print(f'{sc.name}')
        for res, cnt in Counter(all_res).items():
            print(f'{res.name:<5} {cnt / len(all_res) * 100:.0f}%')