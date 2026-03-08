import pytest
import dice9 as d9

from dice9.environment import Environment
from dice9.factor import Register


def test_dist_call_options_do_not_leak_across_calls():
    @d9.dist
    def f():
        return d(2)

    first = f()
    assert first[1] == pytest.approx(0.5)
    assert first[2] == pytest.approx(0.5)

    second = f(_options={"semiring": d9.BigInteger()})
    assert second[1] == 1
    assert second[2] == 1

    third = f()
    assert third[1] == pytest.approx(0.5)
    assert third[2] == pytest.approx(0.5)


def test_environment_split_missing_condition_is_explicit_error():
    env = Environment(d9.Real64(), [])

    with pytest.raises(ValueError, match="not found"):
        env.split(Register.new())
