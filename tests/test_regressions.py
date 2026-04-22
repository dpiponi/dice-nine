# pyright: reportUndefinedVariable=false, reportGeneralTypeIssues=false

import ast
import textwrap

import numpy as np
import pytest
import dice9 as d9

from dice9.analysis import move_analysis
from dice9.environment import Environment
from dice9.factor import Register
from dice9.factor import Factor


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


def test_matmult_last_use_name_rhs_produces_repeated_same_roll():
    @d9.dist
    def f():
        x = d(6)
        return lazy_sum(2 @ x)

    pmf = f()
    for even in [2, 4, 6, 8, 10, 12]:
        assert pmf[even] == pytest.approx(1 / 6)
    for odd in [3, 5, 7, 9, 11]:
        assert odd not in pmf


def test_matmult_requires_definite_repeat_count():
    @d9.dist
    def f():
        return lazy_sum(d(2) @ d(6))

    with pytest.raises(d9.InterpreterError, match="definite value"):
        f()


def test_move_analysis_keeps_complex_matmult_rhs_non_moved():
    source = textwrap.dedent(
        """
        def f():
            x = d(6)
            return lazy_sum(2 @ (x + 0))
        """
    )
    transformed = move_analysis(ast.parse(source))
    code = ast.unparse(transformed)
    assert "2 @ (x + 0)" in code
    assert "2 @ move(x)" not in code


def test_dd5e_advantage_check_vs_dc15():
    @d9.dist
    def f():
        return max(d(20), d(20)) >= 15

    pmf = f()
    # 1 - (14/20)^2 = 0.51
    assert pmf[True] == pytest.approx(0.51)


def test_dd4d6_drop_lowest_stat_distribution_edges():
    @d9.dist
    def f():
        return reduce_sum(lazy_topk(4 @ d(6), 3), -1)

    pmf = f()
    assert pmf[3] == pytest.approx(1 / 1296)
    assert pmf[18] == pytest.approx(21 / 1296)


def test_shadowrun_six_die_pool_hits():
    @d9.dist
    def f():
        # In Shadowrun-like d6 pools, 5-6 are hits.
        return lazy_sum(6 @ (d(6) >= 5))

    pmf = f()
    # Binomial(n=6, p=1/3)
    assert pmf[0] == pytest.approx((2 / 3) ** 6)
    assert pmf[3] == pytest.approx(160 / 729)


def test_bigfraction_factor_rich_can_sort_rows():
    semiring = d9.BigFraction(64)
    probs = semiring.concat([
        semiring.const_ratio(1, 3, (1,)),
        semiring.const_ratio(2, 3, (1,)),
    ])
    factor = Factor(semiring, probs, {Register.new(): np.array([10, 20])})

    table = factor.rich()

    assert table is not None
