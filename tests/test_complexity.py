from minteval.complexity import nesting_depth, state_complexity


def test_state_complexity_pooled():
    sc = state_complexity({"a": [1, 1, 1, 1, 1, 1, 1, 1, 1, 10], "b": [], "c": [2]})
    assert sc["tau_max"] == 10 and sc["n_registers"] == 2


def test_nesting_elif_not_counted():
    src = ("def f(x):\n"
           "    if x:\n"
           "        if x > 1:\n"
           "            return 1\n"
           "    elif x < 0:\n"
           "        return 2\n"
           "    else:\n"
           "        return 3\n")
    assert nesting_depth(src) == 2
