from abc import abstractmethod
from typing import Union, List
import logging
import numpy as np


class Formula:
    def __init__(self, *subops):
        self.subops = subops

    @abstractmethod
    def eval(self, **kwargs):
        pass

    @abstractmethod
    def show(self, **kwargs) -> str:
        pass

    @abstractmethod
    def tokens(self, **kwargs) -> List[str]:
        pass

    def depth(self):
        return max(s.depth() for s in self.subops) + 1


class Object(Formula):
    def __init__(self, object_string):
        self.object_string = object_string

    def eval(self, **kwargs):
        return kwargs[self.object_string]

    def show(self, **kwargs):
        if len(kwargs) > 0:
            return str(self.eval(**kwargs))
        else:
            return self.object_string

    def tokens(self):
        return [self.object_string]

    def depth(self):
        return 1


class Arithmetics(Formula):
    def __init__(self, *subops: Union[Formula, Object]):
        self.subops = subops

    @abstractmethod
    def eval(self, **kwargs) -> Union[float, int]:
        pass

    @abstractmethod
    def tokens(self) -> List[str]:
        pass


class Add(Arithmetics):
    def __init__(self, *subops):
        super(Add, self).__init__(*subops)

    def eval(self, **kwargs):
        return sum(ops.eval(**kwargs) for ops in self.subops)

    def show(self, **kwargs):
        return f"({'+'.join(ops.show(**kwargs) for ops in self.subops)})"

    def tokens(self):
        tokens = ["(", "+", ","]
        for ops in self.subops:
            tokens.extend(ops.tokens())
            tokens.append(",")
        tokens.pop(-1)
        tokens.append(")")
        return tokens


class Sub(Arithmetics):
    def __init__(self, *subops):
        super(Sub, self).__init__(*subops)

    def eval(self, **kwargs):
        assert len(self.subops) == 2
        l, r = self.subops
        return l.eval(**kwargs) - r.eval(**kwargs)

    def show(self, **kwargs):
        assert len(self.subops) == 2
        l, r = self.subops
        return f"({l.show(**kwargs)}-{r.show(**kwargs)})"

    def tokens(self):
        assert len(self.subops) == 2
        l, r = self.subops
        tokens = ["(", "-", ","]
        tokens.extend(l.tokens())
        tokens.append(",")
        tokens.extend(r.tokens())
        tokens.append(")")
        return tokens


class Mul(Arithmetics):
    def __init__(self, *subops):
        super(Mul, self).__init__(*subops)

    def eval(self, **kwargs):
        ans = 1
        for ops in self.subops:
            ans *= ops.eval(**kwargs)
        return ans

    def show(self, **kwargs):
        return f"({'*'.join(ops.show(**kwargs) for ops in self.subops)})"

    def tokens(self):
        tokens = ["(", "+", ","]
        for ops in self.subops:
            tokens.extend(ops.tokens())
            tokens.append(",")
        tokens.pop(-1)
        tokens.append(")")
        return tokens


class Div(Arithmetics):
    def __init__(self, *subops):
        super(Div, self).__init__(*subops)

    def eval(self, **kwargs):
        assert len(self.subops) == 2
        l, r = self.subops
        if r.eval(**kwargs) != 0:
            return l.eval(**kwargs) / r.eval(**kwargs)
        else:
            return float('inf')

    def show(self, **kwargs):
        assert len(self.subops) == 2
        l, r = self.subops
        return f"({l.show(**kwargs)}/{r.show(**kwargs)})"

    def tokens(self):
        assert len(self.subops) == 2
        l, r = self.subops
        tokens = ["(", "/", ","]
        tokens.extend(l.tokens())
        tokens.append(",")
        tokens.extend(r.tokens())
        tokens.append(")")
        return tokens


class Predicates(Formula):
    def __init__(self, *subops: Arithmetics):
        self.subops = subops

    @abstractmethod
    def eval(self, **kwargs) -> bool:
        pass


class Eq(Predicates):
    def __init__(self, *subops):
        assert len(subops) == 2
        super(Eq, self).__init__(*subops)

    def eval(self, **kwargs):
        l, r = self.subops
        return np.abs(l.eval(**kwargs) - r.eval(**kwargs)) < 1e-6

    def show(self, **kwargs):
        assert len(self.subops) == 2
        l, r = self.subops
        return f"({l.show(**kwargs)} = {r.show(**kwargs)})"

    def tokens(self):
        assert len(self.subops) == 2
        l, r = self.subops
        tokens = ["(", "=", ","]
        tokens.extend(l.tokens())
        tokens.append(",")
        tokens.extend(r.tokens())
        tokens.append(")")
        return tokens


class GreaterThan(Predicates):
    def __init__(self, *subops):
        assert len(subops) == 2
        super(GreaterThan, self).__init__(*subops)

    def eval(self, **kwargs):
        l, r = self.subops
        return l.eval(**kwargs) > r.eval(**kwargs)

    def show(self, **kwargs):
        assert len(self.subops) == 2
        l, r = self.subops
        return f"({l.show(**kwargs)} > {r.show(**kwargs)})"

    def tokens(self):
        assert len(self.subops) == 2
        l, r = self.subops
        tokens = ["(", ">", ","]
        tokens.extend(l.tokens())
        tokens.append(",")
        tokens.extend(r.tokens())
        tokens.append(")")
        return tokens


class LessThan(Predicates):
    def __init__(self, *subops):
        assert len(subops) == 2
        super(LessThan, self).__init__(*subops)

    def eval(self, **kwargs):
        l, r = self.subops
        return l.eval(**kwargs) < r.eval(**kwargs)

    def show(self, **kwargs):
        assert len(self.subops) == 2
        l, r = self.subops
        return f"({l.show(**kwargs)} < {r.show(**kwargs)})"

    def tokens(self):
        assert len(self.subops) == 2
        l, r = self.subops
        tokens = ["(", "<", ","]
        tokens.extend(l.tokens())
        tokens.append(",")
        tokens.extend(r.tokens())
        tokens.append(")")
        return tokens


class GreaterEqual(Predicates):
    def __init__(self, *subops):
        assert len(subops) == 2
        super(GreaterEqual, self).__init__(*subops)

    def eval(self, **kwargs):
        l, r = self.subops
        return l.eval(**kwargs) >= r.eval(**kwargs)

    def show(self, **kwargs):
        assert len(self.subops) == 2
        l, r = self.subops
        return f"({l.show(**kwargs)} >= {r.show(**kwargs)})"

    def tokens(self):
        assert len(self.subops) == 2
        l, r = self.subops
        tokens = ["(", ">=", ","]
        tokens.extend(l.tokens())
        tokens.append(",")
        tokens.extend(r.tokens())
        tokens.append(")")
        return tokens


class LessEqual(Predicates):
    def __init__(self, *subops):
        assert len(subops) == 2
        super(LessEqual, self).__init__(*subops)

    def eval(self, **kwargs):
        l, r = self.subops
        return l.eval(**kwargs) <= r.eval(**kwargs)

    def show(self, **kwargs):
        assert len(self.subops) == 2
        l, r = self.subops
        return f"({l.show(**kwargs)} <= {r.show(**kwargs)})"

    def tokens(self):
        assert len(self.subops) == 2
        l, r = self.subops
        tokens = ["(", "<=", ","]
        tokens.extend(l.tokens())
        tokens.append(",")
        tokens.extend(r.tokens())
        tokens.append(")")
        return tokens


class NotEqual(Predicates):
    def __init__(self, *subops):
        assert len(subops) == 2
        super(NotEqual, self).__init__(*subops)

    def eval(self, **kwargs):
        l, r = self.subops
        return l.eval(**kwargs) != r.eval(**kwargs)

    def show(self, **kwargs):
        assert len(self.subops) == 2
        l, r = self.subops
        return f"({l.show(**kwargs)} != {r.show(**kwargs)})"

    def tokens(self):
        assert len(self.subops) == 2
        l, r = self.subops
        tokens = ["(", "!=", ","]
        tokens.extend(l.tokens())
        tokens.append(",")
        tokens.extend(r.tokens())
        tokens.append(")")
        return tokens


class Logical(Formula):
    def __init__(self, *subops: Predicates):
        self.subops = subops

    @abstractmethod
    def eval(self, **kwargs) -> bool:
        pass


class Conjunction(Logical):
    def __init__(self, *subops):
        assert len(subops) == 2
        super(Conjunction, self).__init__(*subops)

    def eval(self, **kwargs):
        l, r = self.subops
        return l.eval(**kwargs) and r.eval(**kwargs)

    def show(self, **kwargs):
        assert len(self.subops) == 2
        l, r = self.subops
        return f"({l.show(**kwargs)} & {r.show(**kwargs)})"

    def tokens(self):
        assert len(self.subops) == 2
        l, r = self.subops
        tokens = ["(", "&", ","]
        tokens.extend(l.tokens())
        tokens.append(",")
        tokens.extend(r.tokens())
        tokens.append(")")
        return tokens


class Disjunction(Logical):
    def __init__(self, *subops):
        assert len(subops) == 2
        super(Disjunction, self).__init__(*subops)

    def eval(self, **kwargs):
        l, r = self.subops
        return l.eval(**kwargs) or r.eval(**kwargs)

    def show(self, **kwargs):
        assert len(self.subops) == 2
        l, r = self.subops
        return f"({l.show(**kwargs)} | {r.show(**kwargs)})"

    def tokens(self):
        assert len(self.subops) == 2
        l, r = self.subops
        tokens = ["(", "|", ","]
        tokens.extend(l.tokens())
        tokens.append(",")
        tokens.extend(r.tokens())
        tokens.append(")")
        return tokens


class Negation:
    def __init__(self, subop: Union[Predicates, Logical]):
        self.subop = subop

    def eval(self, **kwargs):
        return not self.subop.eval(**kwargs)

    def show(self, **kwargs):
        return f"!{self.subop.show(**kwargs)}"

    def tokens(self):
        tokens = ["(", "1", ","]
        tokens.extend(self.subop.tokens())
        tokens.append(")")
        return tokens


class Parser:
    operators = {
        "+": Add,
        "-": Sub,
        "*": Mul,
        "/": Div,
        "=": Eq,
        ">": GreaterThan,
        "<": LessThan,
        ">=": GreaterEqual,
        "<": LessThan,
        "<=": GreaterEqual,
        "!=": NotEqual,
        "&": Conjunction,
        "|": Disjunction,
        "!": Negation
    }

    # def __init__(self, lisp):
    # self.original = lisp
    # self.formula = Parser.parse_recursive(self.original)

    @classmethod
    def parentheses_check(cls, string):
        depth_of_stack = 0
        max_depth_of_stack = 0
        for c in string:
            if c == '(':
                depth_of_stack += 1
                max_depth_of_stack = max(max_depth_of_stack, depth_of_stack)
            elif c == ')':
                depth_of_stack -= 1
            if depth_of_stack < 0:
                return -1
        if depth_of_stack != 0:
            return -1
        return max_depth_of_stack

    @classmethod
    def parse_recursive(cls, input_string):
        # remove the braces
        if cls.parentheses_check(input_string) == 0:
            if input_string == '!ENTAIL':
                return input_string
            elif input_string == 'NA':
                return input_string
            else:
                assert '[num:' in input_string
                return Object(object_string=input_string.strip())

        # cur = input_string.strip('()')
        cur = input_string[1:-1]
        # get the operator
        operator_string, raw_operands_string = cur.split(',', 1)
        # get the operand_strings list
        # there are two definition of the operands
        # the first way is the comma devision
        # the second way is the comma devision plus parentheses
        operand_strings = []
        head_segment = ""
        while len(raw_operands_string) > 0:
            if not ',' in raw_operands_string:
                head = raw_operands_string
                tail = ""
            else:
                head, tail = raw_operands_string.split(',', 1)
            if len(head_segment) > 0:
                head = ",".join([head_segment, head])

            pck = cls.parentheses_check(head)

            if pck >= 0:  # we have a valid lisp
                operand_strings.append(head)
                raw_operands_string = tail
                head_segment = ""
            else:  # we have an invalid lisp
                head_segment = head
                raw_operands_string = tail

        operands = [Parser.parse_recursive(opstr) for opstr in operand_strings]
        operation = Parser.operators[operator_string.strip()](*operands)
        return operation


class LISPCausalLMHead:
    def __init__(self, hidden_states_dim, hidden_dim, lisp_vocabulary):
        self.hidden_states_dim = hidden_states_dim
        self.hidden_dim = hidden_dim
        self.lisp_vocabulary = lisp_vocabulary


def robust_parse(s):
    try:
        return Parser.parse_recursive(s)
    except:
        return s


def nli_label_decider(eformula, cformula, num_dict):
    eformula = robust_parse(eformula)
    cformula = robust_parse(cformula)
    if isinstance(eformula, Formula):
        try:
            e_value = eformula.eval(**num_dict)
        except:
            logging.info('eval failure')
            e_value = None
    else:
        logging.info(f'not a formula {eformula}')
        e_value = None

    if isinstance(cformula, Formula):
        try:
            c_value = cformula.eval(**num_dict)
        except:
            c_value = None
    elif cformula == '!ENTAIL':
        c_value = not e_value
    else:
        c_value = None

    if e_value is None and c_value is None:
        return "undecidable"

    if (e_value == True) and (c_value != True):
        return "entailment"

    if (e_value != True) and (c_value == True):
        return "contradiction"

    if (e_value != True) and (c_value != True):
        return "neutral"


if __name__ == "__main__":
    test_lisps = [
        '(&,(>=,M1,(/,N1,N2)),(!,(!=,M2,N3)))'
    ]
    for lisp in test_lisps:
        formula = Parser.parse_recursive(lisp)
        print(lisp, formula.show(), formula.tokens(), "".join(formula.tokens()))
