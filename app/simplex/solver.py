from typing import Dict, List

from scipy.optimize import linprog


class SimplexSolver:
    def __init__(self, objective_func: Dict, restrictions: List[Dict], var_qtd: int):
        self._objective_func = objective_func
        self._restrictions = restrictions
        self._var_qtd = var_qtd

    def _prepare_data(self):
        c = [float(self._objective_func[f"x{i + 1}"]) for i in range(self._var_qtd)]
        if self._objective_func["operator"] == "=":
            c = [-ci for ci in c]

        A, b = [], []
        for restr in self._restrictions:
            A.append([float(restr[f"x{i + 1}"]) for i in range(self._var_qtd)])
            b.append(float(restr["result"]))

        A_ub, b_ub, A_eq, b_eq = [], [], [], []

        for i, restr in enumerate(self._restrictions):
            if "operator" not in restr:
                raise ValueError(f"Restriction {i} is missing 'operator': {restr}")

            if restr["operator"] in ["<=", "<"]:
                A_ub.append(A[i])
                b_ub.append(b[i])
            elif restr["operator"] in [">=", ">"]:
                A_ub.append([-aij for aij in A[i]])
                b_ub.append(-b[i])
            elif restr["operator"] == "=":
                A_eq.append(A[i])
                b_eq.append(b[i])

        return c, A_ub, b_ub, A_eq, b_eq

    def solve(self, modified_b_ub=None):
        c, A_ub, b_ub, A_eq, b_eq = self._prepare_data()

        if modified_b_ub:
            b_ub = modified_b_ub

        res = linprog(
            c=c,
            A_ub=A_ub if A_ub else None,
            b_ub=b_ub if b_ub else None,
            A_eq=A_eq if A_eq else None,
            b_eq=b_eq if b_eq else None,
            method="simplex",
        )
        return res, A_ub, b_ub
