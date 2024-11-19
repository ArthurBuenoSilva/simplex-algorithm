from typing import Dict, List

import numpy as np
from scipy.optimize import linprog


class Controller:
    def __init__(self):
        self._var_qtd = 0
        self._objective_func: Dict = dict()
        self._restrictions: List[Dict] = list()
        self._basic_vars: List = list()
        self._tableau: List = list()

    @property
    def var_qtd(self):
        return self._var_qtd

    @var_qtd.setter
    def var_qtd(self, value):
        self._var_qtd = value

    @property
    def restrictions(self):
        return self._restrictions

    @restrictions.setter
    def restrictions(self, value):
        self._restrictions = value

    @property
    def objective_func(self):
        return self._objective_func

    @objective_func.setter
    def objective_func(self, value):
        self._objective_func = value

    def generate_result(self):
        c = [float(self._objective_func[f'x{i + 1}']) for i in range(self._var_qtd)]
        if self._objective_func['operator'] == '=':
            c = [-ci for ci in c]

        A = []
        b = []
        for restr in self._restrictions:
            A.append([float(restr[f'x{i + 1}']) for i in range(self._var_qtd)])
            b.append(float(restr['result']))

        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []

        for i, restr in enumerate(self._restrictions):
            if 'operator' not in restr:
                raise ValueError(f"Restriction {i} is missing 'operator': {restr}")

            if restr['operator'] in ['<=', '<']:
                A_ub.append(A[i])
                b_ub.append(b[i])
            elif restr['operator'] in ['>=', '>']:
                A_ub.append([-aij for aij in A[i]])
                b_ub.append(-b[i])
            elif restr['operator'] == '=':
                A_eq.append(A[i])
                b_eq.append(b[i])

        A = np.array(A)
        b = np.array(b)
        tableau = np.hstack([A, np.eye(A.shape[0]), b.reshape(-1, 1)])
        c_row = np.hstack([c, np.zeros(A.shape[0] + 1)])
        tableau = np.vstack([tableau, c_row])

        self._basic_vars = ['s' + str(i + 1) for i in range(len(self._restrictions))]

        while any(tableau[-1, :-1] < 0):
            pivot_col = np.argmin(tableau[-1, :-1])
            if all(tableau[:-1, pivot_col] <= 0):
                raise ValueError("The problem is unbounded.")

            ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
            ratios[ratios <= 0] = np.inf
            pivot_row = np.argmin(ratios)

            pivot_value = tableau[pivot_row, pivot_col]
            tableau[pivot_row, :] /= pivot_value

            # Update other rows
            for i in range(tableau.shape[0]):
                if i != pivot_row:
                    tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

            if pivot_col < self._var_qtd:
                self._basic_vars[pivot_row] = 'x' + str(pivot_col + 1)
            elif pivot_col < self._var_qtd + len(self._restrictions):
                self._basic_vars[pivot_row] = 's' + str(pivot_col - self._var_qtd + 1)
            else:
                raise ValueError(f"Unexpected pivot column: {pivot_col}")

        tableau = tableau.round(3)
        optimal_values = [0] * self._var_qtd
        for i, var in enumerate(self._basic_vars):
            if var.startswith('x'):
                var_index = int(var[1:]) - 1
                optimal_values[var_index] = tableau[i, -1]

        shadow_prices = [
            tableau[-1, self._var_qtd + i] for i in range(len(self._restrictions))
        ]

        optimal_value_of_objective_func = float(tableau[-1, -1])
        final_tableau_with_labels = self.add_labels_to_tableau(tableau)

        self._tableau = tableau

        result = {
            'tableau': final_tableau_with_labels,
            'x': optimal_values[:self._var_qtd],
            'shadow_prices': shadow_prices,
            'fun': optimal_value_of_objective_func
        }

        return result

    def add_labels_to_tableau(self, tableau):
        rows, cols = tableau.shape

        tableau_with_labels = np.zeros((rows + 1, cols + 1), dtype=object)
        tableau_with_labels[1:, 1:] = tableau

        for i in range(rows - 1):
            tableau_with_labels[i + 1, 0] = self._basic_vars[i]
        tableau_with_labels[-1, 0] = "Z"

        tableau_with_labels[0, 0] = "VB"

        for i in range(self._var_qtd):
            tableau_with_labels[0, i + 1] = 'x' + str(i + 1)

        slack_index = self._var_qtd
        for i in range(slack_index, slack_index + rows - 1):
            tableau_with_labels[0, i + 1] = 's' + str(i - slack_index + 1)

        tableau_with_labels[0, -1] = 'LD'

        tableau_with_labels = np.insert(tableau_with_labels, 1, tableau_with_labels[-1], axis=0)

        tableau_with_labels = np.delete(tableau_with_labels, -1, axis=0)

        return tableau_with_labels.tolist()

    def post_optimization(self, deltas):
        num_vars = len(deltas)
        num_constraints = len(self._tableau) - 1
        deltas = [float(delta) for delta in deltas]

        constraints = []
        for i in range(num_constraints):
            equation = [
                self._tableau[i, j] for j in range(num_vars)
            ]
            rhs = self._tableau[i, -1]
            constraints.append((equation, rhs))

        print(constraints)
        z_coefficients = self._tableau[-1, :num_vars]
        z_constant = self._tableau[-1, -1]

        feasible = True
        for equation, rhs in constraints:
            lhs = sum(coeff * delta for coeff, delta in zip(equation, deltas)) + rhs
            if lhs < 0:
                feasible = False
                break

        if not feasible:
            return {
                "z": z_constant,
                "feasible": False,
                "message": "Os valores de delta não são viáveis; as restrições foram violadas."
            }

        new_z_value = sum(coeff * delta for coeff, delta in zip(z_coefficients, deltas)) + z_constant

        original_z_value = float(self._tableau[-1, -1])

        if new_z_value > original_z_value:
            message = "A modificação é viável; o novo valor de Z é melhor."
        elif new_z_value == original_z_value:
            message = "A modificação não tem efeito sobre o valor ótimo."
        else:
            message = "A modificação não é viável; o novo valor de Z é pior."

        return {
            "feasible": True,
            "z": new_z_value,
            "message": message
        }


controller = Controller()
