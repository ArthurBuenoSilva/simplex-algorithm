from typing import Dict, List

import numpy as np
from jinja2.optimizer import optimize
from scipy.optimize import linprog


class Controller:
    def __init__(self):
        self._var_qtd = 0
        self._objective_func: Dict = dict()
        self._restrictions: List[Dict] = list()

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
            print(f"Processing restriction {i}: {restr}")  # Debug statement

            # Check for the existence of the 'operator' key
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

        res = linprog(
            c=c,
            A_ub=A_ub if A_ub else None,
            b_ub=b_ub if b_ub else None,
            A_eq=A_eq if A_eq else None,
            b_eq=b_eq if b_eq else None,
            method='simplex'
        )

        original_optimal_value = res.fun
        original_x = res.x.tolist()

        shadow_prices = []

        # Now calculate the shadow prices for each constraint
        for i, b_value in enumerate(b_ub):
            # Create a modified constraint: increase b[i] by 1
            modified_b = b_ub[:]
            modified_b[i] += 1  # Increase the constraint by 1 unit

            # Solve the LP problem again with the modified constraint
            res_modified = linprog(
                c=c,
                A_ub=A_ub if A_ub else None,
                b_ub=modified_b if modified_b else None,
                A_eq=A_eq if A_eq else None,
                b_eq=b_eq if b_eq else None,
                method='simplex'
            )

            if not res_modified.success:
                shadow_prices.append(None)  # If no solution, shadow price cannot be calculated
                continue

            # Calculate the shadow price (difference in the objective function)
            modified_optimal_value = res_modified.fun
            shadow_price = modified_optimal_value - original_optimal_value

            # Append the shadow price
            shadow_prices.append(-shadow_price)

        result = {
            'x': original_x,
            'fun': -original_optimal_value,
            'shadow_prices': shadow_prices,
            'success': res.success,
            'message': res.message
        }

        return result

    from scipy.optimize import linprog

    def post_optimization(self, variable_modifications):
        """
        Post-optimization based on modifications to the variable values.

        variable_modifications: List containing the amount to modify each variable (x1, x2, etc.).
        Example: [10, -5] means increasing x1 by 10 and decreasing x2 by 5.
        """
        # Solve the original problem to get the base result
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

        # Solve the original problem (base case)
        res = linprog(
            c=c,
            A_ub=A_ub if A_ub else None,
            b_ub=b_ub if b_ub else None,
            A_eq=A_eq if A_eq else None,
            b_eq=b_eq if b_eq else None,
            method='simplex'
        )

        if not res.success:
            return {'message': 'The original LP could not be solved successfully.', 'new_z': None}

        original_z_value = -res.fun  # The original optimal objective value

        # Modify the RHS of the constraints based on the modifications list
        b = [b[i] + float(variable_modifications[i]) for i in range(len(b))]

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

        print(A_ub, b_ub, A_eq, b_eq)

        # Solve the problem again with the modified RHS
        res_modified = linprog(
            c=c,
            A_ub=A_ub if A_ub else None,
            b_ub=b_ub if b_ub else None,
            A_eq=A_eq if A_eq else None,
            b_eq=b_eq if b_eq else None,
            method='simplex'
        )

        if not res_modified.success:
            return {'message': 'A modificação não pode ser resolvida, cheque os inputs.', 'z': None}

        new_z_value = -res_modified.fun  # The new optimal objective value with modified constraints

        # Compare the new Z value with the original one
        if new_z_value > original_z_value:
            message = "A modificação é viável; o novo valor de Z é melhor."
        elif new_z_value == original_z_value:
            message = "A modificação não tem efeito sobre o valor ótimo."
        else:
            message = "A modificação não é viável; o novo valor de Z é pior."

        return {'message': message, 'z': new_z_value}


controller = Controller()
