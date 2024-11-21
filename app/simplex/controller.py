from typing import Dict, List

from app.simplex.solver import SimplexSolver


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
        simplex_solver = SimplexSolver(self._objective_func, self._restrictions, self._var_qtd)
        res, _, b_ub = simplex_solver.solve()

        if not res.success:
            return {"message": "The problem could not be solved.", "shadow_prices": []}

        original_optimal_value = res.fun
        original_x = res.x.tolist()

        shadow_prices = []
        for i, b_value in enumerate(b_ub):
            modified_b = b_ub[:]
            modified_b[i] += 1

            res_modified, _, _ = simplex_solver.solve()

            if not res_modified.success:
                shadow_prices.append(None)
                continue

            modified_optimal_value = res_modified.fun
            shadow_price = modified_optimal_value - original_optimal_value
            shadow_prices.append(-shadow_price)

        result = {
            "x": original_x,
            "fun": -original_optimal_value,
            "shadow_prices": shadow_prices,
            "success": res.success,
            "message": res.message,
        }

        return result

    def post_optimization(self, variable_modifications):
        simplex_solver = SimplexSolver(self._objective_func, self._restrictions, self._var_qtd)

        res, _, b_ub = simplex_solver.solve()
        if not res.success:
            return {"message": "The original LP could not be solved successfully.", "new_z": None}

        original_z_value = -res.fun

        b = [b_val + float(variable_modifications[i]) for i, b_val in enumerate(b_ub)]

        res_modified, _, _ = simplex_solver.solve(b)
        if not res_modified.success:
            return {"message": "The modified LP could not be solved successfully.", "new_z": None}

        new_z_value = -res_modified.fun

        if new_z_value > original_z_value:
            message = "A modificação é viável; o novo valor de Z é melhor."
        elif new_z_value == original_z_value:
            message = "A modificação não tem efeito sobre o valor ótimo."
        else:
            message = "A modificação não é viável; o novo valor de Z é pior."

        return {"message": message, "z": new_z_value}


controller = Controller()
