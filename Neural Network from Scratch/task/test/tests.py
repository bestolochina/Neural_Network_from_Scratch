import numpy as np
from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
from utils.utils import get_list, full_check, custom_uniform
np.random.uniform = custom_uniform

# The source data I will test on

true_mse_res = [9.0]
true_mse_der_res = [-10, -6, -2, 2]
true_sigmoid_der_res = [0.19661193324148185, 0.25, 0.19661193324148185, 0.10499358540350662]
true_backprop_res = [0.027703041616827684]


class Tests3(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):
        reply = reply.strip().lower()

        if len(reply) == 0:
            return CheckResult.wrong("No output was printed")

        if reply.count('[') != 4 or reply.count(']') != 4:
            return CheckResult.wrong('No expected lists were found in output')

        # Getting the student's results from the reply

        try:
            mse_res, reply = get_list(reply)
        except Exception:
            return CheckResult.wrong('Seems that MSE output is in wrong format')

        try:
            mse_der_res, reply = get_list(reply)
        except Exception:
            return CheckResult.wrong('Seems that MSE derivative output is in wrong format')

        try:
            sigmoid_der_res, reply = get_list(reply)
        except Exception:
            return CheckResult.wrong('Seems that sigmoid derivative output is in wrong format')

        try:
            backprop_res, _ = get_list(reply)
        except Exception:
            return CheckResult.wrong('Seems that backpropagation output is in wrong format')

        check_result = full_check(mse_res, true_mse_res, 'MSE')
        if check_result:
            return check_result
        check_result = full_check(mse_der_res, true_mse_der_res, 'MSE derivative')
        if check_result:
            return check_result
        check_result = full_check(sigmoid_der_res, true_sigmoid_der_res, 'sigmoid derivative')
        if check_result:
            return check_result
        check_result = full_check(backprop_res, true_backprop_res, 'backpropagation')
        if check_result:
            return check_result

        return CheckResult.correct()


if __name__ == '__main__':
    Tests3().run_tests()
