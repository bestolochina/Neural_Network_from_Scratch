import numpy as np
from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
from utils.utils import get_list, full_check, custom_uniform

np.random.uniform = custom_uniform


# The source data I will test on
true_backprop_res = [0.17205149953305518]


class Tests6(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):
        reply = reply.strip().lower()

        if len(reply) == 0:
            return CheckResult.wrong("No output was printed")

        if reply.count('[') != 1 or reply.count(']') != 1:
            return CheckResult.wrong('No expected lists were found in output')

        try:
            backprop_res, _ = get_list(reply)
        except Exception:
            return CheckResult.wrong('Seems that backpropagation output is in wrong format')

        check_result = full_check(backprop_res, true_backprop_res, 'backpropagation')
        if check_result:
            return check_result

        return CheckResult.correct()


if __name__ == '__main__':
    Tests6().run_tests()
