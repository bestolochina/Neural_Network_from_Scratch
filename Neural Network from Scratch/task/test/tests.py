import numpy as np
from hstest import StageTest, TestCase, CheckResult, PlottingTest
from hstest.stage_test import List
from utils.utils import get_list, full_check, custom_uniform

np.random.uniform = custom_uniform


# The source data I will test on

true_acc_res = [0.0792]
true_acc_history_res = [0.6438, 0.8349, 0.8429, 0.8456, 0.8466, 0.848, 0.8497, 0.8507, 0.851, 0.8525, 0.8531, 0.8525,
                        0.8534, 0.8533, 0.8541, 0.8539, 0.854, 0.8541, 0.8543, 0.8549]


class Tests4(PlottingTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):
        reply = reply.strip().lower()

        if len(reply) == 0:
            return CheckResult.wrong("No output was printed")

        if reply.count('[') != 2 or reply.count(']') != 2:
            return CheckResult.wrong('No expected lists were found in output')

        # Getting the student's results from the reply

        try:
            acc_res, reply = get_list(reply)
        except Exception:
            return CheckResult.wrong('Seems that accuracy output is in wrong format')

        try:
            acc_history_res, reply = get_list(reply)
        except Exception:
            return CheckResult.wrong('Seems that accuracy log output is in wrong format')

        check_result = full_check(acc_res, true_acc_res, 'accuracy')
        if check_result:
            return check_result

        check_result = full_check(acc_history_res, true_acc_history_res, 'train sequence')
        if check_result:
            return check_result

        return CheckResult.correct()


if __name__ == '__main__':
    Tests4().run_tests()