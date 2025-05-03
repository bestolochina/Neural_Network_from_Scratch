from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
from utils.utils import get_list, full_check, custom_uniform
import numpy as np

np.random.uniform = custom_uniform

# The source data I will test on
true_data_res = [0.16862745098039217, 0.403921568627451]
true_xavier_res = [-1.0715903886795484, 0.4973359442973788, 0.8762538954575065, 0.25949835095465223,
                   -0.8956623974165681, -1.0133874996404446]
true_sigmoid_res = [0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823]


class Tests1(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):
        reply = reply.strip().lower()

        if len(reply) == 0:
            return CheckResult.wrong("No output was printed")

        if reply.count('[') != 3 or reply.count(']') != 3:
            return CheckResult.wrong('No expected lists were found in output')

        # Getting the student's results from the reply

        try:
            data_res, reply = get_list(reply)
        except Exception:
            return CheckResult.wrong('Seems that scaled data output is in wrong format')

        try:
            xavier_res, reply = get_list(reply)
        except Exception:
            return CheckResult.wrong('Seems that xavier derivative output is in wrong format')

        try:
            sigmoid_res, reply = get_list(reply)
        except Exception:
            return CheckResult.wrong('Seems that sigmoid output is in wrong format')

        check_result = full_check(data_res, true_data_res, 'data scaling')
        if check_result:
            return check_result
        check_result = full_check(xavier_res, true_xavier_res, 'xavier')
        if check_result:
            return check_result
        check_result = full_check(sigmoid_res, true_sigmoid_res, 'sigmoid')
        if check_result:
            return check_result

        return CheckResult.correct()


if __name__ == '__main__':
    Tests1().run_tests()
