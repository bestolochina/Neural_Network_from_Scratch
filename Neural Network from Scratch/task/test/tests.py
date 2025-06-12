from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
from utils.utils import get_list, full_check, custom_uniform

import numpy as np
np.random.uniform = custom_uniform

# The source data I will test on

true_forward_res = [0.42286725173173645, 0.7863175754895447, 0.8539526054946633, 0.5878649450449149,
                    0.25332037818521813, 0.10846218815633141, 0.16132366288535735, 0.5036812915517839,
                    0.39649108111105275, 0.3378884293704015, 0.418550728613325, 0.7054997389006183,
                    0.7505531230813576, 0.5719368355036792, 0.14896990554768216, 0.2714895260027178,
                    0.36007653088357094, 0.5919441786952234, 0.6768202587658813, 0.5221310346087061]


class Tests2(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):
        reply = reply.strip().lower()

        if len(reply) == 0:
            return CheckResult.wrong("No output was printed")

        if reply.count('[') != 1 or reply.count(']') != 1:
            return CheckResult.wrong('No expected lists were found in output')

        try:
            forward_res, _ = get_list(reply)
        except Exception:
            return CheckResult.wrong('Seems that feedforward output is in wrong format')

        check_result = full_check(forward_res, true_forward_res, 'feedforward')
        if check_result:
            return check_result

        return CheckResult.correct()


if __name__ == '__main__':
    Tests2().run_tests()
