import numpy as np
from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
from utils.utils import get_list, full_check, custom_uniform

np.random.uniform = custom_uniform

# The source data I will test on

true_forward_res = [0.08145814974665896, 0.7078373861901438, 0.7880474359319048, 0.5460663760416861,
                    0.4142727590233215, 0.3900817619848179, 0.30867710854793123, 0.8821701624025712,
                    0.5251529930398801, 0.6860112311436223, 0.10830624423659274, 0.7042764793747258,
                    0.7896127455170074, 0.5666891692622699, 0.42785055300952424, 0.3886295210179232,
                    0.31683637641989687, 0.8657427046740684, 0.5576544572719793, 0.7183489046929683]


class Tests5(StageTest):

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
            return CheckResult.wrong('Seems that MSE output is in wrong format')

        check_result = full_check(forward_res, true_forward_res, 'feedforward')
        if check_result:
            return check_result

        return CheckResult.correct()


if __name__ == '__main__':
    Tests5().run_tests()
