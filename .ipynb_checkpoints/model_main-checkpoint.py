# coding = utf-8

from v1_0_20260327.run_model import generate_score


class PredictMain(object):
    """
    作用: 模型推理类
    """

    def __init__(self, model_dir):
        """
        :param model_dir: string, 模型文件所在目录的绝对路径
        """
        # 模型的加载必须在 __init__ 方法中实现, 否则会导致推理响应过慢甚至报错

    def predict(self, input_dict):
        """
        作用: 主模型推理方法
        :param input_dict: dict, 模型的入参，请与调用方提前沟通好入参格式，比如 key 的大小写，value 的类型等
        :return: 模型的出参，必须是可以 json 序列化的对象，如 list, dict 等
        """
        return generate_score(input_dict)


