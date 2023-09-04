import os
import torch
from eval import verification
from typing import List
from backbones import get_model
from utils.utils_config import get_config
import matplotlib.pyplot as plt

class Evaluate(object):

    def __init__(self, model, data_dir, result_dir, val_targets=["lfw", "cfp_fp", "agedb_30"], image_size=(112, 112)):

        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.result_dir = result_dir

        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def gen_plot(self, name, result_dir, fpr, tpr):
        """Create a pyplot plot and save to buffer."""
        plt.figure()
        plt.xlabel("FPR", fontsize=14)
        plt.ylabel("TPR", fontsize=14)
        plt.title("ROC Curve", fontsize=14)
        plot = plt.plot(fpr, tpr, linewidth=2)
        plt.savefig(f"{result_dir}/roc-{name}.jpg", format='jpg')
        plt.close()

    def ver_test(self, backbone: torch.nn.Module):
        results = []
        for i in range(len(self.ver_list)):
            tpr, fpr, acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
            print('[%s]XNorm: %f' % (self.ver_name_list[i], xnorm))
            print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], acc2, std2))

            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            print(
                '[%s]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], self.highest_acc_list[i]))
            results.append(acc2)
            self.gen_plot(self.ver_name_list[i], self.result_dir, fpr, tpr)

    def run(self):
        model.eval()
        self.ver_test(model)

if __name__ == "__main__":

    cfg = get_config("configs/custom.py")

    if not os.path.exists(cfg.result_dir):
        os.mkdir(cfg.result_dir)

    model = backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size)

    weight = torch.load("/home/jovyan/haseeb-data/insightface/work_dirs/custom/model.pt")
    model.load_state_dict(weight, strict=True)
    evaluate = Evaluate(model, data_dir=cfg.rec, result_dir=cfg.result_dir)
    evaluate.run()
