import numpy as np
import onnx
import torch


def convert_jit(net, path_module, output):
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(float)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()

    weight = torch.load(path_module)
    net.load_state_dict(weight, strict=True)
    net.eval()

    mod = torch.jit.trace(net, img)
    mod.save(output)
    
if __name__ == '__main__':
    import os
    import argparse
    from backbones import get_model
    from utils.utils_config import get_config

    cfg = get_config("configs/custom.py")

    parser = argparse.ArgumentParser(description='ArcFace PyTorch to torch jit')
    parser.add_argument('--input', type=str, default=cfg.output + "/model.pt", help='input backbone.pth file or path')
    parser.add_argument('--output', type=str, default="model.pt", help='output torch jit path')
    parser.add_argument('--network', type=str, default=cfg.network, help='backbone network')
    args = parser.parse_args()
    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "model.pt")
    assert os.path.exists(input_file)

    assert args.network is not None
    print(args)
    backbone = get_model(
        args.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size, global_layer="GDC")
    if args.output is None:
        args.output = os.path.join(args.input, "model.pt")
    convert_jit(backbone, input_file, args.output)
