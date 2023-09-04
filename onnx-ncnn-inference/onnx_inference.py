import glob

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision.transforms as T
from numpy.linalg import norm
from PIL import Image


def cosine_similarity(arr1, arr2):
    dist = np.dot(arr1, arr2) / (norm(arr1) * norm(arr2))
    return dist


if __name__ == "__main__":
    onnx_model = onnx.load("./models/onnx/model.onnx")
    onnx.checker.check_model(onnx_model)

    ort_sess = ort.InferenceSession("./models/onnx/model.onnx")

    images1 = [
        "test_dataset/Aaron_Peirsol/Aaron_Peirsol_0001.jpg",
        "test_dataset/Aaron_Sorkin/Aaron_Sorkin_0001.jpg",
        "test_dataset/Alec_Baldwin/Alec_Baldwin_0001.jpg",
        "test_dataset/Jake_Gyllenhaal/Jake_Gyllenhaal_0001.jpg",
        "test_dataset/Jennifer_Garner/Jennifer_Garner_0001.jpg",
    ]

    images2 = [
        "test_dataset/Aaron_Peirsol/Aaron_Peirsol_0002.jpg",
        "test_dataset/Aaron_Sorkin/Aaron_Sorkin_0002.jpg",
        "test_dataset/Alec_Baldwin/Alec_Baldwin_0002.jpg",
        "test_dataset/Jake_Gyllenhaal/Jake_Gyllenhaal_0003.jpg",
        "test_dataset/Jennifer_Garner/Jennifer_Garner_0003.jpg",
    ]

    pixel_mean = [0.5, 0.5, 0.5]
    pixel_std = [0.5, 0.5, 0.5]
    transforms = []
    transforms += [T.Resize((112, 112))]
    transforms += [T.ToTensor()]
    transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
    preprocess = T.Compose(transforms)

    features1 = []
    features2 = []
    res = []

    for image in images1:
        img = Image.open(image).convert("RGB")
        img = preprocess(img)
        img = torch.unsqueeze(img, 0)
        out = ort_sess.run(None, {"input": img.numpy()})
        out = np.array(out).squeeze(axis=0).squeeze(axis=0)
        features1.append(out)
    features1 = np.array(features1)

    for image in images2:
        img = Image.open(image).convert("RGB")
        img = preprocess(img)
        img = torch.unsqueeze(img, 0)
        out = ort_sess.run(None, {"input": img.numpy()})
        out = np.array(out).squeeze(axis=0).squeeze(axis=0)
        features2.append(out)
    features2 = np.array(features2)

    for feat1 in features1:
        tmp = []
        for feat2 in features2:
            similarity = cosine_similarity(feat1, feat2)
            tmp.append(f"{similarity:.5f}")
        print(tmp)

    # similarity = cosine_similarity(features[0], features[1])
    # print(f"Similarity: {similarity}")
