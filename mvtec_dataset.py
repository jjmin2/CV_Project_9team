"""노트북과 앱이 같은 분할·전처리를 사용하도록 제공하는 MVTec 데이터셋."""

from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from common import mvtec_mask_path, preprocess, read_bgr, read_mask_file, scan_mvtec
from data_split import category_split


class MVTecDataset(Dataset):
    def __init__(self, data_dir, category, split="train", img_size=(256, 256),
                 val_ratio=0.2, seed=0, split_path=None):
        if tuple(img_size) != (256, 256) or split not in ("train", "val", "test"):
            raise ValueError("입력은 256×256이며 분할은 train, val, test 중 하나여야 합니다.")
        root = Path(data_dir) / category
        self.category = scan_mvtec(str(root))
        if self.category is None:
            raise ValueError("MVTec 카테고리의 train/test 폴더 구조를 확인하세요.")
        self.split_definition = category_split(self.category, split_path, seed, val_ratio)
        subset = {"train": "statistics_images", "val": "validation_images", "test": "test_images"}[split]
        relative = self.split_definition[subset]
        self.image_paths = [str(root / p) for p in relative]
        self.labels = [int(p.startswith("test/") and p.split("/")[1] != "good") for p in relative]
        self.mask_paths = [mvtec_mask_path(self.category, p.split("/")[1], str(root / p)) if label else None
                           for p, label in zip(relative, self.labels)]
        if any(label and path is None for label, path in zip(self.labels, self.mask_paths)):
            raise ValueError("불량 이미지에 대응하는 정답 마스크가 없습니다.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        raw = read_bgr(path)
        if raw is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
        rgb = cv2.cvtColor(preprocess(raw, 256), cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255
        mask = torch.zeros((1, 256, 256))
        if self.mask_paths[idx] is not None:
            from common import preprocess_mask

            raw_mask = read_mask_file(self.mask_paths[idx])
            if raw_mask is None:
                raise ValueError("정답 마스크를 읽을 수 없습니다.")
            mask = torch.from_numpy(preprocess_mask(raw_mask, 256)).float().unsqueeze(0)
        return image, self.labels[idx], mask, path
