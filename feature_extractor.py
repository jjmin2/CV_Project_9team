"""
DefectVision - ResNet18 특징 추출 모듈 (담당: 김도원)

사용 예시 (Streamlit 앱 등에서):
    from feature_extractor import FeatureExtractor
    fe = FeatureExtractor('bottle_train_features.pt')   # 저장된 idx를 그대로 불러옴
    emb = fe.from_path('test.png')                       # (1, 100, 64, 64)
    # 또는 업로드된 이미지(numpy, RGB)라면: emb = fe.from_rgb(img_rgb)

필요 패키지: torch, torchvision, opencv-python
"""
import cv2
import torch
import torch.nn.functional as F
import torchvision

IMG_SIZE = (256, 256)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FeatureExtractor:
    def __init__(self, train_pt_path=None, *, idx=None):
        # 학습 특징 파일에 저장된 채널 번호(idx)를 그대로 사용해야 PaDiM 통계와 맞음
        if idx is None:
            if train_pt_path is None:
                raise ValueError("학습 특징 경로 또는 채널 인덱스가 필요합니다.")
            idx = torch.load(train_pt_path, map_location='cpu', weights_only=True, mmap=True)['idx']
        self.idx = torch.as_tensor(idx, dtype=torch.int64).detach().cpu().clone()

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        self.model = torchvision.models.resnet18(weights=weights)
        self.model.eval()

    def preprocess(self, img_rgb):
        # 최민준 님 전처리와 동일: 256x256 리사이즈, 0~1 스케일, (1, 3, H, W)
        img = cv2.resize(img_rgb, IMG_SIZE)
        return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    def extract_features(self, x):
        # x: (N, 3, 256, 256), 0~1 -> (N, 448, 64, 64)
        m = self.model
        x = (x - MEAN) / STD
        with torch.no_grad():
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            f1 = m.layer1(x)
            f2 = m.layer2(f1)
            f3 = m.layer3(f2)
        size = f1.shape[-2:]
        f2 = F.interpolate(f2, size=size, mode='nearest')
        f3 = F.interpolate(f3, size=size, mode='nearest')
        return torch.cat([f1, f2, f3], dim=1)

    def get_embedding(self, x):
        # (N, 3, 256, 256) -> (N, 100, 64, 64)
        return self.extract_features(x)[:, self.idx]

    def from_rgb(self, img_rgb):
        # numpy RGB 이미지 한 장 -> (1, 100, 64, 64)
        return self.get_embedding(self.preprocess(img_rgb))

    def from_path(self, img_path):
        # 이미지 파일 경로 한 장 -> (1, 100, 64, 64)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        return self.from_rgb(img)
