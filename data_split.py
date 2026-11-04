"""모든 실행 경로가 같은 이미지 목록을 쓰도록 분할 파일을 생성·검증한다.

처음 정한 분할은 자동으로 다시 만들지 않는다. 원본이 추가되어도 기존 목록만
사용하며, 목록에 있는 파일이 빠지면 중단한다. 경로는 카테고리 기준 상대 경로다.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import random


SPLIT_DIR = Path(__file__).resolve().parent / "splits"
DEFAULT_SEED = 0
DEFAULT_VAL_FRACTION = 0.2


def image_id(path):
    """로컬·Colab 경로를 train/good/파일 또는 test/종류/파일로 통일한다."""
    parts = str(path).replace("\\", "/").split("/")
    for index in range(len(parts) - 3, -1, -1):
        if parts[index] in ("train", "test"):
            result = "/".join(parts[index:])
            rel = PurePosixPath(result)
            if len(rel.parts) == 3 and all(p not in (".", "..", "") for p in rel.parts):
                if rel.parts[0] == "train" and rel.parts[1] != "good":
                    break
                return result
    raise ValueError(f"MVTec 카테고리 기준 이미지 경로가 아닙니다: {path}")


def split_digest(split):
    """공통 설정과 분할 목록의 식별자를 반환한다."""
    return hashlib.sha256(json.dumps(split, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def validate_split(split):
    """학습·검증의 중복, 테스트 혼입, 누락과 잘못된 상대 경로를 거부한다."""
    required = {"version", "category", "seed", "val_fraction", "image_size", "normal_images",
                "statistics_images", "validation_images", "test_images"}
    if not isinstance(split, dict) or not required.issubset(split) or split["version"] != 1:
        raise ValueError("지원하지 않거나 불완전한 데이터 분할 파일입니다.")
    if split["image_size"] != 256:
        raise ValueError("공통 입력 해상도는 256으로 고정되어야 합니다.")
    for key in ("normal_images", "statistics_images", "validation_images", "test_images"):
        values = split[key]
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError("분할 이미지 목록은 문자열 목록이어야 합니다.")
        if len(set(values)) != len(values) or any(image_id(v) != v for v in values):
            raise ValueError("분할 목록에 중복 또는 잘못된 경로가 있습니다.")
        prefix = "test/" if key == "test_images" else "train/good/"
        if any(not v.startswith(prefix) for v in values):
            raise ValueError("정상 학습·검증 목록에 테스트 이미지가 포함될 수 없습니다.")
    stats, val = set(split["statistics_images"]), set(split["validation_images"])
    if len(stats) < 2 or not val or stats & val or stats | val != set(split["normal_images"]):
        raise ValueError("분할은 서로 겹치지 않는 정상 통계 두 장 이상과 검증 한 장 이상이어야 합니다.")


def get_or_create_split(category, normal_paths, test_paths, split_path=None, seed=DEFAULT_SEED,
                        val_fraction=DEFAULT_VAL_FRACTION):
    """기존 분할을 우선 사용하고, 없을 때만 정렬한 목록에서 분할을 생성한다."""
    if not category or any(c in str(category) for c in ("/", "\\")) or category in (".", ".."):
        raise ValueError("카테고리 이름이 올바르지 않습니다.")
    normal = sorted(image_id(p) for p in normal_paths)
    test = sorted(image_id(p) for p in test_paths)
    if len(set(normal)) != len(normal) or len(set(test)) != len(test):
        raise ValueError("입력 이미지 목록에 중복 경로가 있습니다.")
    target = Path(split_path) if split_path is not None else SPLIT_DIR / f"{category}.json"
    if target.is_file():
        split = json.loads(target.read_text(encoding="utf-8"))
        validate_split(split)
        if split["category"] != category or split["seed"] != seed or split["val_fraction"] != val_fraction:
            raise ValueError("기존 분할과 카테고리·시드·검증 비율이 다릅니다. 최종 분할 설정을 그대로 사용하세요.")
        missing = (set(split["normal_images"]) - set(normal)) | (set(split["test_images"]) - set(test))
        if missing:
            raise ValueError(f"고정 분할의 이미지 {len(missing)}장이 없습니다. 첫 누락: {sorted(missing)[0]}")
        return split
    if len(normal) < 3 or not 0 < val_fraction < 1:
        raise ValueError("정상 이미지 세 장 이상과 0~1 사이의 검증 비율이 필요합니다.")
    order = list(normal)
    random.Random(seed).shuffle(order)
    n_val = min(max(1, round(len(order) * val_fraction)), len(order) - 2)
    split = {"version": 1, "category": category, "seed": seed, "val_fraction": val_fraction,
             "image_size": 256, "normal_images": normal,
             "statistics_images": sorted(order[n_val:]), "validation_images": sorted(order[:n_val]),
             "test_images": test}
    validate_split(split)
    target.parent.mkdir(parents=True, exist_ok=True)
    # 배타적으로 생성해 동시에 접근해도 이미 정한 최종 분할을 덮어쓰지 않는다.
    try:
        with target.open("x", encoding="utf-8") as output:
            json.dump(split, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except FileExistsError:
        return get_or_create_split(category, normal_paths, test_paths, target, seed, val_fraction)
    return split


def category_split(category, split_path=None, seed=DEFAULT_SEED, val_fraction=DEFAULT_VAL_FRACTION):
    """원본 카테고리의 파일 목록을 공통 분할에 연결한다."""
    from common import list_images

    if not category.train_good:
        raise ValueError("카테고리에 train/good 폴더가 없습니다.")
    return get_or_create_split(Path(category.root).name, list_images(category.train_good),
                               [p for paths in category.test.values() for p in paths],
                               split_path, seed, val_fraction)


def read_split_images(root, split, subset, size=256):
    """고정 목록을 순서대로 읽으며, 읽기 실패를 조용히 제외하지 않는다."""
    from common import preprocess, read_bgr

    if size != split["image_size"]:
        raise ValueError("입력 해상도가 공통 분할의 256과 다릅니다.")
    images, paths = [], []
    for relative in split[subset]:
        path = Path(root) / relative
        raw = read_bgr(str(path))
        if raw is None:
            raise ValueError(f"고정 분할 이미지를 읽을 수 없습니다: {path}")
        images.append(preprocess(raw, size))
        paths.append(str(path))
    return images, paths
