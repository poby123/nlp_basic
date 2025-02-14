import re
import os


def get_latest_checkpoint(checkpoint_path: str):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    files = os.listdir(checkpoint_dir)

    # .ckpt.index와 .ckpt.data-00000-of-00001 형식의 파일들만 필터링
    ckpt_files = [
        f for f in files
        if f.endswith('.ckpt.index') or f.endswith('.ckpt.data-00000-of-00001')
    ]

    # 파일명에서 숫자 추출하여 정렬
    sorted_files = sorted(
        ckpt_files,
        key=lambda x: int(re.search(r'(\d+)', x).group(0)),
        reverse=True
    )

    # 가장 높은 번호의 .ckpt만 가져오기
    latest_ckpt = re.sub(
        r'\.data-00000-of-00001$|\.index$',
        '',
        sorted_files[0]

    )
    return f'{checkpoint_dir}/{latest_ckpt}'
