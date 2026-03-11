import argparse
import importlib.util
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from module.base.utils import load_image
from module.ocr.models import OCR_MODEL


def parse_args():
    parser = argparse.ArgumentParser(description='OCR backend smoke test on Windows')
    parser.add_argument(
        '--backend',
        default='auto',
        choices=['auto', 'mxnet', 'onnx'],
        help='Prefer OCR backend for this run',
    )
    parser.add_argument(
        '--langs',
        default='azur_lane,azur_lane_jp,cnocr,jp,tw',
        help='Comma separated OCR models to test',
    )
    parser.add_argument(
        '--images',
        nargs='*',
        default=[],
        help='Optional image files for quick recognition output',
    )
    parser.add_argument(
        '--load-model',
        action='store_true',
        help='Actually load OCR models. This may require network/model assets.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['ALAS_OCR_BACKEND'] = args.backend

    langs = [x.strip() for x in args.langs.split(',') if x.strip()]
    print(f'ALAS_OCR_BACKEND={args.backend}')
    print(f'LANGS={langs}')
    print(f'cnocr_installed={importlib.util.find_spec("cnocr") is not None}')
    print(f'onnxruntime_installed={importlib.util.find_spec("onnxruntime") is not None}')
    print(f'load_model={args.load_model}')

    models = []
    for lang in langs:
        t0 = time.time()
        model = getattr(OCR_MODEL, lang)
        dt = time.time() - t0
        if args.load_model:
            backend = model.backend
            print(f'[{lang}] backend={backend}, init={dt:.3f}s')
        else:
            print(f'[{lang}] import_ok, init={dt:.3f}s')
        models.append((lang, model))

    if not args.load_model or not args.images:
        return

    for image_path in args.images:
        image = load_image(image_path)
        print(f'\\nIMAGE: {image_path}')
        for lang, model in models:
            t0 = time.time()
            try:
                text = ''.join(model.atomic_ocr_for_single_line(image))
                dt = time.time() - t0
                print(f'  [{lang}] {dt:.3f}s -> {text[:120]}')
            except Exception as e:
                print(f'  [{lang}] ERROR: {e}')


if __name__ == '__main__':
    main()
