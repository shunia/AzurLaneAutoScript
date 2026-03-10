import os
import sys

import cv2
import numpy as np
from PIL import Image

from module.exception import RequestHumanTakeover
from module.logger import logger


def _get_model_dir(base_dir):
    """Support both frozen (PyInstaller) and normal execution."""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_path, base_dir)


class AlOcr:
    """
    Cross-platform OCR wrapper.

    Backends:
      - 'mxnet': legacy CnOCR 1.x path (Windows historical compatibility)
      - 'onnx' : ONNX Runtime path via CnOCR 2.x (cross-platform target)

    Backend can be selected by:
      1) env var `ALAS_OCR_BACKEND` in ['auto', 'mxnet', 'onnx']
      2) deploy config `OcrBackend` in ['auto', 'mxnet', 'onnx']
      3) fallback to 'auto'
    """

    CNOCR_CONTEXT = 'cpu'

    def __init__(
            self,
            model_name='densenet-lite-gru',
            model_epoch=None,
            cand_alphabet=None,
            root='./bin/cnocr_models/azur_lane',
            context=None,
            name=None,
    ):
        self._model_name = model_name
        self._model_epoch = model_epoch
        self._cand_alphabet = cand_alphabet
        self._root = root
        self._context = self.CNOCR_CONTEXT if context is None else context
        self._name = name
        self._model_loaded = False
        self._backend = None  # 'mxnet' or 'onnx'

    @staticmethod
    def _normalize_backend(value):
        if value is None:
            return ''
        value = str(value).strip().lower()
        if value in ['auto', 'mxnet', 'onnx']:
            return value
        return ''

    def _get_preferred_backend(self):
        env_backend = self._normalize_backend(os.environ.get('ALAS_OCR_BACKEND'))
        if env_backend:
            return env_backend

        try:
            from module.webui.setting import State
            cfg_backend = self._normalize_backend(getattr(State.deploy_config, 'OcrBackend', 'auto'))
            if cfg_backend:
                return cfg_backend
        except Exception:
            pass

        return 'auto'

    def _resolve_model_dir(self):
        root = self._root
        if os.path.isabs(root):
            return root
        root = root.replace('\\', '/').lstrip('./')
        return _get_model_dir(root)

    @staticmethod
    def _mxnet_available():
        try:
            import mxnet  # noqa: F401
            from cnocr.cn_ocr import gen_network  # noqa: F401
            from cnocr.hyperparams.cn_hyperparams import CnHyperparams  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def _onnx_available():
        try:
            import onnxruntime  # noqa: F401
            from cnocr import CnOcr  # noqa: F401
            return True
        except Exception:
            return False

    def _detect_backend(self):
        preferred = self._get_preferred_backend()
        if preferred == 'mxnet':
            order = ['mxnet', 'onnx']
        elif preferred == 'onnx':
            order = ['onnx', 'mxnet']
        else:
            order = ['mxnet', 'onnx']

        for backend in order:
            if backend == 'mxnet' and self._mxnet_available():
                return 'mxnet'
            if backend == 'onnx' and self._onnx_available():
                return 'onnx'

        logger.critical(
            'No OCR backend found. Install one of:\n'
            '  - legacy: cnocr==1.2.2 + mxnet==1.6.0\n'
            '  - target: "cnocr[ort-cpu]" + onnxruntime'
        )
        raise RequestHumanTakeover

    def _load(self):
        if self._model_loaded:
            return

        primary = self._detect_backend()
        backups = [b for b in ['mxnet', 'onnx'] if b != primary]
        for backend in [primary] + backups:
            try:
                if backend == 'mxnet':
                    self._load_mxnet()
                else:
                    self._load_onnx()
                self._model_loaded = True
                return
            except Exception as e:
                logger.error(f'Failed to load OCR backend={backend}: {e}')

        logger.critical('All OCR backends failed to load')
        raise RequestHumanTakeover

    @property
    def backend(self):
        self._load()
        return self._backend

    def _load_mxnet(self):
        """Load using legacy MXNet-based CnOCR 1.x."""
        model_dir = self._resolve_model_dir()
        logger.info(f'Loading OCR model (MXNet backend): {model_dir}')
        from cnocr.cn_ocr import gen_network, load_module, read_charset
        from cnocr.hyperparams.cn_hyperparams import CnHyperparams as Hyperparams

        if self._model_epoch is None:
            logger.critical('Model epoch is required for MXNet backend')
            raise RequestHumanTakeover

        model_file_prefix = 'cnocr-v1.2.0-{}'.format(self._model_name)
        model_files = [
            'label_cn.txt',
            '%s-%04d.params' % (model_file_prefix, self._model_epoch),
            '%s-symbol.json' % model_file_prefix,
        ]
        for f in model_files:
            fp = os.path.join(model_dir, f)
            if not os.path.exists(fp):
                logger.critical(f'OCR model file missing: {fp}')
                raise RequestHumanTakeover

        self._alphabet, self._inv_alph_dict = read_charset(
            os.path.join(model_dir, 'label_cn.txt')
        )
        self._hp = Hyperparams()
        self._hp._loss_type = None
        self._hp._num_classes = len(self._alphabet)
        self._net_prefix = None if self._name == '' else self._name

        network, self._hp = gen_network(self._model_name, self._hp, self._net_prefix)
        hp = self._hp
        prefix = os.path.join(model_dir, model_file_prefix)
        data_names = ['data']
        data_shapes = [(data_names[0], (hp.batch_size, 1, hp.img_height, hp.img_width))]
        self._mod = load_module(
            prefix,
            self._model_epoch,
            data_names,
            data_shapes,
            network=network,
            net_prefix=self._net_prefix,
            context=self._context,
        )
        self._backend = 'mxnet'

    def _create_cnocr(self, cnocr_cls, rec_model_name):
        """
        CnOCR constructor changed across versions.
        Try rec_model_name first, then model_name.
        """
        attempts = [
            {'rec_model_name': rec_model_name},
            {'model_name': rec_model_name},
        ]

        for kwargs in attempts:
            try:
                return cnocr_cls(**kwargs)
            except TypeError:
                continue
            except Exception as e:
                logger.warning(f'CnOcr init failed with {kwargs}: {e}')

        return None

    def _load_onnx(self):
        """Load using ONNX Runtime with CnOCR models."""
        model_dir = self._resolve_model_dir()
        logger.info(f'Loading OCR model (ONNX backend): {model_dir}')
        try:
            from cnocr import CnOcr
        except ImportError:
            logger.critical(
                'cnocr package not found. Install it: pip install "cnocr[ort-cpu]"'
            )
            raise RequestHumanTakeover

        # CnOCR v2 recommended naming (without *_rec suffix)
        model_map = {
            'azur_lane': ['en_PP-OCRv3', 'en_PP-OCRv4'],
            'azur_lane_jp': ['en_PP-OCRv3', 'en_PP-OCRv4'],
            'cnocr': ['ch_PP-OCRv3', 'ch_PP-OCRv4'],
            'jp': ['japan_PP-OCRv3', 'en_PP-OCRv3'],
            'tw': ['chinese_cht_PP-OCRv3', 'chinese_cht_PP-OCRv4', 'ch_PP-OCRv3'],
        }
        dir_name = os.path.basename(os.path.normpath(model_dir))
        candidates = model_map.get(dir_name, ['ch_PP-OCRv3'])

        self._cnocr_v2 = None
        for model_name in candidates:
            logger.info(f'  Trying CnOCR model: {model_name}')
            self._cnocr_v2 = self._create_cnocr(CnOcr, rec_model_name=model_name)
            if self._cnocr_v2 is not None:
                logger.info(f'  Using CnOCR model: {model_name}')
                break

        if self._cnocr_v2 is None:
            logger.critical('Unable to initialize CnOcr ONNX model')
            raise RequestHumanTakeover

        self._backend = 'onnx'

        # Read charset for cand_alphabet support
        charset_file = os.path.join(model_dir, 'label_cn.txt')
        if os.path.exists(charset_file):
            with open(charset_file, 'r', encoding='utf-8') as f:
                self._alphabet = [line.strip() for line in f if line.strip()]
        else:
            self._alphabet = None

    def ocr(self, img_fp):
        self._load()
        if self._backend == 'mxnet':
            return self._mxnet_ocr(img_fp)
        return self._onnx_ocr(img_fp)

    def ocr_for_single_line(self, img_fp):
        self._load()
        if self._backend == 'mxnet':
            return self._mxnet_ocr_for_single_line(img_fp)
        return self._onnx_ocr_for_single_line(img_fp)

    def ocr_for_single_lines(self, img_list):
        self._load()
        if self._backend == 'mxnet':
            return self._mxnet_ocr_for_single_lines(img_list)
        return self._onnx_ocr_for_single_lines(img_list)

    def set_cand_alphabet(self, cand_alphabet):
        self._cand_alphabet = cand_alphabet

    def atomic_ocr(self, img_fp, cand_alphabet=None):
        self.set_cand_alphabet(cand_alphabet)
        return self.ocr(img_fp)

    def atomic_ocr_for_single_line(self, img_fp, cand_alphabet=None):
        self.set_cand_alphabet(cand_alphabet)
        return self.ocr_for_single_line(img_fp)

    def atomic_ocr_for_single_lines(self, img_list, cand_alphabet=None):
        self.set_cand_alphabet(cand_alphabet)
        return self.ocr_for_single_lines(img_list)

    def debug(self, img_list):
        img_list = [img.astype(np.uint8) for img in img_list]
        image = cv2.hconcat(img_list) if len(img_list) > 1 else img_list[0]
        Image.fromarray(image).show()

    def _mxnet_ocr(self, img_fp):
        """Reproduce CnOcr.ocr() behavior from cnocr 1.x."""
        lines = self._split_lines(img_fp)
        return [self._mxnet_ocr_for_single_line(line) for line in lines]

    def _mxnet_ocr_for_single_line(self, img_fp):
        img = self._preprocess_img_array(img_fp)
        img_arr, img_widths = self._pad_arrays([img])
        import mxnet as mx
        img_mx = mx.nd.array(img_arr)
        self._mod.forward(mx.io.DataBatch(data=[img_mx]))
        prob = self._mod.get_outputs()[0].asnumpy()
        return self._gen_line_pred_chars(prob[0], img_widths[0], img_arr.shape[-1])

    def _mxnet_ocr_for_single_lines(self, img_list):
        return [self._mxnet_ocr_for_single_line(img) for img in img_list]

    def _preprocess_img_array(self, img):
        hp = self._hp
        new_width = int(round(hp.img_height / img.shape[0] * img.shape[1]))
        img = cv2.resize(img, (new_width, hp.img_height))
        img = np.expand_dims(img, 0).astype('float32') / 255.0
        return img

    def _pad_arrays(self, img_list):
        hp = self._hp
        max_width = max(img.shape[-1] for img in img_list)
        max_width = max(max_width, hp.img_width)
        widths = [img.shape[-1] for img in img_list]
        padded = np.zeros((len(img_list), 1, hp.img_height, max_width), dtype='float32')
        for i, img in enumerate(img_list):
            padded[i, :, :, :img.shape[-1]] = img
        return padded, widths

    def _gen_line_pred_chars(self, line_prob, img_width, max_img_width):
        from cnocr.fit.ctc_metrics import CtcMetrics
        class_ids = np.argmax(line_prob, axis=-1)
        class_ids *= np.max(line_prob, axis=-1) > 0.5
        hp = self._hp
        if img_width < max_img_width:
            end_idx = img_width // hp.seq_len_cmpr_ratio
            if end_idx < len(class_ids):
                class_ids[end_idx:] = 0
        prediction, _ = CtcMetrics.ctc_label(class_ids.tolist())
        alphabet = self._alphabet
        return [alphabet[p] if alphabet[p] != '<space>' else ' ' for p in prediction]

    def _split_lines(self, img):
        """Stub: treat entire image as single line."""
        return [img]

    def _extract_onnx_text(self, result):
        if result is None:
            return ''
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            for key in ['text', 'txt']:
                value = result.get(key)
                if isinstance(value, str):
                    return value
            for key in ['texts', 'results', 'res']:
                value = result.get(key)
                if value is not None:
                    return self._extract_onnx_text(value)
            return ''
        if isinstance(result, (list, tuple)):
            if not result:
                return ''
            if all(isinstance(x, str) for x in result):
                return ''.join(result)
            if len(result) == 2 and isinstance(result[0], str):
                return result[0]
            parts = [self._extract_onnx_text(x) for x in result]
            parts = [p for p in parts if p]
            return ''.join(parts)

        return str(result)

    def _onnx_ocr(self, img_fp):
        result = self._cnocr_v2.ocr(img_fp)
        if not isinstance(result, (list, tuple)):
            result = [result]
        return [self._filter_by_alphabet(self._extract_onnx_text(item)) for item in result]

    def _onnx_ocr_for_single_line(self, img_fp):
        result = self._cnocr_v2.ocr_for_single_line(img_fp)
        return self._filter_by_alphabet(self._extract_onnx_text(result))

    def _onnx_ocr_for_single_lines(self, img_list):
        if hasattr(self._cnocr_v2, 'ocr_for_single_lines'):
            results = self._cnocr_v2.ocr_for_single_lines(img_list)
        else:
            results = [self._cnocr_v2.ocr_for_single_line(img) for img in img_list]
        return [self._filter_by_alphabet(self._extract_onnx_text(r)) for r in results]

    def _filter_by_alphabet(self, text):
        if not isinstance(text, str):
            text = str(text)
        if not self._cand_alphabet:
            return list(text)
        return [c for c in text if c in self._cand_alphabet]

