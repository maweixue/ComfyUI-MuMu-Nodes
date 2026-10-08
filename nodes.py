import base64
import io
import json
import socket
import ssl
import urllib.error
import time
import threading
import os
try:
    import winreg
except ImportError:
    winreg = None
import urllib.parse
import urllib.request
import http.client

import numpy as np


def _dec(s):
    return base64.b64decode(s).decode("utf-8")


try:
    import torch
except ImportError:
    torch = None

try:
    from PIL import Image
except ImportError:
    Image = None

_BASE_URL = "https://api.smile-ai-studio.com"

MSG_VIOLATION = "内容违规，可能涉嫌色情内容，请修改图片或提示词后重新生成"
MSG_SIZE_TOO_LARGE = "图片尺寸太大 AI无法承载 请将最长边分辨率调整到6500像素以下"
MSG_TIMEOUT = "任务超时，未链接成功，请重新生成"
MSG_GENERIC = "生成失败，请稍后重试"

_VIOLATION_KEYS = [
    "safety", "content_policy", "blocklist", "blocked", "policy",
    "violat", _dec("Im1vZGVyYXRpb24i"), "disallowed", "abuse",
    "sexual", "porn", "explicit", "inappropriate", "nudity",
    "违法", "违规", "内容", "色情", "审核", "不适宜",
]
_SIZE_KEYS = [
    "size", "resolution", "dimension", "too large", "too_large",
    "too big", "exceed", "larger than", "maximum", "limit",
    "像素", "尺寸", "分辨率", "6500", "超限", "过大",
]

RESOLUTION_CHOICES = ["1K", "2K", "4K"]
RESOLUTION_CHOICES_NO_1K = ["2K", "4K"]
ASPECT_CHOICES = ["auto", "1:1", "16:9", "9:16", "4:3", "3:4",
                  "3:2", "2:3", "5:4", "4:5", "21:9", "9:21",
                  "18:9", "16:10"]
BANANA_ASPECT_CHOICES = ["1:1", "16:9", "9:16", "4:3", "3:4",
                         "3:2", "2:3", "5:4", "4:5", "21:9"]
BANANA_ASPECT_CHOICES_EDIT = ["auto"] + BANANA_ASPECT_CHOICES

_SERIES_MODEL = {
    "banana_v2": {
        "1K": "gemini-3.1-flash-image-preview-1k",
        "2K": "gemini-3.1-flash-image-preview-2k",
        "4K": "gemini-3.1-flash-image-preview-4k",
    },
    "banana_pro": {
        "1K": "gemini-3-pro-image-preview-1k",
        "2K": "gemini-3-pro-image-preview-2k",
        "4K": "gemini-3-pro-image-preview-4k",
    },
    "banana_v21": {
        "4K": _dec("ImdlbWluaS1uYW5vLWJhbmFuYS0yLjEtNGsi"),
    },
}

_BR_BASE_URL = "https://api.bananarouter.com"
_BR_SIZES = {
    "1K": {"auto": "1024x1024", "1:1": "1024x1024", "16:9": "1536x864",
            "9:16": "864x1536", "4:3": "1360x1024", "3:4": "1024x1360",
            "3:2": "1536x1024", "2:3": "1024x1536", "5:4": "1280x1024",
            "4:5": "1024x1280", "21:9": "1536x672", "9:21": "672x1536",
            "18:9": "1536x768", "16:10": "1536x960"},
    "2K": {"auto": "2048x2048", "1:1": "2048x2048", "16:9": "2048x1152",
            "9:16": "1152x2048", "4:3": "2048x1536", "3:4": "1536x2048",
            "3:2": "2048x1360", "2:3": "1360x2048", "5:4": "2048x1632",
            "4:5": "1632x2048", "21:9": "2048x880", "9:21": "880x2048",
            "18:9": "2048x1024", "16:10": "2048x1280"},
    "4K": {"auto": "2880x2880", "1:1": "2880x2880", "16:9": "3840x2160",
            "9:16": "2160x3840", "4:3": "3312x2496", "3:4": "2496x3312",
            "3:2": "3520x2352", "2:3": "2352x3520", "5:4": "3216x2576",
            "4:5": "2576x3216", "21:9": "3840x1648", "9:21": "1648x3840",
            "18:9": "3840x1920", "16:10": "3648x2272"},
}


CATEGORY = "木木"

_IMAGE_KEYS = ["image_%d" % i for i in range(1, 11)]


def _resolve_model(series, resolution):
    table = _SERIES_MODEL[series]
    return table.get(resolution) or table.get("1K") or next(iter(table.values()))


def _extract_api_message(raw_bytes, fallback_text):
    try:
        err = json.loads(fallback_text)
        return err.get("error", {}).get("message", fallback_text)
    except Exception:
        return fallback_text


def _categorize_text(raw_text):
    t = (raw_text or "").lower()
    if any(k in t for k in _VIOLATION_KEYS):
        return MSG_VIOLATION
    if any(k in t for k in _SIZE_KEYS):
        return MSG_SIZE_TOO_LARGE
    return None


def _check_deps():
    if torch is None:
        raise RuntimeError("缺少必要依赖（torch）")
    if Image is None:
        raise RuntimeError("缺少必要依赖（Pillow）")


def _handle_http(e):
    raw = e.read()
    text = raw.decode("utf-8", "ignore")
    msg = _extract_api_message(raw, text)
    friendly = _categorize_text(msg) or _categorize_text(text)
    if friendly:
        raise RuntimeError(friendly)
    raise RuntimeError(MSG_GENERIC)


class _ReadError(RuntimeError):
    pass


def _read_all(resp, timeout):
    buf = io.BytesIO()
    end = time.time() + timeout
    while True:
        if time.time() > end:
            raise _ReadError(MSG_TIMEOUT)
        chunk = resp.read(65536)
        if not chunk:
            return buf.getvalue()
        buf.write(chunk)


def _request(url, data, headers, timeout, method="POST"):
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _read_all(resp, timeout)
    except urllib.error.HTTPError as e:
        _handle_http(e)
    except (socket.timeout, TimeoutError):
        raise RuntimeError(MSG_TIMEOUT)
    except http.client.IncompleteRead:
        raise _ReadError(MSG_TIMEOUT)
    except urllib.error.URLError as e:
        if isinstance(e.reason, ssl.SSLCertVerificationError):
            ctx = ssl._create_unverified_context()
            req2 = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req2, timeout=timeout, context=ctx) as resp:
                    return _read_all(resp, timeout)
            except urllib.error.HTTPError as e2:
                _handle_http(e2)
            except (socket.timeout, TimeoutError):
                raise RuntimeError(MSG_TIMEOUT)
            except http.client.IncompleteRead:
                raise _ReadError(MSG_TIMEOUT)
            except urllib.error.URLError:
                raise RuntimeError(MSG_TIMEOUT)
            except ConnectionError:
                raise RuntimeError(MSG_GENERIC)
            except http.client.HTTPException:
                raise RuntimeError(MSG_GENERIC)
        raise RuntimeError(MSG_TIMEOUT)
    except ConnectionError:
        raise RuntimeError(MSG_GENERIC)
    except http.client.HTTPException:
        raise RuntimeError(MSG_GENERIC)


def _compress(b):
    _check_deps()
    try:
        pil = Image.open(io.BytesIO(b))
        if pil.mode in ("RGBA", "LA", "P"):
            rgba = pil.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            pil = bg
        else:
            pil = pil.convert("RGB")
    except Exception:
        raise RuntimeError(MSG_GENERIC)
    pil.thumbnail((2048, 2048), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _bytes_to_img_tensor(data):
    _check_deps()
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise RuntimeError(MSG_GENERIC)
    arr = np.array(pil, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    return torch.from_numpy(arr)


def _img_tensor_to_bytes(img_tensor):
    _check_deps()
    arr = img_tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        frames = [arr[i] for i in range(arr.shape[0])]
    else:
        frames = [arr]
    out = []
    for fr in frames:
        a = (np.clip(fr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(a).convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def _parse_result(raw):
    try:
        result = json.loads(raw.decode("utf-8"))
    except Exception:
        raise RuntimeError(MSG_GENERIC)
    b64_list = []
    for cand in result.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            d = part.get("inlineData") or part.get("inline_data") or {}
            data = d.get("data")
            if data:
                b64_list.append(data)
    if not b64_list:
        raise RuntimeError(MSG_GENERIC)
    out = []
    for b64 in b64_list:
        try:
            out.append(_bytes_to_img_tensor(base64.b64decode(b64)))
        except Exception:
            raise RuntimeError(MSG_GENERIC)
    return out


def _build_variants(seed):
    variants = []
    if seed is not None and int(seed) >= 0:
        variants.append({"quality": "high", "seed": int(seed)})
        variants.append({"seed": int(seed)})
    variants.append({"quality": "high"})
    variants.append({})
    return variants


def _generate(model, prompt, aspect, api_key, seed=-1, timeout=180):
    url = _BASE_URL + "/v1beta/models/" + model + ":generateContent"
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    last = None
    for extra in _build_variants(seed):
        cfg = {"responseModalities": ["IMAGE"], "imageConfig": {_dec("ImFzcGVjdFJhdGlvIg=="): aspect}}
        cfg.update(extra)
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": cfg,
        }
        try:
            raw = _request(url, json.dumps(payload).encode("utf-8"), headers, timeout)
        except _ReadError:
            raise
        except RuntimeError as e:
            if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                raise
            last = e
            continue
        return _parse_result(raw)
    raise last if last else RuntimeError(MSG_GENERIC)


def _edits(model, prompt, aspect, ref_items, api_key, seed=-1, timeout=180):
    url = _BASE_URL + "/v1beta/models/" + model + ":generateContent"
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    last = None
    for extra in _build_variants(seed):
        parts = [{"text": prompt}]
        for n, b in ref_items:
            parts.append({"text": "图%d" % n})
            parts.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": base64.b64encode(b).decode("ascii"),
                }
            })
        cfg = {"responseModalities": ["IMAGE"], "imageConfig": {_dec("ImFzcGVjdFJhdGlvIg=="): aspect}}
        cfg.update(extra)
        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": cfg}
        try:
            raw = _request(url, json.dumps(payload).encode("utf-8"), headers, timeout)
        except _ReadError:
            raise
        except RuntimeError as e:
            if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                raise
            last = e
            continue
        return _parse_result(raw)
    raise last if last else RuntimeError(MSG_GENERIC)


def _pack_output(tensors):
    if len(tensors) == 1:
        return tensors[0]
    h = tensors[0].shape[1]
    w = tensors[0].shape[2]
    return torch.cat(
        [t if t.shape[1] == h and t.shape[2] == w else _resize(t, h, w) for t in tensors],
        dim=0,
    )


def _resize(t, h, w):
    return torch.nn.functional.interpolate(
        t.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)


def _seed_widget():
    return ("INT", {
        "default": -1,
        "min": -1,
        "max": 0xFFFFFFFFFFFFFFFF,
        "control_after_generate": True,
        "display": "随机种子",
        "tooltip": "固定后每次生成使用相同种子；随机则每次不同。模型是否完全复现固定结果取决于接口支持。",
    })


class _Text2Image:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图片",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    SERIES = "banana_v2"
    RESOLUTIONS = RESOLUTION_CHOICES
    ASPECTS = BANANA_ASPECT_CHOICES
    ASPECTS_EDIT = BANANA_ASPECT_CHOICES_EDIT
    ASPECT_DEFAULT = "1:1"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "display": "密钥"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "display": "提示词"}),
                "resolution": (cls.RESOLUTIONS, {"default": cls.RESOLUTIONS[0], "display": "分辨率"}),
                "aspect": (cls.ASPECTS, {"default": cls.ASPECT_DEFAULT, "display": "宽高比"}),
                "seed": _seed_widget(),
            },
        }

    def generate(self, api_key, prompt, resolution, aspect, seed=-1):
        model = _resolve_model(self.SERIES, resolution)
        tensors = _generate(model, prompt, aspect, api_key, seed=seed)
        return (_pack_output(tensors),)


class _Image2Image(_Text2Image):
    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        required = {k: v for k, v in base["required"].items() if k != "prompt"}
        required["aspect"] = (cls.ASPECTS_EDIT, {"default": cls.ASPECT_DEFAULT, "display": "宽高比"})
        optional = {
            "prompt": ("STRING", {"default": "", "multiline": True, "display": "提示词"}),
        }
        for i in range(1, 11):
            optional["image_%d" % i] = ("IMAGE", {"display": "图%d" % i})
        return {"required": required, "optional": optional}

    def generate(self, api_key, resolution, aspect, seed=-1, prompt="", **kwargs):
        model = _resolve_model(self.SERIES, resolution)
        imgs = [kwargs.get(k) for k in _IMAGE_KEYS]
        slots = {i + 1: imgs[i] for i in range(10) if imgs[i] is not None}
        if not slots:
            if aspect == "auto":
                aspect = "1:1"
            tensors = _generate(model, prompt, aspect, api_key, seed=seed)
        else:
            ref_items = [(n, _img_tensor_to_bytes(val)[0]) for n, val in sorted(slots.items())]
            tensors = _edits(model, prompt, aspect, ref_items, api_key, seed=seed)
        return (_pack_output(tensors),)


class NineWanLiPlugin2(_Text2Image):
    SERIES = "banana_v2"
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPlugin2_1(_Image2Image):
    SERIES = "banana_v2"
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPlugin3(_Text2Image):
    SERIES = "banana_pro"
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPlugin3_1(_Image2Image):
    SERIES = "banana_pro"
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPluginV2(_Text2Image):
    SERIES = "banana_v21"
    RESOLUTIONS = ["4K"]
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPluginV2_1(_Image2Image):
    SERIES = "banana_v21"
    RESOLUTIONS = ["4K"]
    DESCRIPTION = "测试节点 请勿使用"



import uuid


def _br_multipart(fields, files):
    boundary = "----br" + uuid.uuid4().hex
    buf = io.BytesIO()
    for k, v in fields.items():
        buf.write(("--" + boundary + "\r\nContent-Disposition: form-data; name=\"" + k + "\"\r\n\r\n" + str(v) + "\r\n").encode("utf-8"))
    for fname, fbytes in files:
        buf.write(("--" + boundary + "\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"" + fname + "\"\r\nContent-Type: image/png\r\n\r\n").encode("utf-8"))
        buf.write(fbytes)
        buf.write(b"\r\n")
    buf.write(("--" + boundary + "--\r\n").encode("utf-8"))
    return boundary, buf.getvalue()


def _sys_proxy():
    out = []
    if winreg is not None:
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Internet Settings")
            try:
                v = winreg.QueryValueEx(key, "ProxyServer")[0]
                if v:
                    out.append(v)
            except Exception:
                pass
            winreg.CloseKey(key)
        except Exception:
            pass
    for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        v = os.environ.get(k)
        if v and v not in out:
            out.append(v)
    return out


def _br_opener(proxy):
    if proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return None


def _br_open(url, timeout, proxy):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    op = _br_opener(proxy)
    if op:
        return op.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def _br_probe(url, proxy):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-262143"})
    op = _br_opener(proxy)
    t0 = time.time()
    if op:
        with op.open(req, timeout=3) as resp:
            n = len(resp.read())
    else:
        with urllib.request.urlopen(req, timeout=3) as resp:
            n = len(resp.read())
    return n, time.time() - t0


def _br_pick_channel(url):
    for proxy in _sys_proxy():
        try:
            _br_probe(url, proxy)
            return ("proxy", proxy)
        except Exception:
            continue
    try:
        n, dt = _br_probe(url, None)
        if dt > 0 and n / dt > 800 * 1024:
            return ("direct", None)
    except Exception:
        pass
    return ("direct8", None)


def _br_slice(url, start, end, timeout, proxy):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=%d-%d" % (start, end)})
    op = _br_opener(proxy)
    if op:
        with op.open(req, timeout=timeout) as resp:
            return _read_all(resp, timeout)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _read_all(resp, timeout)


def _br_download(url, timeout, n, proxy):
    try:
        with _br_open(url, timeout, proxy) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            if n == 1 or total <= 4194304 or resp.status != 200:
                return _read_all(resp, timeout)
    except (socket.timeout, TimeoutError):
        raise RuntimeError(MSG_TIMEOUT)
    except http.client.IncompleteRead:
        raise _ReadError(MSG_TIMEOUT)
    except Exception:
        raise RuntimeError(MSG_GENERIC)
    chunk = total // n
    ranges = [(i * chunk, (i + 1) * chunk - 1 if i < n - 1 else total - 1) for i in range(n)]
    results = [None] * n
    failed = []
    def _work(i):
        try:
            results[i] = _br_slice(url, ranges[i][0], ranges[i][1], timeout, proxy)
        except Exception:
            failed.append(i)
    threads = [threading.Thread(target=_work, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if not failed and all(r is not None for r in results):
        buf = io.BytesIO()
        for r in results:
            buf.write(r)
        return buf.getvalue()
    try:
        with _br_open(url, timeout, proxy) as resp:
            return _read_all(resp, timeout)
    except (socket.timeout, TimeoutError):
        raise RuntimeError(MSG_TIMEOUT)
    except http.client.IncompleteRead:
        raise _ReadError(MSG_TIMEOUT)
    except Exception:
        raise RuntimeError(MSG_GENERIC)


def _br_fetch_image(url, timeout):
    channel, proxy = _br_pick_channel(url)
    if channel == "proxy":
        try:
            return _br_download(url, timeout, 8, proxy)
        except RuntimeError:
            return _br_download(url, timeout, 8, None)
    if channel == "direct":
        return _br_download(url, timeout, 1, None)
    return _br_download(url, timeout, 8, None)


def _br_parse_result(raw):
    try:
        result = json.loads(raw.decode("utf-8"))
    except Exception:
        raise RuntimeError(MSG_GENERIC)
    data = result.get("data") or []
    item = {}
    for d in data:
        if d.get("url") or d.get("b64_json"):
            item = d
            break
    url = item.get("url")
    if url:
        last = None
        end = time.time() + 240
        for _ in range(3):
            remain = int(end - time.time())
            if remain <= 0:
                break
            try:
                img = _br_fetch_image(url, min(120, remain))
                return [_bytes_to_img_tensor(img)]
            except RuntimeError as e:
                last = e
        raise last if last else RuntimeError(MSG_GENERIC)
    b64 = item.get("b64_json")
    if not b64:
        raise RuntimeError(MSG_GENERIC)
    try:
        return [_bytes_to_img_tensor(base64.b64decode(b64))]
    except Exception:
        raise RuntimeError(MSG_GENERIC)


def _br_submit(url, body, headers, timeout):
    try:
        raw = _request(url, json.dumps(body).encode("utf-8"), headers, timeout)
    except _ReadError:
        raise
    except RuntimeError:
        raise
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        raise RuntimeError(MSG_GENERIC)


def _br_poll_task(tid, api_key, timeout):
    url = _BR_BASE_URL + "/v1/async-tasks/" + tid
    headers = {"Authorization": "Bearer " + api_key}
    deadline = time.time() + timeout
    while True:
        remain = deadline - time.time()
        if remain <= 0:
            raise RuntimeError(MSG_TIMEOUT)
        try:
            raw = _request(url, None, headers, min(15, max(5, remain)), method="GET")
        except Exception:
            time.sleep(5)
            continue
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            time.sleep(5)
            continue
        st = data.get("status")
        if st == "success":
            for it in (data.get("resultImages") or []):
                u = it.get("url")
                if u:
                    return u
            raise RuntimeError(MSG_GENERIC)
        if st in ("failed", "expired", _dec("ImNhbmNlbGxlZCI="), _dec("ImNhbmNlbGVkIg==")):
            msg = data.get("statusMessage") or data.get("error") or ""
            friendly = _categorize_text(msg)
            if friendly:
                raise RuntimeError(friendly)
            if msg:
                raise RuntimeError(MSG_GENERIC + "（" + str(msg).strip()[:100] + "）")
            raise RuntimeError(MSG_GENERIC)
        time.sleep(5)


def _br_cancel(tid, api_key):
    try:
        _request(_BR_BASE_URL + "/v1/async-tasks/" + tid, None,
                 {"Authorization": "Bearer " + api_key}, 10, method="DELETE")
    except Exception:
        pass


def _img_bytes_to_data_url(b):
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")


def _br_download_retry(url, timeout):
    last = None
    end = time.time() + timeout
    for _ in range(3):
        remain = int(end - time.time())
        if remain <= 0:
            break
        try:
            img = _br_fetch_image(url, min(120, remain))
            return [_bytes_to_img_tensor(img)]
        except RuntimeError as e:
            last = e
    raise last if last else RuntimeError(MSG_GENERIC)


def _br_build_variants(quality, seed):
    variants = []
    if seed is not None and int(seed) >= 0:
        v = {"seed": int(seed)}
        if quality:
            v["quality"] = quality
        variants.append(v)
        if quality:
            variants.append({"quality": quality})
    elif quality:
        variants.append({"quality": quality})
    variants.append({})
    return variants


def _img_generate(base_url, model, prompt, size, quality, api_key, seed=-1, timeout=240, response_format="url"):
    url = base_url + "/v1/images/generations"
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    last = None
    for extra in _br_build_variants(quality, seed):
        payload = {"model": model, "prompt": prompt, "size": size, _dec("Im1vZGVyYXRpb24i"): "low"}
        if response_format:
            payload["response_format"] = response_format
        payload.update(extra)
        try:
            raw = _request(url, json.dumps(payload).encode("utf-8"), headers, timeout)
        except _ReadError:
            raise
        except RuntimeError as e:
            if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                raise
            last = e
            continue
        return _br_parse_result(raw)
    raise last if last else RuntimeError(MSG_GENERIC)


def _br_generate(model, prompt, size, quality, api_key, seed=-1, timeout=360):
    url = _BR_BASE_URL + "/v1/images/generations/async"
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    last = None
    for extra in _br_build_variants(quality, seed):
        body = {"model": model, "prompt": prompt, "n": 1, "size": size, "output_format": "png",
                _dec("Im1vZGVyYXRpb24i"): "low"}
        body.update(extra)
        for attempt in range(2):
            try:
                data = _br_submit(url, body, headers, 30)
            except _ReadError:
                raise
            except RuntimeError as e:
                if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                    raise
                last = e
                break
            tid = data.get("taskID")
            if not tid:
                raise RuntimeError(MSG_GENERIC)
            try:
                img_url = _br_poll_task(tid, api_key, timeout)
            except RuntimeError as e:
                _br_cancel(tid, api_key)
                if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                    raise
                last = e
                continue
            return _br_download_retry(img_url, 240)
    raise last if last else RuntimeError(MSG_GENERIC)


def _sm_generate(model, prompt, size, api_key, seed=-1, timeout=240):
    return _img_generate(_BASE_URL, model, prompt, size, None, api_key, seed=seed, timeout=timeout, response_format=None)


def _img_edits(base_url, model, prompt, size, quality, ref_items, api_key, seed=-1, timeout=240, response_format="url"):
    url = base_url + "/v1/images/edits"
    last = None
    for variant in _br_build_variants(quality, seed):
        fields = {"model": model, "prompt": prompt, "size": size, _dec("Im1vZGVyYXRpb24i"): "low"}
        if response_format:
            fields["response_format"] = response_format
        fields.update({k: str(v) for k, v in variant.items()})
        files = [("img%d.png" % n, b) for n, b in ref_items]
        boundary, body = _br_multipart(fields, files)
        headers = {"Authorization": "Bearer " + api_key, "Content-Type": "multipart/form-data; boundary=" + boundary}
        try:
            raw = _request(url, body, headers, timeout)
        except _ReadError:
            raise
        except RuntimeError as e:
            if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                raise
            last = e
            continue
        return _br_parse_result(raw)
    raise last if last else RuntimeError(MSG_GENERIC)


def _br_edits(model, prompt, size, quality, ref_items, api_key, seed=-1, timeout=360):
    url = _BR_BASE_URL + "/v1/images/edits/async"
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    last = None
    for extra in _br_build_variants(quality, seed):
        body = {"model": model, "prompt": prompt, "n": 1, "size": size, "output_format": "png",
                _dec("Im1vZGVyYXRpb24i"): "low"}
        body["images"] = [_img_bytes_to_data_url(b) for n, b in ref_items]
        body.update(extra)
        for attempt in range(2):
            try:
                data = _br_submit(url, body, headers, 30)
            except _ReadError:
                raise
            except RuntimeError as e:
                if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                    raise
                last = e
                break
            tid = data.get("taskID")
            if not tid:
                raise RuntimeError(MSG_GENERIC)
            try:
                img_url = _br_poll_task(tid, api_key, timeout)
            except RuntimeError as e:
                _br_cancel(tid, api_key)
                if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                    raise
                last = e
                continue
            return _br_download_retry(img_url, 240)
    raise last if last else RuntimeError(MSG_GENERIC)


def _sm_edits(model, prompt, size, ref_items, api_key, seed=-1, timeout=240):
    return _img_edits(_BASE_URL, model, prompt, size, None, ref_items, api_key, seed=seed, timeout=timeout, response_format=None)


_MJ_ASPECTS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21", "4:5", "5:4"]
_MJ_ASPECTS_EDIT = ["auto"] + _MJ_ASPECTS


def _mj_nearest(w, h):
    w, h = int(w), int(h)
    if h <= 0:
        return "1:1"
    best = _MJ_ASPECTS[0]
    bd = 1e18
    r = w / float(h)
    for a in _MJ_ASPECTS:
        aw, ah = map(int, a.split(":"))
        d = abs(r - aw / float(ah))
        if d < bd:
            bd = d
            best = a
    return best


def _int_or(v, d):
    try:
        if v in (None, ""):
            return d
        return int(v)
    except Exception:
        return d


def _mj_prompt(prompt, aspect, stylize, seed, chaos=None, quality="4", no="", iw=None):
    p = (prompt or "")[:900]
    if aspect and aspect != "auto":
        p += " --ar " + aspect
    if stylize is not None and str(stylize).strip() != "不设置":
        p += " --stylize " + str(_int_or(stylize, 100))
    if seed is not None and str(seed).strip() and str(seed).strip() != "不设置":
        sd = _int_or(seed, -1)
        if sd >= 0:
            sd &= 0xFFFFFFFF
            p += " --seed " + str(sd)
    if chaos is not None and str(chaos).strip() and str(chaos).strip() != "不设置":
        c = _int_or(chaos, 0)
        if c > 0:
            p += " --chaos " + str(c)
    if quality and quality != "不设置":
        p += " --q " + quality
    if no and str(no).strip() and str(no).strip() != "不设置":
        p += " --no " + str(no).strip()
    if iw is not None and str(iw).strip() and str(iw).strip() != "不设置":
        try:
            f = float(iw)
            if f >= 0:
                p += " --iw " + str(f)
        except Exception:
            pass
    return p


def _mj_submit(prompt, api_key, ref_b64=None):
    url = _BR_BASE_URL + "/mj/submit/imagine"
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    body = {"model": "midjourney", "prompt": prompt}
    if ref_b64:
        body["base64Array"] = ref_b64
    try:
        raw = _request(url, json.dumps(body).encode("utf-8"), headers, 30)
    except _ReadError:
        raise
    except RuntimeError:
        raise
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception:
        raise RuntimeError(MSG_GENERIC)
    tid = obj.get("result")
    if not tid:
        raise RuntimeError(MSG_GENERIC)
    return tid


def _mj_poll(tid, api_key, timeout):
    url = _BR_BASE_URL + "/mj/task/" + tid + "/fetch"
    headers = {"Authorization": "Bearer " + api_key}
    deadline = time.time() + timeout
    while True:
        remain = deadline - time.time()
        if remain <= 0:
            raise RuntimeError(MSG_TIMEOUT)
        try:
            raw = _request(url, None, headers, min(15, max(5, remain)), method="GET")
        except Exception:
            time.sleep(5)
            continue
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            time.sleep(5)
            continue
        st = data.get("status")
        if st == "SUCCESS":
            urls = [it.get("url") for it in (data.get("imageUrls") or []) if it.get("url")]
            if urls:
                return urls
            raise RuntimeError(MSG_GENERIC)
        if st in ("FAILURE", "EXPIRED", "CANCELLED", "CANCELED"):
            msg = data.get("failReason") or data.get("error") or ""
            friendly = _categorize_text(msg)
            if friendly:
                raise RuntimeError(friendly)
            if msg:
                raise RuntimeError(MSG_GENERIC + "（" + str(msg).strip()[:100] + "）")
            raise RuntimeError(MSG_GENERIC)
        time.sleep(5)


def _mj_fetch_images(urls, timeout):
    outs = [None] * len(urls)
    def _w(i):
        outs[i] = _br_fetch_image(urls[i], timeout)
    threads = [threading.Thread(target=_w, args=(i,)) for i in range(len(urls))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if any(o is None for o in outs):
        raise RuntimeError(MSG_GENERIC)
    return [_bytes_to_img_tensor(o) for o in outs]


def _mj_generate(prompt, api_key, timeout=360, max_count=0):
    last = None
    for attempt in range(2):
        try:
            tid = _mj_submit(prompt, api_key)
        except _ReadError:
            raise
        except RuntimeError as e:
            if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                raise
            last = e
            break
        try:
            urls = _mj_poll(tid, api_key, timeout)
        except _ReadError:
            raise
        except RuntimeError as e:
            if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                raise
            last = e
            continue
        if max_count and max_count > 0:
            urls = urls[:max_count]
        try:
            return _mj_fetch_images(urls, 240)
        except _ReadError:
            raise
        except RuntimeError as e:
            last = e
            continue
    raise last if last else RuntimeError(MSG_GENERIC)


def _mj_edits(prompt, ref_items, api_key, timeout=360, max_count=0):
    ref_b64 = [base64.b64encode(b).decode("ascii") for n, b in ref_items]
    last = None
    for attempt in range(2):
        try:
            tid = _mj_submit(prompt, api_key, ref_b64=ref_b64)
        except _ReadError:
            raise
        except RuntimeError as e:
            if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                raise
            last = e
            break
        try:
            urls = _mj_poll(tid, api_key, timeout)
        except _ReadError:
            raise
        except RuntimeError as e:
            if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                raise
            last = e
            continue
        if max_count and max_count > 0:
            urls = urls[:max_count]
        try:
            return _mj_fetch_images(urls, 240)
        except _ReadError:
            raise
        except RuntimeError as e:
            last = e
            continue
    raise last if last else RuntimeError(MSG_GENERIC)


def _ref_size(t):
    s = list(t.shape)
    if len(s) == 4:
        h, w = int(s[1]), int(s[2])
    elif len(s) == 3:
        h, w = int(s[0]), int(s[1])
    else:
        return 1024, 1024
    return max(16, w), max(16, h)


def _auto_size(w, h):
    w, h = int(w), int(h)
    ratio = w / float(h)
    if ratio >= 1.0:
        W, H = 3840, int(3840 / ratio)
    else:
        W, H = int(3840 * ratio), 3840
    if W / float(H) > 3.0:
        if W >= H:
            H = int(W / 3.0)
        else:
            W = int(H / 3.0)
    pix = W * H
    if pix > 8294400:
        s = (8294400.0 / pix) ** 0.5
        W, H = int(W * s), int(H * s)
    elif pix < 655360:
        s = (655360.0 / pix) ** 0.5
        W, H = int(W * s), int(H * s)
    W = max(16, (W // 16) * 16)
    H = max(16, (H // 16) * 16)
    pix = W * H
    if pix > 8294400:
        s = (8294400.0 / pix) ** 0.5
        W = max(16, (int(W * s) // 16) * 16)
        H = max(16, (int(H * s) // 16) * 16)
    if pix < 655360:
        s = (655360.0 / pix) ** 0.5
        W = max(16, (int(W * s) // 16) * 16 + 16)
        H = max(16, (int(H * s) // 16) * 16 + 16)
    return "%dx%d" % (W, H)


class _BRText2Image:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图片",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    SIZES = _BR_SIZES
    MODEL = None
    MODEL_VERSIONS = {}
    RESOLUTIONS = RESOLUTION_CHOICES
    QUALITY = False
    QUALITY_FIXED = None
    QUALITY_MAP = {"低": "low", "中": "medium", "高": "high", "超高": "xhigh", "最高": "max"}
    QUALITY_DEFAULT = "高"

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "api_key": ("STRING", {"default": "", "multiline": False, "display": "密钥"}),
            "prompt": ("STRING", {"default": "", "multiline": True, "display": "提示词"}),
        }
        if cls.MODEL_VERSIONS:
            required["model_version"] = (list(cls.MODEL_VERSIONS), {"default": list(cls.MODEL_VERSIONS)[0], "display": "模型版本"})
        if cls.QUALITY:
            required["quality"] = (list(cls.QUALITY_MAP), {"default": cls.QUALITY_DEFAULT, "display": "画质"})
        required["resolution"] = (cls.RESOLUTIONS, {"default": cls.RESOLUTIONS[0], "display": "分辨率"})
        required["aspect"] = (ASPECT_CHOICES, {"default": "1:1", "display": "宽高比"})
        required["seed"] = _seed_widget()
        return {"required": required}

    def _resolve_cfg(self, model_version, quality, resolution):
        model = self.MODEL_VERSIONS.get(model_version, self.MODEL) if self.MODEL_VERSIONS else self.MODEL
        if self.QUALITY_FIXED:
            q = self.QUALITY_FIXED
        elif self.QUALITY and quality:
            q = self.QUALITY_MAP.get(quality)
        else:
            q = None
        return model, q, resolution

    def generate(self, api_key, prompt, resolution, aspect, seed=-1, model_version=None, quality=None):
        model, q, res = self._resolve_cfg(model_version, quality, resolution)
        size = self.SIZES[res][aspect]
        tensors = _br_generate(model, prompt, size, q, api_key, seed=seed)
        return (_pack_output(tensors),)


class _BRImage2Image(_BRText2Image):
    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        required = {k: v for k, v in base["required"].items() if k != "prompt"}
        optional = {
            "prompt": ("STRING", {"default": "", "multiline": True, "display": "提示词"}),
        }
        for i in range(1, 11):
            optional["image_%d" % i] = ("IMAGE", {"display": "图%d" % i})
        return {"required": required, "optional": optional}

    def generate(self, api_key, resolution, aspect, seed=-1, model_version=None, quality=None, prompt="", **kwargs):
        model, q, res = self._resolve_cfg(model_version, quality, resolution)
        imgs = [kwargs.get(k) for k in _IMAGE_KEYS]
        slots = {i + 1: imgs[i] for i in range(10) if imgs[i] is not None}
        if not slots:
            if aspect == "auto":
                aspect = "1:1"
            size = self.SIZES[res][aspect]
            tensors = _br_generate(model, prompt, size, q, api_key, seed=seed)
        else:
            if aspect == "auto":
                w, h = _ref_size(next(iter(slots.values())))
                size = _auto_size(w, h)
            else:
                size = self.SIZES[res][aspect]
            ref_items = [(n, _img_tensor_to_bytes(val)[0]) for n, val in sorted(slots.items())]
            tensors = _br_edits(model, prompt, size, q, ref_items, api_key, seed=seed)
        return (_pack_output(tensors),)


class NineWanLiPlugin5(_BRText2Image):
    MODEL = _dec("ImdwdC1pbWFnZS0yLjUtZmxhcmUi")
    RESOLUTIONS = RESOLUTION_CHOICES_NO_1K
    QUALITY_FIXED = "max"
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPlugin5_1(_BRImage2Image):
    MODEL = _dec("ImdwdC1pbWFnZS0yLjUtZmxhcmUi")
    RESOLUTIONS = RESOLUTION_CHOICES_NO_1K
    QUALITY_FIXED = "max"
    DESCRIPTION = "测试节点 请勿使用"


class _SMText2Image:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图片",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    MODEL = None
    RESOLUTIONS = RESOLUTION_CHOICES

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "display": "密钥"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "display": "提示词"}),
                "resolution": (cls.RESOLUTIONS, {"default": cls.RESOLUTIONS[0], "display": "分辨率"}),
                "aspect": (ASPECT_CHOICES, {"default": "1:1", "display": "宽高比"}),
                "seed": _seed_widget(),
            },
        }

    def generate(self, api_key, prompt, resolution, aspect, seed=-1):
        size = _BR_SIZES[resolution][aspect]
        tensors = _sm_generate(self.MODEL, prompt, size, api_key, seed=seed)
        return (_pack_output(tensors),)


class _SMImage2Image(_SMText2Image):
    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        required = {k: v for k, v in base["required"].items() if k != "prompt"}
        optional = {
            "prompt": ("STRING", {"default": "", "multiline": True, "display": "提示词"}),
        }
        for i in range(1, 11):
            optional["image_%d" % i] = ("IMAGE", {"display": "图%d" % i})
        return {"required": required, "optional": optional}

    def generate(self, api_key, resolution, aspect, seed=-1, prompt="", **kwargs):
        imgs = [kwargs.get(k) for k in _IMAGE_KEYS]
        slots = {i + 1: imgs[i] for i in range(10) if imgs[i] is not None}
        if not slots:
            if aspect == "auto":
                aspect = "1:1"
            size = _BR_SIZES[resolution][aspect]
            tensors = _sm_generate(self.MODEL, prompt, size, api_key, seed=seed)
        else:
            if aspect == "auto":
                w, h = _ref_size(next(iter(slots.values())))
                size = _auto_size(w, h)
            else:
                size = _BR_SIZES[resolution][aspect]
            ref_items = [(n, _img_tensor_to_bytes(val)[0]) for n, val in sorted(slots.items())]
            tensors = _sm_edits(self.MODEL, prompt, size, ref_items, api_key, seed=seed)
        return (_pack_output(tensors),)


class NineWanLiPlugin6(_SMText2Image):
    MODEL = _dec("ImdwdC1pbWFnZS0yIg==")
    RESOLUTIONS = RESOLUTION_CHOICES
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPlugin6_1(_SMImage2Image):
    MODEL = _dec("ImdwdC1pbWFnZS0yIg==")
    RESOLUTIONS = RESOLUTION_CHOICES
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPlugin7(_SMText2Image):
    MODEL = _dec("ImdwdC1pbWFnZS0yLjUtZmxhcmUi")
    RESOLUTIONS = RESOLUTION_CHOICES
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPlugin7_1(_SMImage2Image):
    MODEL = _dec("ImdwdC1pbWFnZS0yLjUtZmxhcmUi")
    RESOLUTIONS = RESOLUTION_CHOICES
    DESCRIPTION = "测试节点 请勿使用"


class _MJText2Image:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图片",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    ASPECTS = _MJ_ASPECTS
    ASPECTS_EDIT = _MJ_ASPECTS_EDIT
    ASPECT_DEFAULT = "1:1"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "display": "密钥"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "display": "提示词"}),
                "aspect": (cls.ASPECTS, {"default": cls.ASPECT_DEFAULT, "display": "宽高比"}),
                "chaos": ("STRING", {"default": "0", "display": "混乱度", "tooltip": "随机性强度，0 最稳定"}),
                "no": ("STRING", {"default": "", "multiline": True, "display": "负面提示词", "tooltip": "不希望出现在图中的内容，用逗号分隔"}),
                "stylize": ("STRING", {"default": "100", "display": "风格化"}),
                "seed": _seed_widget(),
            },
        }

    def generate(self, api_key, prompt, aspect, chaos, no, stylize, seed=-1):
        p = _mj_prompt(prompt, aspect, stylize, seed, chaos=chaos, no=no)
        tensors = _mj_generate(p, api_key)
        return (_pack_output(tensors),)


class _MJImage2Image(_MJText2Image):
    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        required = {}
        for k, v in base["required"].items():
            if k != "prompt":
                required[k] = v
        required["aspect"] = (cls.ASPECTS_EDIT, {"default": cls.ASPECT_DEFAULT, "display": "宽高比"})
        required["iw"] = ("STRING", {"default": "-1", "display": "参考权重", "tooltip": "参考图影响强度，0 完全忽略参考图，3 最大影响"})
        optional = {
            "prompt": ("STRING", {"default": "", "multiline": True, "display": "提示词"}),
        }
        for i in range(1, 11):
            optional["image_%d" % i] = ("IMAGE", {"display": "图%d" % i})
        return {"required": required, "optional": optional}

    def generate(self, api_key, aspect, chaos, no, stylize, seed=-1, iw=-1.0, prompt="", **kwargs):
        imgs = [kwargs.get(k) for k in _IMAGE_KEYS]
        slots = {i + 1: imgs[i] for i in range(10) if imgs[i] is not None}
        if not slots:
            if aspect == "auto":
                aspect = "1:1"
            p = _mj_prompt(prompt, aspect, stylize, seed, chaos=chaos, no=no)
            tensors = _mj_generate(p, api_key)
        else:
            w, h = _ref_size(next(iter(slots.values())))
            if aspect == "auto":
                aspect = _mj_nearest(w, h)
            p = _mj_prompt(prompt, aspect, stylize, seed, chaos=chaos, no=no, iw=iw)
            ref_items = [(n, _img_tensor_to_bytes(val)[0]) for n, val in sorted(slots.items())]
            tensors = _mj_edits(p, ref_items, api_key)
        return (_pack_output(tensors),)


class NineWanLiPluginV8(_MJText2Image):
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPluginV8_1(_MJImage2Image):
    DESCRIPTION = "测试节点 请勿使用"


class NineWanLiPlugin5_6:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "测试节点 请勿使用"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "prompt": ("STRING", {"default": "", "multiline": True, "display": "提问"}),
        }
        for i in range(1, 11):
            optional["image_%d" % i] = ("IMAGE", {"display": "图%d" % i})
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "display": "密钥"}),
            },
            "optional": optional,
        }

    def generate(self, api_key, prompt="", **kwargs):
        imgs = [kwargs.get(k) for k in _IMAGE_KEYS]
        slots = {i + 1: imgs[i] for i in range(10) if imgs[i] is not None}
        if slots:
            parts = [{"type": "text", "text": prompt or ""}]
            for n, val in sorted(slots.items()):
                b = _img_tensor_to_bytes(val)[0]
                parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b).decode("ascii")}})
            content = parts
        else:
            content = prompt or ""
        url = _BR_BASE_URL + "/v1/chat/completions"
        headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
        body = {
            "model": "gpt-5.6-luna",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.6,
            "max_completion_tokens": 4096,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }
        last = None
        for attempt in range(2):
            try:
                raw = _request(url, json.dumps(body).encode("utf-8"), headers, 180)
            except _ReadError:
                raise
            except RuntimeError as e:
                if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                    raise
                last = e
                continue
            try:
                obj = json.loads(raw.decode("utf-8"))
                text = obj["choices"][0]["message"]["content"]
                return (text or "",)
            except Exception:
                raise RuntimeError(MSG_GENERIC)
        raise last if last else RuntimeError(MSG_GENERIC)


NODE_CLASS_MAPPINGS = {
    "NineWanLiPlugin2": NineWanLiPlugin2,
    "NineWanLiPlugin2_1": NineWanLiPlugin2_1,
    "NineWanLiPlugin3": NineWanLiPlugin3,
    "NineWanLiPlugin3_1": NineWanLiPlugin3_1,
    "NineWanLiPlugin5": NineWanLiPlugin5,
    "NineWanLiPlugin5_1": NineWanLiPlugin5_1,
    "NineWanLiPlugin6": NineWanLiPlugin6,
    "NineWanLiPlugin6_1": NineWanLiPlugin6_1,
    "NineWanLiPlugin7": NineWanLiPlugin7,
    "NineWanLiPlugin7_1": NineWanLiPlugin7_1,
    "NineWanLiPluginV2": NineWanLiPluginV2,
    "NineWanLiPluginV2_1": NineWanLiPluginV2_1,
    "NineWanLiPluginV8": NineWanLiPluginV8,
    "NineWanLiPluginV8_1": NineWanLiPluginV8_1,
    "NineWanLiPlugin5_6": NineWanLiPlugin5_6,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NineWanLiPlugin2": "木木2",
    "NineWanLiPlugin2_1": "木木2.1",
    "NineWanLiPlugin3": "木木3",
    "NineWanLiPlugin3_1": "木木3.1",
    "NineWanLiPlugin5": "南南5",
    "NineWanLiPlugin5_1": "南南5.1",
    "NineWanLiPlugin6": "木木4-备用",
    "NineWanLiPlugin6_1": "木木4.1-备用",
    "NineWanLiPlugin7": "木木5-备用",
    "NineWanLiPlugin7_1": "木木5.1-备用",
    "NineWanLiPluginV2": "木木V2",
    "NineWanLiPluginV2_1": "木木V2.1",
    "NineWanLiPluginV8": "南南V8",
    "NineWanLiPluginV8_1": "南南V8.1",
    "NineWanLiPlugin5_6": "南南5.6",
}

for _cls in [NineWanLiPlugin2, NineWanLiPlugin2_1, NineWanLiPlugin3, NineWanLiPlugin3_1,
             NineWanLiPlugin5, NineWanLiPlugin5_1, NineWanLiPlugin6, NineWanLiPlugin6_1,
             NineWanLiPlugin7, NineWanLiPlugin7_1, NineWanLiPluginV2, NineWanLiPluginV2_1,
             NineWanLiPluginV8, NineWanLiPluginV8_1, NineWanLiPlugin5_6]:
    _cls.NODE_NAME = NODE_DISPLAY_NAME_MAPPINGS.get(_cls.__name__, _cls.__name__)

_GR_BASE_URL = _dec("Imh0dHBzOi8vZ3JzYWkuZGFra2EuY29tLmNuIg==")

_GR_ASPECTS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9", "9:21"]
_GR_ASPECTS_WIDE = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9", "9:21", "4:1", "1:4", "8:1", "1:8"]
_GR_ASPECTS_EDIT = ["auto"] + _GR_ASPECTS
_GR_ASPECTS_WIDE_EDIT = ["auto"] + _GR_ASPECTS_WIDE

_GR_GPT_STD = {
    "1:1": "1024x1024", "16:9": "1672x941", "9:16": "941x1672",
    "4:3": "1443x1090", "3:4": "1090x1443", "3:2": "1536x1024",
    "2:3": "1024x1536", "5:4": "1408x1120", "4:5": "1120x1408",
    "21:9": "1920x832", "9:21": "832x1920", "2:1": "1792x896", "1:2": "896x1792",
}

_GR_GPT_VIP = {
    "1K": {
        "1:1": "1024x1024", "16:9": "1280x720", "9:16": "720x1280",
        "4:3": "1152x864", "3:4": "864x1152", "3:2": "1536x1024", "2:3": "1024x1536",
        "5:4": "1120x896", "4:5": "896x1120", "21:9": "1456x624", "9:21": "624x1456",
        "2:1": "1536x768", "1:2": "768x1536",
    },
    "2K": {
        "1:1": "2048x2048", "16:9": "2048x1152", "9:16": "1152x2048",
        "4:3": "2304x1728", "3:4": "1728x2304", "3:2": "2048x1360", "2:3": "1360x2048",
        "5:4": "2240x1792", "4:5": "1792x2240", "21:9": "2912x1248", "9:21": "1248x2912",
        "2:1": "3072x1536", "1:2": "1536x3072", "3:1": "2048x688", "1:3": "688x2048",
    },
    "4K": {
        "1:1": "2880x2880", "16:9": "3840x2160", "9:16": "2160x3840",
        "4:3": "3264x2448", "3:4": "2448x3264", "3:2": "3504x2336", "2:3": "2336x3504",
        "5:4": "3200x2560", "4:5": "2560x3200", "21:9": "3840x1648", "9:21": "1648x3840",
        "2:1": "3840x1920", "1:2": "1920x3840", "3:1": "3840x1280", "1:3": "1280x3840",
    },
}


def _gr_req(method, path, api_key, data=None, params=None, timeout=120):
    url = _GR_BASE_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body,
                                 headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "ignore")
        friendly = _categorize_text(text)
        if friendly:
            raise RuntimeError(friendly)
        raise RuntimeError(MSG_GENERIC)
    except (socket.timeout, TimeoutError):
        raise RuntimeError(MSG_TIMEOUT)
    except http.client.IncompleteRead:
        raise _ReadError(MSG_TIMEOUT)
    except urllib.error.URLError:
        raise RuntimeError(MSG_TIMEOUT)
    except ConnectionError:
        raise RuntimeError(MSG_GENERIC)
    except http.client.HTTPException:
        raise RuntimeError(MSG_GENERIC)
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        raise RuntimeError(MSG_GENERIC)


def _gr_submit(model, prompt, api_key, ar=None, image_size=None, quality=None, images=None, seed=None, timeout=60):
    payload = {"model": model, "prompt": prompt, _dec("InJlcGx5VHlwZSI="): _dec("ImFzeW5jIg==")}
    if ar:
        payload[_dec("ImFzcGVjdFJhdGlvIg==")] = ar
    if image_size:
        payload[_dec("ImltYWdlU2l6ZSI=")] = image_size
    if quality:
        payload["quality"] = quality
    if images:
        payload["images"] = images
    if seed is not None and int(seed) >= 0:
        payload["seed"] = int(seed)
    resp = _gr_req("POST", _dec("Ii92MS9hcGkvZ2VuZXJhdGUi"), api_key, payload, timeout=timeout)
    tid = resp.get("id")
    if not tid:
        raise RuntimeError(MSG_GENERIC)
    return tid


def _gr_poll(tid, api_key, timeout):
    deadline = time.time() + timeout
    while True:
        remain = deadline - time.time()
        if remain <= 0:
            raise RuntimeError(MSG_TIMEOUT)
        try:
            data = _gr_req("GET", _dec("Ii92MS9hcGkvcmVzdWx0Ig=="), api_key, params={"id": tid}, timeout=min(30, max(10, remain)))
        except Exception:
            time.sleep(3)
            continue
        st = data.get("status", "")
        if st == _dec("InN1Y2NlZWRlZCI="):
            urls = [x.get("url") for x in (data.get(_dec("InJlc3VsdHMi")) or []) if isinstance(x, dict) and x.get("url")]
            if not urls:
                raise RuntimeError(MSG_GENERIC)
            return urls
        if st in ("failed", _dec("InZpb2xhdGlvbiI="), _dec("ImNhbmNlbGxlZCI="), _dec("ImNhbmNlbGVkIg==")):
            reason = str(data.get(_dec("ImZhaWx1cmVfcmVhc29uIg==")) or "")
            if _dec("Im1vZGVyYXRpb24i") in reason or st == _dec("InZpb2xhdGlvbiI="):
                raise RuntimeError(MSG_VIOLATION)
            msg = data.get("error") or ""
            if msg:
                raise RuntimeError(MSG_GENERIC + "（" + str(msg).strip()[:100] + "）")
            raise RuntimeError(MSG_GENERIC)
        time.sleep(2)


def _gr_fetch_image(url, timeout):
    last = None
    try:
        return _br_download(url, timeout, 1, None)
    except RuntimeError as e:
        last = e
    try:
        return _br_download(url, timeout, 8, None)
    except RuntimeError as e:
        last = e
    for proxy in _sys_proxy():
        try:
            return _br_download(url, timeout, 8, proxy)
        except RuntimeError as e:
            last = e
    raise last if last else RuntimeError(MSG_GENERIC)


def _gr_fetch_images(urls, timeout):
    outs = [None] * len(urls)

    def _w(i):
        outs[i] = _gr_fetch_image(urls[i], timeout)

    threads = [threading.Thread(target=_w, args=(i,)) for i in range(len(urls))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if any(o is None for o in outs):
        raise RuntimeError(MSG_GENERIC)
    return [_bytes_to_img_tensor(o) for o in outs]


def _gr_generate(model, prompt, api_key, ar=None, image_size=None, quality=None, seed=None, timeout=360):
    last = None
    for attempt in range(2):
        try:
            tid = _gr_submit(model, prompt, api_key, ar, image_size, quality, None, seed)
        except _ReadError:
            raise
        except RuntimeError as e:
            if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                raise
            last = e
            break
        try:
            urls = _gr_poll(tid, api_key, timeout)
        except _ReadError:
            raise
        except RuntimeError as e:
            if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                raise
            last = e
            continue
        try:
            return _gr_fetch_images(urls, 240)
        except _ReadError:
            raise
        except RuntimeError as e:
            last = e
            continue
    raise last if last else RuntimeError(MSG_GENERIC)


def _gr_edits(model, prompt, api_key, ar, image_size, quality, ref_items, seed=None, timeout=360):
    images = [base64.b64encode(b).decode("ascii") for n, b in ref_items]
    last = None
    for attempt in range(2):
        try:
            tid = _gr_submit(model, prompt, api_key, ar, image_size, quality, images, seed)
        except _ReadError:
            raise
        except RuntimeError as e:
            if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                raise
            last = e
            break
        try:
            urls = _gr_poll(tid, api_key, timeout)
        except _ReadError:
            raise
        except RuntimeError as e:
            if str(e) in (MSG_VIOLATION, MSG_SIZE_TOO_LARGE):
                raise
            last = e
            continue
        try:
            return _gr_fetch_images(urls, 240)
        except _ReadError:
            raise
        except RuntimeError as e:
            last = e
            continue
    raise last if last else RuntimeError(MSG_GENERIC)


def _gr_nearest_ratio(w, h, choices):
    if h <= 0:
        return choices[0]
    best = choices[0]
    bd = 1e18
    r = w / float(h)
    for a in choices:
        if ":" not in a:
            continue
        aw, ah = map(int, a.split(":"))
        d = abs(r - aw / float(ah))
        if d < bd:
            bd = d
            best = a
    return best


def _gr_nearest_pixel(w, h, table):
    if h <= 0:
        return next(iter(table.values()))
    best = None
    bd = 1e18
    r = w / float(h)
    for label, px in table.items():
        if ":" not in label:
            continue
        aw, ah = map(int, label.split(":"))
        d = abs(r - aw / float(ah))
        if d < bd:
            bd = d
            best = px
    return best or next(iter(table.values()))


class _GRText2Image:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图片",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    MODEL = None
    RESOLUTIONS = RESOLUTION_CHOICES
    ASPECTS = _GR_ASPECTS
    ASPECTS_EDIT = _GR_ASPECTS_EDIT
    GPT_PIXELS = None
    QUALITY_FIXED = None
    DESCRIPTION = "测试节点 请勿使用"

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "api_key": ("STRING", {"default": "", "multiline": False, "display": "密钥"}),
            "prompt": ("STRING", {"default": "", "multiline": True, "display": "提示词"}),
        }
        if len(cls.RESOLUTIONS) > 1:
            required["resolution"] = (cls.RESOLUTIONS, {"default": cls.RESOLUTIONS[0], "display": "分辨率"})
        required["aspect"] = (cls.ASPECTS, {"default": "1:1", "display": "宽高比"})
        required["seed"] = _seed_widget()
        return {"required": required}

    def _build(self, resolution, aspect):
        if self.GPT_PIXELS:
            res = resolution or "1K"
            table = self.GPT_PIXELS.get(res) or next(iter(self.GPT_PIXELS.values()))
            ar = table.get(aspect)
            if not ar:
                ar = next(iter(table.values()))
            return None, ar
        return resolution, aspect

    def generate(self, api_key, prompt, aspect, seed=-1, resolution=None):
        image_size, ar = self._build(resolution, aspect)
        tensors = _gr_generate(self.MODEL, prompt, api_key, ar, image_size, self.QUALITY_FIXED, seed)
        return (_pack_output(tensors),)


class _GRImage2Image(_GRText2Image):
    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        required = {k: v for k, v in base["required"].items() if k not in ("prompt", "aspect")}
        required["aspect"] = (cls.ASPECTS_EDIT, {"default": "auto", "display": "宽高比"})
        optional = {
            "prompt": ("STRING", {"default": "", "multiline": True, "display": "提示词"}),
        }
        for i in range(1, 11):
            optional["image_%d" % i] = ("IMAGE", {"display": "图%d" % i})
        return {"required": required, "optional": optional}

    def generate(self, api_key, aspect, seed=-1, prompt="", resolution=None, **kwargs):
        imgs = [kwargs.get(k) for k in _IMAGE_KEYS]
        slots = {i + 1: imgs[i] for i in range(10) if imgs[i] is not None}
        image_size, ar = self._build(resolution, aspect)
        if not slots:
            if aspect == "auto":
                if self.GPT_PIXELS:
                    table = self.GPT_PIXELS.get(image_size or "1K") or next(iter(self.GPT_PIXELS.values()))
                    ar = next(iter(table.values()))
                else:
                    ar = "1:1"
            tensors = _gr_generate(self.MODEL, prompt, api_key, ar, image_size, self.QUALITY_FIXED, seed)
        else:
            w, h = _ref_size(next(iter(slots.values())))
            if aspect == "auto":
                if self.GPT_PIXELS:
                    table = self.GPT_PIXELS.get(image_size or "1K") or next(iter(self.GPT_PIXELS.values()))
                    ar = _gr_nearest_pixel(w, h, table)
                else:
                    ar = _gr_nearest_ratio(w, h, self.ASPECTS)
            ref_items = [(n, _img_tensor_to_bytes(val)[0]) for n, val in sorted(slots.items())]
            tensors = _gr_edits(self.MODEL, prompt, api_key, ar, image_size, self.QUALITY_FIXED, ref_items, seed)
        return (_pack_output(tensors),)


class NineWanLiPluginM1(_GRText2Image):
    MODEL = _dec("Im5hbm8tYmFuYW5hLTItbGl0ZSI=")
    RESOLUTIONS = RESOLUTION_CHOICES
    ASPECTS = _GR_ASPECTS


class NineWanLiPluginM1_1(_GRImage2Image):
    MODEL = _dec("Im5hbm8tYmFuYW5hLTItbGl0ZSI=")
    RESOLUTIONS = RESOLUTION_CHOICES
    ASPECTS = _GR_ASPECTS


class NineWanLiPluginM3(_GRText2Image):
    MODEL = _dec("Im5hbm8tYmFuYW5hLXBybyI=")
    RESOLUTIONS = RESOLUTION_CHOICES
    ASPECTS = _GR_ASPECTS


class NineWanLiPluginM3_1(_GRImage2Image):
    MODEL = _dec("Im5hbm8tYmFuYW5hLXBybyI=")
    RESOLUTIONS = RESOLUTION_CHOICES
    ASPECTS = _GR_ASPECTS


class NineWanLiPluginM4(_GRText2Image):
    MODEL = _dec("ImdwdC1pbWFnZS0yIg==")
    RESOLUTIONS = ["1K"]
    ASPECTS = list(_GR_GPT_STD)
    GPT_PIXELS = {"1K": _GR_GPT_STD}


class NineWanLiPluginM4_1(_GRImage2Image):
    MODEL = _dec("ImdwdC1pbWFnZS0yIg==")
    RESOLUTIONS = ["1K"]
    ASPECTS = list(_GR_GPT_STD)
    GPT_PIXELS = {"1K": _GR_GPT_STD}


class NineWanLiPluginM5(_GRText2Image):
    MODEL = _dec("ImdwdC1pbWFnZS0yLjUtZmxhcmUi")
    RESOLUTIONS = RESOLUTION_CHOICES
    ASPECTS = list(_GR_GPT_VIP["1K"])
    GPT_PIXELS = _GR_GPT_VIP
    QUALITY_FIXED = "high"


class NineWanLiPluginM5_1(_GRImage2Image):
    MODEL = _dec("ImdwdC1pbWFnZS0yLjUtZmxhcmUi")
    RESOLUTIONS = RESOLUTION_CHOICES
    ASPECTS = list(_GR_GPT_VIP["1K"])
    GPT_PIXELS = _GR_GPT_VIP
    QUALITY_FIXED = "high"


class NineWanLiPluginM6(_GRText2Image):
    MODEL = _dec("Im5hbm8tYmFuYW5hLTIuMSI=")
    RESOLUTIONS = RESOLUTION_CHOICES
    ASPECTS = _GR_ASPECTS_WIDE


class NineWanLiPluginM6_1(_GRImage2Image):
    MODEL = _dec("Im5hbm8tYmFuYW5hLTIuMSI=")
    RESOLUTIONS = RESOLUTION_CHOICES
    ASPECTS = _GR_ASPECTS_WIDE


class NineWanLiPluginMV2(_GRText2Image):
    MODEL = _dec("Im5hbm8tYmFuYW5hLTIi")
    RESOLUTIONS = RESOLUTION_CHOICES
    ASPECTS = _GR_ASPECTS_WIDE


class NineWanLiPluginMV2_1(_GRImage2Image):
    MODEL = _dec("Im5hbm8tYmFuYW5hLTIi")
    RESOLUTIONS = RESOLUTION_CHOICES
    ASPECTS = _GR_ASPECTS_WIDE


NODE_CLASS_MAPPINGS.update({
    "NineWanLiPluginM1": NineWanLiPluginM1,
    "NineWanLiPluginM1_1": NineWanLiPluginM1_1,
    "NineWanLiPluginM3": NineWanLiPluginM3,
    "NineWanLiPluginM3_1": NineWanLiPluginM3_1,
    "NineWanLiPluginM4": NineWanLiPluginM4,
    "NineWanLiPluginM4_1": NineWanLiPluginM4_1,
    "NineWanLiPluginM5": NineWanLiPluginM5,
    "NineWanLiPluginM5_1": NineWanLiPluginM5_1,
    "NineWanLiPluginM6": NineWanLiPluginM6,
    "NineWanLiPluginM6_1": NineWanLiPluginM6_1,
    "NineWanLiPluginMV2": NineWanLiPluginMV2,
    "NineWanLiPluginMV2_1": NineWanLiPluginMV2_1,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "NineWanLiPluginM1": "云1",
    "NineWanLiPluginM1_1": "云1.1",
    "NineWanLiPluginM3": "云3",
    "NineWanLiPluginM3_1": "云3.1",
    "NineWanLiPluginM4": "云4",
    "NineWanLiPluginM4_1": "云4.1",
    "NineWanLiPluginM5": "云5",
    "NineWanLiPluginM5_1": "云5.1",
    "NineWanLiPluginM6": "云6",
    "NineWanLiPluginM6_1": "云6.1",
    "NineWanLiPluginMV2": "云V2",
    "NineWanLiPluginMV2_1": "云V2.1",
})

for _cls in [NineWanLiPluginM1, NineWanLiPluginM1_1, NineWanLiPluginM3, NineWanLiPluginM3_1,
             NineWanLiPluginM4, NineWanLiPluginM4_1, NineWanLiPluginM5, NineWanLiPluginM5_1,
             NineWanLiPluginM6, NineWanLiPluginM6_1, NineWanLiPluginMV2, NineWanLiPluginMV2_1]:
    _cls.NODE_NAME = NODE_DISPLAY_NAME_MAPPINGS.get(_cls.__name__, _cls.__name__)
NODE_CLASS_MAPPINGS.update({
    "木木2": NineWanLiPlugin2,
    "木木2.1": NineWanLiPlugin2_1,
    "木木3": NineWanLiPlugin3,
    "木木3.1": NineWanLiPlugin3_1,
    "南南5": NineWanLiPlugin5,
    "南南5.1": NineWanLiPlugin5_1,
    "木木4-备用": NineWanLiPlugin6,
    "木木4.1-备用": NineWanLiPlugin6_1,
    "木木5-备用": NineWanLiPlugin7,
    "木木5.1-备用": NineWanLiPlugin7_1,
    "木木V2": NineWanLiPluginV2,
    "木木V2.1": NineWanLiPluginV2_1,
    "南南V8": NineWanLiPluginV8,
    "南南V8.1": NineWanLiPluginV8_1,
    "南南5.6": NineWanLiPlugin5_6,
    "云1": NineWanLiPluginM1,
    "云1.1": NineWanLiPluginM1_1,
    "云3": NineWanLiPluginM3,
    "云3.1": NineWanLiPluginM3_1,
    "云4": NineWanLiPluginM4,
    "云4.1": NineWanLiPluginM4_1,
    "云5": NineWanLiPluginM5,
    "云5.1": NineWanLiPluginM5_1,
    "云6": NineWanLiPluginM6,
    "云6.1": NineWanLiPluginM6_1,
    "云V2": NineWanLiPluginMV2,
    "云V2.1": NineWanLiPluginMV2_1,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "木木2": "木木2",
    "木木2.1": "木木2.1",
    "木木3": "木木3",
    "木木3.1": "木木3.1",
    "南南5": "南南5",
    "南南5.1": "南南5.1",
    "木木4-备用": "木木4-备用",
    "木木4.1-备用": "木木4.1-备用",
    "木木5-备用": "木木5-备用",
    "木木5.1-备用": "木木5.1-备用",
    "木木V2": "木木V2",
    "木木V2.1": "木木V2.1",
    "南南V8": "南南V8",
    "南南V8.1": "南南V8.1",
    "南南5.6": "南南5.6",
    "云1": "云1",
    "云1.1": "云1.1",
    "云3": "云3",
    "云3.1": "云3.1",
    "云4": "云4",
    "云4.1": "云4.1",
    "云5": "云5",
    "云5.1": "云5.1",
    "云6": "云6",
    "云6.1": "云6.1",
    "云V2": "云V2",
    "云V2.1": "云V2.1",
})