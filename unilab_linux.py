#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniLab for Linux  —  数码显微镜实时查看与测量工具
A Linux re-implementation of the UniLab / ScopeView digital-microscope app.

功能 / Features:
  * 实时画面 (v4l2 / OpenCV)，设备 & 分辨率选择 (MJPG)
  * 相机参数调节 (亮度/对比度/饱和度/Gamma/增益/曝光/白平衡 … 由 v4l2 动态生成)
  * 水平/垂直翻转、十字线、网格、缩放/适应窗口、冻结画面
  * 拍照 (png/jpg/bmp)、录像 (mp4/avi，开始/暂停/停止)
  * 标定 (标定片)：画已知长度 → 输入真实值 → 得到 像素↔µm；多组命名标定，随分辨率自动换算
  * 测量：直线 / 水平线 / 垂直线 / 角度 / 圆(圆心+半径) / 三点圆 / 矩形 / 多边形 / 点 / 文字
  * 测量结果烧录导出

依赖: python3-pyqt6  opencv (cv2)  numpy   系统命令: v4l2-ctl (v4l-utils)
"""

import sys, os, glob, json, re, subprocess, time, math
from dataclasses import dataclass, field, asdict

import numpy as np
import cv2

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF, QTimer, QSize
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QFont, QAction,
                         QPolygonF, QBrush, QIcon, QKeySequence, QShortcut)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QComboBox,
                             QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QSlider,
                             QCheckBox, QDockWidget, QToolBar, QFileDialog, QMessageBox,
                             QInputDialog, QListWidget, QListWidgetItem, QColorDialog,
                             QSpinBox, QDoubleSpinBox, QDialog, QDialogButtonBox, QGroupBox,
                             QFormLayout, QScrollArea, QStatusBar, QStyle)

# ----------------------------------------------------------------------------- paths / config
APP_NAME = "UniLab for Linux"
CFG_DIR  = os.path.join(os.path.expanduser("~"), ".config", "unilab-linux")
CFG_FILE = os.path.join(CFG_DIR, "config.json")
CAL_FILE = os.path.join(CFG_DIR, "calibrations.json")
os.makedirs(CFG_DIR, exist_ok=True)


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("save config failed:", e)


# ----------------------------------------------------------------------------- v4l2 helpers
def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return ""


def device_node_index(path):
    m = re.search(r"(\d+)$", path)
    return int(m.group(1)) if m else 0


def list_capture_devices():
    """返回 [(/dev/videoN, '名称'), ...] 只保留能采集的节点。"""
    devs = []
    for p in sorted(glob.glob("/dev/video*"), key=device_node_index):
        fmts = _run(["v4l2-ctl", "-d", p, "--list-formats"])
        if "'MJPG'" in fmts or "'YUYV'" in fmts or "'YUV" in fmts or "Pixel Format" in fmts:
            info = _run(["v4l2-ctl", "-d", p, "--info"])
            name = p
            m = re.search(r"Card type\s*:\s*(.+)", info)
            if m:
                name = m.group(1).strip()
            devs.append((p, f"{name}  ({os.path.basename(p)})"))
    return devs


def list_resolutions(path, pixfmt="MJPG"):
    """返回 [(w,h), ...] 降序，优先 MJPG。"""
    out = _run(["v4l2-ctl", "-d", path, "--list-formats-ext"])
    res, cur = [], None
    for line in out.splitlines():
        m = re.search(r"\]:\s*'(\w+)'", line)
        if m:
            cur = m.group(1)
        m = re.search(r"Size:\s*Discrete\s+(\d+)x(\d+)", line)
        if m and cur == pixfmt:
            res.append((int(m.group(1)), int(m.group(2))))
    if not res:  # 回退到任何格式
        for line in out.splitlines():
            m = re.search(r"Size:\s*Discrete\s+(\d+)x(\d+)", line)
            if m:
                res.append((int(m.group(1)), int(m.group(2))))
    res = sorted(set(res), key=lambda wh: wh[0] * wh[1], reverse=True)
    return res


def list_controls(path):
    """解析 v4l2-ctl --list-ctrls-menus，返回控制项列表。"""
    out = _run(["v4l2-ctl", "-d", path, "--list-ctrls-menus"])
    ctrls, cur = [], None
    for line in out.splitlines():
        m = re.match(r"\s*([\w_]+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)", line)
        if m:
            name, typ, rest = m.group(1), m.group(2), m.group(3)
            d = {"name": name, "type": typ, "menu": {}}
            for key in ("min", "max", "step", "default", "value"):
                mm = re.search(rf"{key}=(-?\d+)", rest)
                if mm:
                    d[key] = int(mm.group(1))
            if "inactive" in rest:
                d["inactive"] = True
            cur = d
            ctrls.append(d)
        else:
            mm = re.match(r"\s*(\d+):\s*(.+)", line)   # 菜单项
            if mm and cur is not None:
                cur["menu"][int(mm.group(1))] = mm.group(2).strip()
    return ctrls


def set_control(path, name, value):
    subprocess.run(["v4l2-ctl", "-d", path, "--set-ctrl", f"{name}={value}"],
                   capture_output=True)


# ----------------------------------------------------------------------------- calibration
@dataclass
class Calibration:
    name: str
    um_per_px: float = 0.0     # 线性: 每像素多少微米 (在 ref_width 下)
    ref_width: int = 0         # 标定时的图像宽度(像素)
    unit: str = "um"
    kind: str = "linear"       # 'linear' 正对 / 'perspective' 透视 / 'xy' 十字两轴
    img_pts: list = None       # 透视: 标定时点的 4 个图像点 (在 ref_width 下)
    world_pts: list = None     # 透视: 对应的真实平面坐标 (µm)
    um_per_px_y: float = 0.0   # xy: 纵轴(Y)每像素微米；um_per_px 作为横轴(X)

    def effective(self, cur_width):
        if self.ref_width and cur_width:
            return self.um_per_px * self.ref_width / cur_width
        return self.um_per_px

    def mapper(self, cur_width):
        """返回 像素坐标 -> 真实平面(µm) 的映射函数。"""
        if self.kind == "xy":
            rsc = (self.ref_width / cur_width) if (self.ref_width and cur_width) else 1.0
            sx = self.um_per_px * rsc
            sy = self.um_per_px_y * rsc
            return (lambda p: (p[0] * sx, p[1] * sy))
        if self.kind == "perspective" and self.img_pts and self.world_pts:
            sc = (cur_width / self.ref_width) if self.ref_width else 1.0
            src = np.array([[x * sc, y * sc] for x, y in self.img_pts], np.float32)
            dst = np.array(self.world_pts, np.float32)
            H = cv2.getPerspectiveTransform(src, dst)

            def f(p):
                x, y = p
                d = H[2, 0]*x + H[2, 1]*y + H[2, 2]
                if abs(d) < 1e-12:
                    d = 1e-12
                return ((H[0, 0]*x + H[0, 1]*y + H[0, 2]) / d,
                        (H[1, 0]*x + H[1, 1]*y + H[1, 2]) / d)
            return f
        upp = self.effective(cur_width)
        return (lambda p: (p[0] * upp, p[1] * upp))


class CalibStore:
    def __init__(self):
        self.items = []
        self.active = None
        self.load()

    def load(self):
        data = load_json(CAL_FILE, {"items": [], "active": None})
        self.items = [Calibration(**d) for d in data.get("items", [])]
        self.active = data.get("active")

    def save(self):
        save_json(CAL_FILE, {"items": [asdict(c) for c in self.items],
                             "active": self.active})

    def get_active(self):
        for c in self.items:
            if c.name == self.active:
                return c
        return None

    def add(self, cal):
        self.items = [c for c in self.items if c.name != cal.name]
        self.items.append(cal)
        self.active = cal.name
        self.save()

    def remove(self, name):
        self.items = [c for c in self.items if c.name != name]
        if self.active == name:
            self.active = self.items[-1].name if self.items else None
        self.save()


# ----------------------------------------------------------------------------- measurement items
TOOLS = ["select", "line", "hline", "vline", "angle", "circle", "circle3",
         "rect", "polygon", "point", "text"]
TOOL_LABELS = {
    "select": "选择/移动", "line": "直线", "hline": "水平线", "vline": "垂直线",
    "angle": "角度", "circle": "圆(心+半径)", "circle3": "三点圆",
    "rect": "矩形", "polygon": "多边形", "point": "点", "text": "文字",
}
TOOL_NPTS = {"line": 2, "hline": 2, "vline": 2, "angle": 3, "circle": 2,
             "circle3": 3, "rect": 2, "point": 1, "text": 1}  # polygon=可变


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def circle_from_3(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None, 0
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    return (ux, uy), math.hypot(ax - ux, ay - uy)


def poly_area(pts):
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def poly_perim(pts, closed=True):
    n = len(pts)
    p = 0.0
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        p += dist(pts[i], pts[(i + 1) % n])
    return p


def fmt_um(um, unit):
    if unit == "mm" or um >= 1000:
        return f"{um/1000:.3f} mm"
    return f"{um:.1f} µm"


def fmt_um2(um2, unit):
    if unit == "mm" or um2 >= 1e6:
        return f"{um2/1e6:.4f} mm²"
    return f"{um2:.0f} µm²"


def metric_len(a, b, m, unit):
    """a,b 为图像像素点；m 为映射函数或 None。"""
    if m is None:
        return f"{dist(a, b):.1f} px"
    return fmt_um(dist(m(a), m(b)), unit)


def metric_perim(pts, m, closed=True):
    n = len(pts)
    rng = range(n) if closed else range(n - 1)
    if m is None:
        return sum(dist(pts[i], pts[(i+1) % n]) for i in rng)
    return sum(dist(m(pts[i]), m(pts[(i+1) % n])) for i in rng)


def metric_area(pts, m):
    if m is None:
        return poly_area(pts)
    return poly_area([m(p) for p in pts])


@dataclass
class MeasureItem:
    kind: str
    pts: list                      # image-coordinate points [(x,y),...]
    color: tuple = (0, 230, 0)
    width: int = 2
    text: str = ""

    def _rect_corners(self):
        (x0, y0), (x1, y1) = self.pts[0], self.pts[1]
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    # ---- geometry / measurement text ----
    # m: 映射函数(像素->µm) 或 None(显示像素)。线性与透视标定都通过它表达。
    def measure(self, m, unit):
        k = self.kind
        if k in ("line", "hline", "vline"):
            return metric_len(self.pts[0], self.pts[1], m, unit)
        if k == "angle":
            a, b, c = self.pts
            if m is not None:
                a, b, c = m(a), m(b), m(c)
            v1 = (a[0] - b[0], a[1] - b[1]); v2 = (c[0] - b[0], c[1] - b[1])
            n1 = math.hypot(*v1); n2 = math.hypot(*v2)
            if n1 < 1e-6 or n2 < 1e-6:
                return "0°"
            cosv = max(-1, min(1, (v1[0]*v2[0] + v1[1]*v2[1]) / (n1*n2)))
            return f"{math.degrees(math.acos(cosv)):.1f}°"
        if k in ("circle", "circle3"):
            if k == "circle":
                center, edge = self.pts[0], self.pts[1]
            else:
                c3, _ = circle_from_3(*self.pts)
                if c3 is None:
                    return "-"
                center, edge = c3, self.pts[0]
            if m is None:
                r = dist(center, edge)
                return (f"R={r:.1f}px  D={2*r:.1f}px  "
                        f"A={math.pi*r*r:.0f}px²")
            r = dist(m(center), m(edge))
            return (f"R={fmt_um(r,unit)}  D={fmt_um(2*r,unit)}  "
                    f"A={fmt_um2(math.pi*r*r,unit)}")
        if k == "rect":
            c = self._rect_corners()
            w = metric_len(c[0], c[1], m, unit)
            h = metric_len(c[1], c[2], m, unit)
            area = (fmt_um2(metric_area(c, m), unit) if m
                    else f"{metric_area(c, None):.0f}px²")
            return f"{w} × {h}  A={area}"
        if k == "polygon":
            L = metric_perim(self.pts, m)
            A = metric_area(self.pts, m)
            if m is None:
                return f"L={L:.1f}px  A={A:.0f}px²"
            return f"L={fmt_um(L,unit)}  A={fmt_um2(A,unit)}"
        if k == "point":
            if m is not None:
                mx, my = m(self.pts[0])
                return f"({mx:.1f}, {my:.1f}) µm"
            return f"({self.pts[0][0]:.0f},{self.pts[0][1]:.0f}) px"
        if k == "text":
            return self.text
        return ""

    def anchor(self):
        if self.kind == "polygon":
            xs = [p[0] for p in self.pts]; ys = [p[1] for p in self.pts]
            return (sum(xs)/len(xs), min(ys))
        return self.pts[-1]

    def near(self, p, tol):
        for q in self.pts:
            if dist(p, q) < tol:
                return True
        # 线段距离
        for i in range(len(self.pts) - (0 if self.kind == "polygon" else 1)):
            a = self.pts[i]; b = self.pts[(i + 1) % len(self.pts)]
            if _seg_dist(p, a, b) < tol:
                return True
        return False


def _seg_dist(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return dist(p, a)
    t = max(0, min(1, ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)))
    return math.hypot(px - (ax+t*dx), py - (ay+t*dy))


# ----------------------------------------------------------------------------- capture thread
class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, index, w, h, fourcc="MJPG", fps=30):
        super().__init__()
        self.index, self.w, self.h = index, w, h
        self.fourcc, self.fps = fourcc, fps
        self.running = False
        self.flip_h = self.flip_v = False
        self._writer = None
        self._rec_paused = False
        self._rec_lock = False

    def run(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.error.emit(f"无法打开 /dev/video{self.index}")
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        self.running = True
        while self.running:
            ok, frame = cap.read()
            if not ok:
                self.msleep(5)
                continue
            if self.flip_h:
                frame = cv2.flip(frame, 1)
            if self.flip_v:
                frame = cv2.flip(frame, 0)
            if self._writer is not None and not self._rec_paused and not self._rec_lock:
                try:
                    self._writer.write(frame)
                except Exception:
                    pass
            self.frame_ready.emit(frame)
        cap.release()

    def stop(self):
        self.running = False
        self.wait(1500)
        self.stop_recording()

    # recording ----------------------------------------------------------
    def start_recording(self, path, fps=None):
        fps = fps or self.fps
        ext = os.path.splitext(path)[1].lower()
        cc = cv2.VideoWriter_fourcc(*("mp4v" if ext == ".mp4" else "MJPG"))
        self._rec_lock = True
        self._writer = cv2.VideoWriter(path, cc, fps, (self.w, self.h))
        self._rec_paused = False
        self._rec_lock = False
        return self._writer.isOpened()

    def pause_recording(self, paused):
        self._rec_paused = paused

    def stop_recording(self):
        if self._writer is not None:
            self._rec_lock = True
            w = self._writer; self._writer = None
            try:
                w.release()
            except Exception:
                pass
            self._rec_lock = False


# ----------------------------------------------------------------------------- video view widget
class VideoView(QWidget):
    status = pyqtSignal(str)
    items_changed = pyqtSignal()

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(480, 360)
        self.setStyleSheet("background:#111;")
        self.frame = None
        self.pixmap = None
        self.frozen = False
        self.zoom = 1.0
        self.fit = True
        self.scale = 1.0; self.ox = self.oy = 0
        # overlays
        self.show_grid = False
        self.show_cross = False
        # drawing
        self.tool = "select"
        self.color = (0, 230, 0)
        self.penw = 2
        self.unit = "um"
        self.items = []
        self.pending = []
        self.hover = None
        self.selected = -1
        self._drag_idx = None  # (item, point-index) when dragging in select mode
        self._flash_until = 0.0
        self._flash_msg = ""
        self._warned_nocal = False

    # ---- frame input
    def set_frame(self, frame):
        if self.frozen:
            return
        self.frame = frame
        self._update_pixmap()
        self.update()

    def _update_pixmap(self):
        if self.frame is None:
            self.pixmap = None
            return
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.pixmap = QPixmap.fromImage(img)

    def frame_size(self):
        if self.frame is not None:
            h, w, _ = self.frame.shape
            return w, h
        return 0, 0

    # ---- coordinate mapping
    def _recompute(self):
        w, h = self.frame_size()
        if w == 0:
            return
        fitscale = min(self.width() / w, self.height() / h)
        self.scale = fitscale * (1.0 if self.fit else self.zoom)
        dispw, disph = w * self.scale, h * self.scale
        self.ox = (self.width() - dispw) / 2
        self.oy = (self.height() - disph) / 2

    def to_widget(self, p):
        return QPointF(self.ox + p[0] * self.scale, self.oy + p[1] * self.scale)

    def to_image(self, q):
        return ((q.x() - self.ox) / self.scale, (q.y() - self.oy) / self.scale)

    # ---- painting
    def paintEvent(self, ev):
        qp = QPainter(self)
        qp.fillRect(self.rect(), QColor("#111"))
        if self.pixmap is None:
            qp.setPen(QColor("#888"))
            qp.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                        "未连接相机 —  点击左上角『打开相机』")
            return
        self._recompute()
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        w, h = self.frame_size()
        target = QRectF(self.ox, self.oy, w * self.scale, h * self.scale)
        qp.drawPixmap(target, self.pixmap, QRectF(self.pixmap.rect()))
        self._draw_overlays(qp, w, h)
        for i, it in enumerate(self.items):
            self._draw_item(qp, it, selected=(i == self.selected))
        if self.pending:
            self._draw_pending(qp)
        if time.time() < self._flash_until:
            qp.setPen(QPen(QColor(255, 255, 255, 230), 6))
            qp.drawRect(self.rect().adjusted(3, 3, -3, -3))
            f = QFont("Sans", 13); f.setBold(True); qp.setFont(f)
            qp.setPen(QColor(0, 0, 0, 180))
            qp.drawText(self.rect().adjusted(0, 14, 0, 0),
                        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, "  " + self._flash_msg)
            qp.setPen(QColor(60, 230, 90))
            qp.drawText(self.rect().adjusted(0, 12, 0, 0),
                        Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self._flash_msg)

    def flash(self, msg):
        self._flash_until = time.time() + 1.2
        self._flash_msg = msg
        self.update()
        QTimer.singleShot(1300, self.update)

    def _draw_overlays(self, qp, w, h):
        if self.show_grid:
            qp.setPen(QPen(QColor(255, 255, 255, 70), 1))
            for i in range(1, 10):
                x = self.ox + w * self.scale * i / 10
                y = self.oy + h * self.scale * i / 10
                qp.drawLine(QPointF(x, self.oy), QPointF(x, self.oy + h * self.scale))
                qp.drawLine(QPointF(self.ox, y), QPointF(self.ox + w * self.scale, y))
        if self.show_cross:
            qp.setPen(QPen(QColor(255, 80, 80, 180), 1))
            cx = self.ox + w * self.scale / 2
            cy = self.oy + h * self.scale / 2
            qp.drawLine(QPointF(cx, self.oy), QPointF(cx, self.oy + h * self.scale))
            qp.drawLine(QPointF(self.ox, cy), QPointF(self.ox + w * self.scale, cy))

    def _pen(self, color, wd, dashed=False):
        p = QPen(QColor(*color), wd)
        p.setCosmetic(True)
        if dashed:
            p.setStyle(Qt.PenStyle.DashLine)
        return p

    def _text(self, qp, pos_img, s, color):
        if not s:
            return
        wpt = self.to_widget(pos_img)
        f = QFont("Sans", 10); f.setBold(True)
        qp.setFont(f)
        x, y = wpt.x() + 6, wpt.y() - 6
        qp.setPen(QColor(0, 0, 0, 200))
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            qp.drawText(QPointF(x + dx, y + dy), s)
        qp.setPen(QColor(*color))
        qp.drawText(QPointF(x, y), s)

    def _draw_item(self, qp, it, selected=False, preview=False):
        m = self.win.metric_mapper()
        col = it.color
        qp.setPen(self._pen(col, it.width))
        pw = [self.to_widget(p) for p in it.pts]
        k = it.kind
        if k in ("line", "hline", "vline") and len(pw) == 2:
            qp.drawLine(pw[0], pw[1])
        elif k == "angle" and len(pw) >= 2:
            if len(pw) >= 2:
                qp.drawLine(pw[1], pw[0])
            if len(pw) >= 3:
                qp.drawLine(pw[1], pw[2])
        elif k == "circle" and len(pw) == 2:
            r = (pw[1] - pw[0]); rad = math.hypot(r.x(), r.y())
            qp.drawEllipse(pw[0], rad, rad)
            qp.drawLine(pw[0], pw[1])
        elif k == "circle3" and len(it.pts) == 3:
            c, rad = circle_from_3(*it.pts)
            if c:
                cw = self.to_widget(c)
                qp.drawEllipse(cw, rad * self.scale, rad * self.scale)
            for p in pw:
                qp.drawEllipse(p, 3, 3)
        elif k == "rect" and len(pw) == 2:
            qp.drawRect(QRectF(pw[0], pw[1]).normalized())
        elif k == "polygon" and len(pw) >= 2:
            poly = QPolygonF(pw)
            if not preview:
                qp.drawPolygon(poly)
            else:
                qp.drawPolyline(poly)
        elif k == "point" and pw:
            self._marker(qp, pw[0])
        elif k == "text" and pw:
            self._marker(qp, pw[0])
        # 端点小标记
        if k in ("line", "hline", "vline", "rect", "circle", "angle"):
            for p in pw:
                qp.drawEllipse(p, 2.5, 2.5)
        if selected:
            qp.setPen(self._pen((255, 220, 0), it.width + 2, dashed=True))
            for p in pw:
                qp.drawEllipse(p, 6, 6)
        if not preview:
            self._text(qp, it.anchor(), it.measure(m, self.unit), col)

    def _marker(self, qp, p):
        qp.drawLine(QPointF(p.x() - 6, p.y()), QPointF(p.x() + 6, p.y()))
        qp.drawLine(QPointF(p.x(), p.y() - 6), QPointF(p.x(), p.y() + 6))

    def _draw_pending(self, qp):
        pts = list(self.pending)
        if self.hover is not None and self.tool != "select":
            prev = pts + [self.hover]
        else:
            prev = pts
        tmp = MeasureItem(self.tool, prev, self.color, self.penw)
        if self.tool == "point" or self.tool == "text":
            return
        self._draw_item(qp, tmp, preview=True)

    # ---- mouse / keyboard
    def mousePressEvent(self, ev):
        if self.pixmap is None:
            return
        p = self.to_image(ev.position())
        if ev.button() == Qt.MouseButton.RightButton:
            self.pending = []
            self.update()
            return
        if (self.tool != "select" and not self._warned_nocal
                and self.win.metric_mapper() is None):
            self._warned_nocal = True
            self.flash("未标定：当前显示像素，先做一次标定才能得到 µm")
        if self.tool == "select":
            self._select_at(p)
            if self.selected >= 0:
                it = self.items[self.selected]
                for i, q in enumerate(it.pts):
                    if dist(p, q) < 10 / self.scale:
                        self._drag_idx = (self.selected, i)
                        break
            return
        if self.tool == "text":
            s, ok = QInputDialog.getText(self, "文字标注", "输入文字:")
            if ok and s:
                self.items.append(MeasureItem("text", [p], self.color, self.penw, text=s))
                self._commit()
            return
        if self.tool == "point":
            self.items.append(MeasureItem("point", [p], self.color, self.penw))
            self._commit()
            return
        # hline / vline 约束
        if self.tool == "hline" and self.pending:
            p = (p[0], self.pending[0][1])
        if self.tool == "vline" and self.pending:
            p = (self.pending[0][0], p[1])
        self.pending.append(p)
        need = TOOL_NPTS.get(self.tool)
        if need and len(self.pending) >= need:
            self.items.append(MeasureItem(self.tool, list(self.pending), self.color, self.penw))
            self.pending = []
            self._commit()
        self.update()

    def mouseDoubleClickEvent(self, ev):
        if self.tool == "polygon" and len(self.pending) >= 3:
            self.items.append(MeasureItem("polygon", list(self.pending), self.color, self.penw))
            self.pending = []
            self._commit()

    def mouseMoveEvent(self, ev):
        if self.pixmap is None:
            return
        p = self.to_image(ev.position())
        self.hover = p
        if self._drag_idx is not None:
            it_i, pt_i = self._drag_idx
            self.items[it_i].pts[pt_i] = p
            self.update()
        m = self.win.metric_mapper()
        pos = f"x={p[0]:.0f} y={p[1]:.0f}"
        if m is not None:
            mx, my = m(p)
            pos += f"   ({mx:.1f}, {my:.1f}) µm"
        self.status.emit(pos)
        if self.tool != "select":
            self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag_idx is not None:
            self._drag_idx = None
            self.items_changed.emit()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.tool == "polygon" and len(self.pending) >= 3:
                self.items.append(MeasureItem("polygon", list(self.pending), self.color, self.penw))
                self.pending = []
                self._commit()
        elif ev.key() == Qt.Key.Key_Escape:
            self.pending = []
            self.update()
        elif ev.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if 0 <= self.selected < len(self.items):
                del self.items[self.selected]
                self.selected = -1
                self._commit()

    def _select_at(self, p):
        tol = 10 / self.scale
        self.selected = -1
        for i in range(len(self.items) - 1, -1, -1):
            if self.items[i].near(p, tol):
                self.selected = i
                break
        self.update()
        self.items_changed.emit()

    def _commit(self):
        self.update()
        self.items_changed.emit()


# ----------------------------------------------------------------------------- calibration dialog
class CalibrateDialog(QDialog):
    """在冻结画面上画一条已知长度的线，输入真实长度 → 得到 µm/px。"""
    def __init__(self, win, frame):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("新建标定 (标定片)")
        self.resize(900, 640)
        self.frame = frame
        self.p0 = self.p1 = None
        self.scale = 1.0; self.ox = self.oy = 0

        lay = QVBoxLayout(self)
        tip = QLabel("① 标定片对焦清晰（本页用当前冻结画面）\n"
                     "② 沿已知刻度拖一条线：可用直尺(0–40mm)、线段(0.03–0.2mm)或方格边\n"
                     "③ 填『每格长度 × 格数』——画过几格就填几格，右上角放大镜帮你对准刻度")
        lay.addWidget(tip)
        self.canvas = _CalCanvas(self)
        lay.addWidget(self.canvas, 1)

        form = QHBoxLayout()
        form.addWidget(QLabel("名称:"))
        self.name = QComboBox(); self.name.setEditable(True)
        self.name.addItems(["10x", "40x", "100x", "标定1"])
        form.addWidget(self.name)
        form.addWidget(QLabel("每格长度:"))
        self.val = QDoubleSpinBox(); self.val.setRange(0.0001, 1e6)
        self.val.setDecimals(4); self.val.setValue(1.0)
        self.val.valueChanged.connect(self._upd_total)
        form.addWidget(self.val)
        self.unit = QComboBox(); self.unit.addItems(["µm", "mm"])
        self.unit.currentIndexChanged.connect(self._upd_total)
        form.addWidget(self.unit)
        form.addWidget(QLabel("× 格数:"))
        self.ndiv = QSpinBox(); self.ndiv.setRange(1, 100000); self.ndiv.setValue(1)
        self.ndiv.valueChanged.connect(self._upd_total)
        form.addWidget(self.ndiv)
        self.totlbl = QLabel("总长=1 mm")
        form.addWidget(self.totlbl)
        self.pxlbl = QLabel("线长: 0 px")
        form.addWidget(self.pxlbl)
        lay.addLayout(form)
        self._upd_total()

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def px_len(self):
        if self.p0 and self.p1:
            return dist(self.p0, self.p1)
        return 0

    def _total_um(self):
        k = 1000.0 if self.unit.currentText() == "mm" else 1.0
        return self.val.value() * k * self.ndiv.value()

    def _upd_total(self):
        um = self._total_um()
        self.totlbl.setText(f"总长={um/1000:.4g} mm" if um >= 1000 else f"总长={um:.4g} µm")

    def result_calib(self):
        px = self.px_len()
        if px < 1:
            return None
        w = self.frame.shape[1]
        return Calibration(self.name.currentText().strip() or "标定",
                           self._total_um() / px, w, "um")


def draw_loupe(qp, pix, ip, widget_w, size=160, zoom=6):
    """在右上角画一个放大镜：放大 ip(图像坐标) 周围区域 + 十字准星。"""
    src = QRectF(ip[0] - size/(2*zoom), ip[1] - size/(2*zoom), size/zoom, size/zoom)
    dst = QRectF(widget_w - size - 10, 10, size, size)
    qp.setBrush(QColor(0, 0, 0))
    qp.setPen(QPen(QColor(0, 230, 255), 2))
    qp.drawRect(dst)
    qp.drawPixmap(dst, pix, src)
    qp.setPen(QPen(QColor(255, 80, 80), 1))
    c = dst.center()
    qp.drawLine(QPointF(c.x(), dst.top()), QPointF(c.x(), dst.bottom()))
    qp.drawLine(QPointF(dst.left(), c.y()), QPointF(dst.right(), c.y()))
    qp.setBrush(Qt.BrushStyle.NoBrush)


class _CalCanvas(QWidget):
    def __init__(self, dlg):
        super().__init__()
        self.dlg = dlg
        self.setMinimumSize(640, 440)
        self.setMouseTracking(True)
        self.cur = None
        rgb = cv2.cvtColor(dlg.frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        self.pix = QPixmap.fromImage(QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888).copy())
        self.iw, self.ih = w, h
        self.dragging = False

    def _map(self):
        s = min(self.width()/self.iw, self.height()/self.ih)
        ox = (self.width()-self.iw*s)/2; oy = (self.height()-self.ih*s)/2
        return s, ox, oy

    def to_img(self, q):
        s, ox, oy = self._map()
        return ((q.x()-ox)/s, (q.y()-oy)/s)

    def to_w(self, p):
        s, ox, oy = self._map()
        return QPointF(ox+p[0]*s, oy+p[1]*s)

    def paintEvent(self, ev):
        qp = QPainter(self)
        qp.fillRect(self.rect(), QColor("#111"))
        s, ox, oy = self._map()
        qp.drawPixmap(QRectF(ox, oy, self.iw*s, self.ih*s), self.pix, QRectF(self.pix.rect()))
        if self.dlg.p0 and self.dlg.p1:
            qp.setPen(QPen(QColor(0, 230, 0), 2))
            a = self.to_w(self.dlg.p0); b = self.to_w(self.dlg.p1)
            qp.drawLine(a, b)
            for p in (a, b):
                qp.drawEllipse(p, 3, 3)
        lp = self.dlg.p1 if self.dragging else self.cur
        if lp is not None:
            draw_loupe(qp, self.pix, lp, self.width())

    def mousePressEvent(self, ev):
        self.dlg.p0 = self.to_img(ev.position()); self.dlg.p1 = self.dlg.p0
        self.dragging = True; self.update()

    def mouseMoveEvent(self, ev):
        self.cur = self.to_img(ev.position())
        if self.dragging:
            self.dlg.p1 = self.cur
            self.dlg.pxlbl.setText(f"线长: {self.dlg.px_len():.1f} px")
        self.update()

    def mouseReleaseEvent(self, ev):
        self.dragging = False


# ----------------------------------------------------------------------------- perspective calibration
class PerspectiveDialog(QDialog):
    """透视标定：点选一个已知尺寸矩形的 4 个角(顺时针，从左上开始)，
    输入它的真实宽、高 → 得到单应性矩阵，测量时自动纠正倾斜透视。"""
    CORNERS = ["左上", "右上", "右下", "左下"]

    def __init__(self, win, frame):
        super().__init__(win)
        self.win = win
        self.frame = frame
        self.setWindowTitle("新建透视标定 (倾斜纠正)")
        self.resize(900, 680)
        self.pts = []

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "① 放一个已知尺寸的矩形/方格参照（标定片的 0.1mm 方格、直角坐标十字都行）\n"
            "② 依次点 4 个角：左上 → 右上 → 右下 → 左下（放大镜帮你对准）\n"
            "③ 填『每格宽高 × 列数/行数』——跨了几格就填几格 → 确定。之后画面斜了也能正确测量。"))
        self.canvas = _PerspCanvas(self)
        lay.addWidget(self.canvas, 1)

        self.hint = QLabel("请点击：左上 角"); lay.addWidget(self.hint)
        form = QHBoxLayout()
        form.addWidget(QLabel("名称:"))
        self.name = QComboBox(); self.name.setEditable(True)
        self.name.addItems(["透视-10x", "透视-40x", "透视1"])
        form.addWidget(self.name)
        form.addWidget(QLabel("每格宽:"))
        self.w_val = QDoubleSpinBox(); self.w_val.setRange(0.0001, 1e6)
        self.w_val.setDecimals(4); self.w_val.setValue(0.1); form.addWidget(self.w_val)
        form.addWidget(QLabel("× 列:"))
        self.cols = QSpinBox(); self.cols.setRange(1, 100000); self.cols.setValue(10); form.addWidget(self.cols)
        form.addWidget(QLabel("每格高:"))
        self.h_val = QDoubleSpinBox(); self.h_val.setRange(0.0001, 1e6)
        self.h_val.setDecimals(4); self.h_val.setValue(0.1); form.addWidget(self.h_val)
        form.addWidget(QLabel("× 行:"))
        self.rows = QSpinBox(); self.rows.setRange(1, 100000); self.rows.setValue(10); form.addWidget(self.rows)
        self.unit = QComboBox(); self.unit.addItems(["µm", "mm"]); self.unit.setCurrentText("mm")
        form.addWidget(self.unit)
        b = QPushButton("重选角点"); b.clicked.connect(self._reset); form.addWidget(b)
        lay.addLayout(form)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _reset(self):
        self.pts = []
        self.hint.setText("请点击：左上 角")
        self.canvas.update()

    def add_point(self, p):
        if len(self.pts) >= 4:
            return
        self.pts.append(p)
        if len(self.pts) < 4:
            self.hint.setText(f"请点击：{self.CORNERS[len(self.pts)]} 角")
        else:
            self.hint.setText("4 个角已就位，可确定（或『重选角点』）")
        self.canvas.update()

    def result_calib(self):
        if len(self.pts) != 4:
            return None
        k = 1000.0 if self.unit.currentText() == "mm" else 1.0
        W = self.w_val.value() * k * self.cols.value()
        H = self.h_val.value() * k * self.rows.value()
        world = [(0.0, 0.0), (W, 0.0), (W, H), (0.0, H)]
        w = self.frame.shape[1]
        return Calibration(self.name.currentText().strip() or "透视标定",
                           kind="perspective", ref_width=w,
                           img_pts=[list(p) for p in self.pts], world_pts=world)


class _PerspCanvas(QWidget):
    def __init__(self, dlg):
        super().__init__()
        self.dlg = dlg
        self.setMinimumSize(640, 460)
        self.setMouseTracking(True)
        self.cur = None
        rgb = cv2.cvtColor(dlg.frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        self.pix = QPixmap.fromImage(QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888).copy())
        self.iw, self.ih = w, h

    def _map(self):
        s = min(self.width()/self.iw, self.height()/self.ih)
        return s, (self.width()-self.iw*s)/2, (self.height()-self.ih*s)/2

    def to_img(self, q):
        s, ox, oy = self._map()
        return ((q.x()-ox)/s, (q.y()-oy)/s)

    def to_w(self, p):
        s, ox, oy = self._map()
        return QPointF(ox+p[0]*s, oy+p[1]*s)

    def paintEvent(self, ev):
        qp = QPainter(self)
        qp.fillRect(self.rect(), QColor("#111"))
        s, ox, oy = self._map()
        qp.drawPixmap(QRectF(ox, oy, self.iw*s, self.ih*s), self.pix, QRectF(self.pix.rect()))
        pts = self.dlg.pts
        qp.setPen(QPen(QColor(0, 230, 255), 2))
        wpts = [self.to_w(p) for p in pts]
        if len(wpts) >= 2:
            qp.drawPolyline(QPolygonF(wpts + ([wpts[0]] if len(wpts) == 4 else [])))
        for i, wp in enumerate(wpts):
            qp.setPen(QPen(QColor(0, 230, 255), 2))
            qp.drawEllipse(wp, 4, 4)
            qp.setPen(QColor(255, 255, 0))
            qp.drawText(QPointF(wp.x()+6, wp.y()-6), self.dlg.CORNERS[i])
        if self.cur is not None and len(pts) < 4:
            draw_loupe(qp, self.pix, self.cur, self.width())

    def mousePressEvent(self, ev):
        self.dlg.add_point(self.to_img(ev.position()))

    def mouseMoveEvent(self, ev):
        self.cur = self.to_img(ev.position())
        self.update()


# ----------------------------------------------------------------------------- cross (XY) calibration
class CrossDialog(QDialog):
    """十字坐标标定：沿横轴画一条、沿纵轴画一条，分别得到 X / Y 比例。
    适合标定片的直角坐标十字，可处理像素非正方 / XY 不等比(不含倾斜)。"""
    def __init__(self, win, frame):
        super().__init__(win)
        self.win = win
        self.frame = frame
        self.setWindowTitle("新建十字标定 (XY 两轴)")
        self.resize(900, 700)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "① 对准标定片的直角坐标十字\n"
            "② 选『X 轴』沿横轴拖一条已知长度的线；再选『Y 轴』沿纵轴拖一条\n"
            "③ 分别填每格长度和格数 → 确定（放大镜帮你对准刻度）"))
        self.canvas = _CrossCanvas(self)
        lay.addWidget(self.canvas, 1)

        top = QHBoxLayout()
        top.addWidget(QLabel("当前绘制:"))
        self.axis = QComboBox(); self.axis.addItems(["X 轴 (横)", "Y 轴 (纵)"])
        self.axis.currentIndexChanged.connect(
            lambda i: setattr(self.canvas, "axis", "x" if i == 0 else "y"))
        top.addWidget(self.axis)
        self.xpx = QLabel("X: 0 px"); top.addWidget(self.xpx)
        self.ypx = QLabel("Y: 0 px"); top.addWidget(self.ypx)
        b = QPushButton("重画"); b.clicked.connect(self._reset); top.addWidget(b)
        top.addStretch(1)
        lay.addLayout(top)

        form = QHBoxLayout()
        form.addWidget(QLabel("名称:"))
        self.name = QComboBox(); self.name.setEditable(True)
        self.name.addItems(["十字-10x", "十字1"]); form.addWidget(self.name)
        form.addWidget(QLabel("X 每格:"))
        self.xval = QDoubleSpinBox(); self.xval.setRange(0.0001, 1e6); self.xval.setDecimals(4)
        self.xval.setValue(0.1); form.addWidget(self.xval)
        form.addWidget(QLabel("×"))
        self.xn = QSpinBox(); self.xn.setRange(1, 100000); self.xn.setValue(10); form.addWidget(self.xn)
        form.addWidget(QLabel("  Y 每格:"))
        self.yval = QDoubleSpinBox(); self.yval.setRange(0.0001, 1e6); self.yval.setDecimals(4)
        self.yval.setValue(0.1); form.addWidget(self.yval)
        form.addWidget(QLabel("×"))
        self.yn = QSpinBox(); self.yn.setRange(1, 100000); self.yn.setValue(10); form.addWidget(self.yn)
        self.unit = QComboBox(); self.unit.addItems(["µm", "mm"]); self.unit.setCurrentText("mm")
        form.addWidget(self.unit)
        lay.addLayout(form)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _reset(self):
        self.canvas.xline = None; self.canvas.yline = None
        self.xpx.setText("X: 0 px"); self.ypx.setText("Y: 0 px")
        self.canvas.update()

    def result_calib(self):
        xl, yl = self.canvas.xline, self.canvas.yline
        if not xl or not yl:
            return None
        xpx = dist(xl[0], xl[1]); ypx = dist(yl[0], yl[1])
        if xpx < 1 or ypx < 1:
            return None
        k = 1000.0 if self.unit.currentText() == "mm" else 1.0
        xum = self.xval.value() * k * self.xn.value()
        yum = self.yval.value() * k * self.yn.value()
        return Calibration(self.name.currentText().strip() or "十字标定",
                           um_per_px=xum / xpx, ref_width=self.frame.shape[1],
                           kind="xy", um_per_px_y=yum / ypx)


class _CrossCanvas(QWidget):
    def __init__(self, dlg):
        super().__init__()
        self.dlg = dlg
        self.setMinimumSize(640, 460)
        self.setMouseTracking(True)
        self.cur = None
        self.axis = "x"
        self.xline = None
        self.yline = None
        self.dragging = False
        rgb = cv2.cvtColor(dlg.frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        self.pix = QPixmap.fromImage(QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888).copy())
        self.iw, self.ih = w, h

    def _map(self):
        s = min(self.width()/self.iw, self.height()/self.ih)
        return s, (self.width()-self.iw*s)/2, (self.height()-self.ih*s)/2

    def to_img(self, q):
        s, ox, oy = self._map()
        return ((q.x()-ox)/s, (q.y()-oy)/s)

    def to_w(self, p):
        s, ox, oy = self._map()
        return QPointF(ox+p[0]*s, oy+p[1]*s)

    def paintEvent(self, ev):
        qp = QPainter(self)
        qp.fillRect(self.rect(), QColor("#111"))
        s, ox, oy = self._map()
        qp.drawPixmap(QRectF(ox, oy, self.iw*s, self.ih*s), self.pix, QRectF(self.pix.rect()))
        for ln, col in ((self.xline, QColor(0, 230, 0)), (self.yline, QColor(0, 200, 255))):
            if ln:
                qp.setPen(QPen(col, 2))
                a = self.to_w(ln[0]); b = self.to_w(ln[1])
                qp.drawLine(a, b)
                qp.drawEllipse(a, 3, 3); qp.drawEllipse(b, 3, 3)
        lp = None
        if self.dragging:
            lp = (self.xline or self.yline)
            lp = (self.xline[1] if self.axis == "x" and self.xline else
                  (self.yline[1] if self.yline else None))
        if lp is None:
            lp = self.cur
        if lp is not None:
            draw_loupe(qp, self.pix, lp, self.width())

    def mousePressEvent(self, ev):
        p = self.to_img(ev.position())
        if self.axis == "x":
            self.xline = [p, p]
        else:
            self.yline = [p, p]
        self.dragging = True
        self.update()

    def mouseMoveEvent(self, ev):
        self.cur = self.to_img(ev.position())
        if self.dragging:
            if self.axis == "x" and self.xline:
                self.xline[1] = self.cur
                self.dlg.xpx.setText(f"X: {dist(*self.xline):.1f} px")
            elif self.yline:
                self.yline[1] = self.cur
                self.dlg.ypx.setText(f"Y: {dist(*self.yline):.1f} px")
        self.update()

    def mouseReleaseEvent(self, ev):
        self.dragging = False


# ----------------------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1280, 820)
        self.cfg = load_json(CFG_FILE, {})
        self.calib = CalibStore()
        self.thread = None
        self.devices = list_capture_devices()
        self.resolutions = []
        self.recording = False
        self.rec_paused = False
        self.save_dir = self.cfg.get("save_dir", os.path.join(os.path.expanduser("~"), "UniLab Files"))
        os.makedirs(self.save_dir, exist_ok=True)
        self.snap_format = self.cfg.get("snap_format", "png")
        self.snap_annotate = self.cfg.get("snap_annotate", True)
        self._last_saved = None

        self.view = VideoView(self)
        self.view.status.connect(self._on_cursor)
        self.view.items_changed.connect(self._refresh_items)
        self.view.unit = self.cfg.get("unit", "um")
        self.setCentralWidget(self.view)

        self._build_toolbar()
        self._build_camera_dock()
        self._build_measure_dock()
        self.setStatusBar(QStatusBar())
        self._status_cal = QLabel(""); self.statusBar().addPermanentWidget(self._status_cal)
        self._update_cal_status()

        self.fps_timer = QTimer(self); self.fps_timer.timeout.connect(self._tick); self.fps_timer.start(1000)
        self._frames = 0

        # 全局拍照快捷键：空格 / Ctrl+S / F2，无论焦点在哪都生效
        for seq in ("Space", "Ctrl+S", "F2"):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(self.snapshot)

        if self.devices:
            idx = 0
            for i, (p, _) in enumerate(self.devices):
                if p == self.cfg.get("device"):
                    idx = i
            self.dev_combo.setCurrentIndex(idx)
            self._on_device_changed()

    # ---------- UI build
    def _icon(self, std):
        return self.style().standardIcon(std)

    def _build_toolbar(self):
        tb = QToolBar("主工具栏"); tb.setIconSize(QSize(20, 20))
        self.addToolBar(tb)
        st = QStyle.StandardPixmap
        self.btn_open = QPushButton("  打开相机  ")
        self.btn_open.setIcon(self._icon(st.SP_MediaPlay))
        self.btn_open.clicked.connect(self._toggle_camera)
        tb.addWidget(self.btn_open)
        self._update_open_btn()
        tb.addSeparator()
        self.act_snap = QAction(self._icon(st.SP_DialogSaveButton), "拍照", self)
        self.act_snap.setToolTip("拍照 (空格 / Ctrl+S / F2)")
        self.act_snap.triggered.connect(self.snapshot); tb.addAction(self.act_snap)
        self.act_rec = QAction(self._icon(st.SP_DriveDVDIcon), "录像", self)
        self.act_rec.triggered.connect(self.toggle_record); tb.addAction(self.act_rec)
        self.act_pause = QAction(self._icon(st.SP_MediaPause), "暂停录像", self)
        self.act_pause.triggered.connect(self.pause_record); self.act_pause.setEnabled(False)
        tb.addAction(self.act_pause)
        tb.addSeparator()
        self.act_freeze = QAction("冻结", self); self.act_freeze.setCheckable(True)
        self.act_freeze.triggered.connect(self.toggle_freeze); tb.addAction(self.act_freeze)
        self.act_fliph = QAction("水平翻转", self); self.act_fliph.setCheckable(True)
        self.act_fliph.triggered.connect(self._apply_flip); tb.addAction(self.act_fliph)
        self.act_flipv = QAction("垂直翻转", self); self.act_flipv.setCheckable(True)
        self.act_flipv.triggered.connect(self._apply_flip); tb.addAction(self.act_flipv)
        self.act_grid = QAction("网格", self); self.act_grid.setCheckable(True)
        self.act_grid.triggered.connect(lambda c: self._set_overlay("grid", c)); tb.addAction(self.act_grid)
        self.act_cross = QAction("十字线", self); self.act_cross.setCheckable(True)
        self.act_cross.triggered.connect(lambda c: self._set_overlay("cross", c)); tb.addAction(self.act_cross)
        tb.addSeparator()
        a = QAction("放大", self); a.triggered.connect(lambda: self._zoom(1.25)); tb.addAction(a)
        a = QAction("缩小", self); a.triggered.connect(lambda: self._zoom(0.8)); tb.addAction(a)
        a = QAction("适应窗口", self); a.triggered.connect(self._zoom_fit); tb.addAction(a)
        tb.addSeparator()
        a = QAction("保存目录…", self); a.triggered.connect(self.choose_dir); tb.addAction(a)
        a = QAction("打开目录", self); a.triggered.connect(self.open_dir); tb.addAction(a)
        tb.addSeparator()
        tb.addWidget(QLabel(" 格式:"))
        self.fmt_combo = QComboBox(); self.fmt_combo.addItems(["png", "jpg", "bmp"])
        self.fmt_combo.setCurrentText(self.snap_format)
        self.fmt_combo.currentTextChanged.connect(self._set_snap_format)
        tb.addWidget(self.fmt_combo)
        self.annot_cb = QCheckBox("拍照烧录测量")
        self.annot_cb.setChecked(self.snap_annotate)
        self.annot_cb.setToolTip("勾选=把测量线/数值画进照片；不勾=存原始画面")
        self.annot_cb.toggled.connect(self._set_snap_annotate)
        tb.addWidget(self.annot_cb)

    def _set_snap_format(self, t):
        self.snap_format = t; self.cfg["snap_format"] = t; save_json(CFG_FILE, self.cfg)

    def _set_snap_annotate(self, v):
        self.snap_annotate = v; self.cfg["snap_annotate"] = v; save_json(CFG_FILE, self.cfg)

    def open_dir(self):
        subprocess.Popen(["xdg-open", self.save_dir])

    def _build_camera_dock(self):
        dock = QDockWidget("相机", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        w = QWidget(); lay = QVBoxLayout(w)
        row = QFormLayout()
        self.dev_combo = QComboBox()
        for p, name in self.devices:
            self.dev_combo.addItem(name, p)
        if not self.devices:
            self.dev_combo.addItem("未找到相机")
        self.dev_combo.currentIndexChanged.connect(self._on_device_changed)
        row.addRow("设备:", self.dev_combo)
        self.res_combo = QComboBox()
        self.res_combo.currentIndexChanged.connect(self._on_res_changed)
        row.addRow("分辨率:", self.res_combo)
        lay.addLayout(row)

        self.ctrl_box = QGroupBox("图像参数")
        self.ctrl_form = QFormLayout(self.ctrl_box)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(self.ctrl_box)
        lay.addWidget(sc, 1)
        btns = QHBoxLayout()
        b = QPushButton("恢复默认"); b.clicked.connect(self._reset_controls); btns.addWidget(b)
        lay.addLayout(btns)

        dock.setWidget(w)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_measure_dock(self):
        dock = QDockWidget("测量 / 标定", self)
        w = QWidget(); lay = QVBoxLayout(w)

        # calibration
        gb = QGroupBox("标定 (标定片)")
        gl = QVBoxLayout(gb)
        self.cal_combo = QComboBox()
        self.cal_combo.currentTextChanged.connect(self._on_cal_selected)
        gl.addWidget(self.cal_combo)
        grid = QGridLayout()
        b = QPushButton("线性标定…"); b.setToolTip("一条已知长度的线（直尺/线段/方格边）")
        b.clicked.connect(self.new_calibration); grid.addWidget(b, 0, 0)
        b = QPushButton("矩形标定…"); b.setToolTip("点已知矩形/方格的4个角，纠正倾斜透视")
        b.clicked.connect(self.new_perspective); grid.addWidget(b, 0, 1)
        b = QPushButton("十字标定…"); b.setToolTip("沿X、Y轴各画一条，分别定X/Y比例(直角坐标十字)")
        b.clicked.connect(self.new_cross); grid.addWidget(b, 1, 0)
        b = QPushButton("删除"); b.clicked.connect(self.delete_calibration); grid.addWidget(b, 1, 1)
        gl.addLayout(grid)
        lay.addWidget(gb)
        self._refresh_cal_combo()

        # tools
        gb2 = QGroupBox("测量工具")
        grid = QGridLayout(gb2)
        self.tool_btns = {}
        for i, t in enumerate(TOOLS):
            btn = QPushButton(TOOL_LABELS[t]); btn.setCheckable(True)
            btn.clicked.connect(lambda _, tt=t: self.set_tool(tt))
            grid.addWidget(btn, i // 2, i % 2)
            self.tool_btns[t] = btn
        lay.addWidget(gb2)
        self.set_tool("select")

        # style
        gb3 = QGroupBox("样式 / 单位")
        f = QFormLayout(gb3)
        self.color_btn = QPushButton(); self.color_btn.clicked.connect(self._pick_color)
        self._refresh_color_btn()
        f.addRow("线色:", self.color_btn)
        self.width_spin = QSpinBox(); self.width_spin.setRange(1, 10); self.width_spin.setValue(2)
        self.width_spin.valueChanged.connect(lambda v: setattr(self.view, "penw", v))
        f.addRow("线宽:", self.width_spin)
        self.unit_combo = QComboBox(); self.unit_combo.addItems(["µm", "mm"])
        self.unit_combo.setCurrentIndex(0 if self.view.unit == "um" else 1)
        self.unit_combo.currentIndexChanged.connect(self._set_unit)
        f.addRow("单位:", self.unit_combo)
        lay.addWidget(gb3)

        # item list
        self.item_list = QListWidget()
        self.item_list.currentRowChanged.connect(self._list_select)
        lay.addWidget(QLabel("测量列表:"))
        lay.addWidget(self.item_list, 1)
        r = QHBoxLayout()
        b = QPushButton("删除选中"); b.clicked.connect(self._del_sel); r.addWidget(b)
        b = QPushButton("删最后"); b.clicked.connect(self._del_last); r.addWidget(b)
        b = QPushButton("清空"); b.clicked.connect(self._clear); r.addWidget(b)
        lay.addLayout(r)

        dock.setWidget(w)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    # ---------- camera control
    def _update_open_btn(self):
        running = bool(self.thread and self.thread.running)
        sp = QStyle.StandardPixmap
        if running:
            self.btn_open.setText("  关闭相机  ")
            self.btn_open.setIcon(self._icon(sp.SP_MediaStop))
            self.btn_open.setStyleSheet(
                "QPushButton{background:#c0392b;color:white;font-weight:bold;"
                "padding:4px 12px;border:none;border-radius:4px;}"
                "QPushButton:hover{background:#d9503f;}")
        else:
            self.btn_open.setText("  打开相机  ")
            self.btn_open.setIcon(self._icon(sp.SP_MediaPlay))
            self.btn_open.setStyleSheet(
                "QPushButton{background:#2faa4f;color:white;font-weight:bold;"
                "padding:4px 12px;border:none;border-radius:4px;}"
                "QPushButton:hover{background:#38c25b;}")

    def _toggle_camera(self):
        if self.thread and self.thread.running:
            self._stop_camera()
        else:
            self._start_camera()
        self._update_open_btn()

    def _current_device_path(self):
        return self.dev_combo.currentData()

    def _start_camera(self):
        path = self._current_device_path()
        if not path:
            QMessageBox.warning(self, APP_NAME, "没有可用相机设备。")
            return
        if self.res_combo.count() == 0:
            return
        w, h = self.res_combo.currentData()
        self._stop_camera()
        self.thread = VideoThread(device_node_index(path), w, h)
        self.thread.flip_h = self.act_fliph.isChecked()
        self.thread.flip_v = self.act_flipv.isChecked()
        self.thread.frame_ready.connect(self.view.set_frame)
        self.thread.error.connect(lambda m: QMessageBox.critical(self, APP_NAME, m))
        self.thread.start()
        self.cfg["device"] = path
        save_json(CFG_FILE, self.cfg)

    def _stop_camera(self):
        if self.thread:
            try:
                self.thread.frame_ready.disconnect()
            except Exception:
                pass
            self.thread.stop()
            self.thread = None

    def _on_device_changed(self):
        path = self._current_device_path()
        if not path:
            return
        self.resolutions = list_resolutions(path)
        self.res_combo.blockSignals(True)
        self.res_combo.clear()
        for (w, h) in self.resolutions:
            self.res_combo.addItem(f"{w} × {h}", (w, h))
        # 选上次用的分辨率
        last = self.cfg.get("res")
        if last:
            for i, (w, h) in enumerate(self.resolutions):
                if [w, h] == last:
                    self.res_combo.setCurrentIndex(i)
        self.res_combo.blockSignals(False)
        self._build_controls(path)
        if self.thread and self.thread.running:
            self._start_camera()

    def _on_res_changed(self):
        if self.res_combo.currentData():
            self.cfg["res"] = list(self.res_combo.currentData())
            save_json(CFG_FILE, self.cfg)
        if self.thread and self.thread.running:
            self._start_camera()
        self._update_cal_status()

    def _build_controls(self, path):
        # 清空
        while self.ctrl_form.rowCount():
            self.ctrl_form.removeRow(0)
        self._ctrl_widgets = {}
        for c in list_controls(path):
            name = c["name"]
            if c["type"] == "int" and "min" in c:
                s = QSlider(Qt.Orientation.Horizontal)
                s.setRange(c["min"], c["max"]); s.setValue(c.get("value", c.get("default", c["min"])))
                s.setEnabled(not c.get("inactive", False))
                s.valueChanged.connect(lambda v, n=name: set_control(path, n, v))
                self.ctrl_form.addRow(self._nice(name), s)
                self._ctrl_widgets[name] = ("int", s, c.get("default"))
            elif c["type"] == "bool":
                cb = QCheckBox(); cb.setChecked(bool(c.get("value", c.get("default", 0))))
                cb.toggled.connect(lambda v, n=name: self._on_bool(path, n, v))
                self.ctrl_form.addRow(self._nice(name), cb)
                self._ctrl_widgets[name] = ("bool", cb, c.get("default"))
            elif c["type"] == "menu":
                cb = QComboBox()
                for val, txt in sorted(c["menu"].items()):
                    cb.addItem(txt, val)
                idx = cb.findData(c.get("value", c.get("default", 0)))
                if idx >= 0:
                    cb.setCurrentIndex(idx)
                cb.currentIndexChanged.connect(lambda _, n=name, w=cb: set_control(path, n, w.currentData()))
                self.ctrl_form.addRow(self._nice(name), cb)
                self._ctrl_widgets[name] = ("menu", cb, c.get("default"))

    def _on_bool(self, path, name, v):
        set_control(path, name, 1 if v else 0)
        QTimer.singleShot(200, lambda: self._build_controls(path))  # 刷新 inactive 状态

    def _reset_controls(self):
        path = self._current_device_path()
        for name, (typ, w, default) in getattr(self, "_ctrl_widgets", {}).items():
            if default is None:
                continue
            set_control(path, name, default)
        self._build_controls(path)

    _NICE = {"brightness": "亮度", "contrast": "对比度", "saturation": "饱和度",
             "hue": "色调", "gamma": "Gamma", "gain": "增益", "sharpness": "锐度",
             "white_balance_automatic": "自动白平衡", "white_balance_temperature": "色温",
             "power_line_frequency": "抗闪烁", "backlight_compensation": "背光补偿",
             "auto_exposure": "自动曝光", "exposure_time_absolute": "曝光时间",
             "exposure_dynamic_framerate": "动态帧率"}

    def _nice(self, name):
        return self._NICE.get(name, name)

    # ---------- flip / overlay / zoom
    def _apply_flip(self):
        if self.thread:
            self.thread.flip_h = self.act_fliph.isChecked()
            self.thread.flip_v = self.act_flipv.isChecked()

    def _set_overlay(self, which, on):
        if which == "grid":
            self.view.show_grid = on
        else:
            self.view.show_cross = on
        self.view.update()

    def _zoom(self, factor):
        self.view.fit = False
        self.view.zoom = max(0.1, min(10, self.view.zoom * factor))
        self.view.update()

    def _zoom_fit(self):
        self.view.fit = True; self.view.zoom = 1.0; self.view.update()

    def toggle_freeze(self, on):
        self.view.frozen = on
        self.act_freeze.setText("解冻" if on else "冻结")

    # ---------- capture
    def _timestamp(self):
        return time.strftime("%Y%m%d_%H%M%S")

    def snapshot(self):
        if self.view.frame is None:
            self.statusBar().showMessage("未拍照：请先『打开相机』", 3000)
            return
        ext = self.snap_format
        path = os.path.join(self.save_dir, f"IMG_{self._timestamp()}.{ext}")
        if self.snap_annotate and self.view.items:
            img = self._render_annotated()
        else:
            img = self.view.frame.copy()
        params = []
        if ext in ("jpg", "jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        ok = cv2.imwrite(path, img, params)
        if ok:
            self.view.flash(f"已保存 ✓  {os.path.basename(path)}")
            self.statusBar().showMessage(f"已保存: {path}", 5000)
            self._last_saved = path
        else:
            self.view.flash("保存失败")
            self.statusBar().showMessage(f"保存失败: {path}", 5000)

    def _render_annotated(self):
        """把测量结果烧录到全分辨率图片上导出。"""
        frame = self.view.frame.copy()
        if not self.view.items:
            return frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()
        qp = QPainter(qimg); qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        m = self.metric_mapper()
        sc = max(1.0, w / 1000.0)
        for it in self.view.items:
            self._burn_item(qp, it, m, sc)
        qp.end()
        # QImage(RGB888) -> numpy，按 bytesPerLine 处理行对齐
        bpl = qimg.bytesPerLine()
        ptr = qimg.constBits(); ptr.setsize(h * bpl)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, bpl))[:, :w*3].reshape((h, w, 3))
        return cv2.cvtColor(arr.copy(), cv2.COLOR_RGB2BGR)

    def _burn_item(self, qp, it, m, sc):
        pen = QPen(QColor(*it.color), int(it.width*sc)); qp.setPen(pen)
        pts = [QPointF(*p) for p in it.pts]
        k = it.kind
        if k in ("line", "hline", "vline") and len(pts) == 2:
            qp.drawLine(pts[0], pts[1])
        elif k == "angle" and len(pts) == 3:
            qp.drawLine(pts[1], pts[0]); qp.drawLine(pts[1], pts[2])
        elif k == "circle" and len(pts) == 2:
            r = dist(it.pts[0], it.pts[1]); qp.drawEllipse(pts[0], r, r); qp.drawLine(pts[0], pts[1])
        elif k == "circle3" and len(it.pts) == 3:
            c, r = circle_from_3(*it.pts)
            if c:
                qp.drawEllipse(QPointF(*c), r, r)
        elif k == "rect" and len(pts) == 2:
            qp.drawRect(QRectF(pts[0], pts[1]).normalized())
        elif k == "polygon" and len(pts) >= 2:
            qp.drawPolygon(QPolygonF(pts))
        elif k in ("point", "text") and pts:
            p = pts[0]
            qp.drawLine(QPointF(p.x()-8*sc, p.y()), QPointF(p.x()+8*sc, p.y()))
            qp.drawLine(QPointF(p.x(), p.y()-8*sc), QPointF(p.x(), p.y()+8*sc))
        # label
        f = QFont("Sans", int(10*sc)); f.setBold(True); qp.setFont(f)
        a = it.anchor()
        qp.drawText(QPointF(a[0]+6*sc, a[1]-6*sc), it.measure(m, self.view.unit))

    def toggle_record(self):
        if not self.thread or not self.thread.running:
            QMessageBox.information(self, APP_NAME, "请先打开相机。")
            return
        if not self.recording:
            path = os.path.join(self.save_dir, f"VID_{self._timestamp()}.mp4")
            if self.thread.start_recording(path, fps=20):
                self.recording = True
                self.act_rec.setText("停止录像")
                self.act_pause.setEnabled(True)
                self._rec_path = path
                self.statusBar().showMessage(f"录像中: {path}")
            else:
                QMessageBox.critical(self, APP_NAME, "无法开始录像。")
        else:
            self.thread.stop_recording()
            self.recording = False; self.rec_paused = False
            self.act_rec.setText("录像"); self.act_pause.setEnabled(False)
            self.act_pause.setText("暂停录像")
            self.statusBar().showMessage(f"已保存: {self._rec_path}", 4000)

    def pause_record(self):
        if not self.recording:
            return
        self.rec_paused = not self.rec_paused
        self.thread.pause_recording(self.rec_paused)
        self.act_pause.setText("继续录像" if self.rec_paused else "暂停录像")

    def choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择保存目录", self.save_dir)
        if d:
            self.save_dir = d; self.cfg["save_dir"] = d; save_json(CFG_FILE, self.cfg)

    # ---------- calibration ui
    def new_calibration(self):
        if self.view.frame is None:
            QMessageBox.information(self, APP_NAME, "请先打开相机并对准标定片。")
            return
        was = self.view.frozen
        self.view.frozen = True
        dlg = CalibrateDialog(self, self.view.frame.copy())
        if dlg.exec() == QDialog.DialogCode.Accepted:
            cal = dlg.result_calib()
            if cal:
                self.calib.add(cal)
                self._refresh_cal_combo()
                self._update_cal_status()
                self.statusBar().showMessage(
                    f"标定『{cal.name}』: {cal.um_per_px:.4f} µm/px @ {cal.ref_width}px 宽", 5000)
            else:
                QMessageBox.warning(self, APP_NAME, "没有画线或线太短。")
        self.view.frozen = was
        self.act_freeze.setChecked(was)

    def new_perspective(self):
        if self.view.frame is None:
            QMessageBox.information(self, APP_NAME, "请先打开相机并对准参照矩形。")
            return
        was = self.view.frozen
        self.view.frozen = True
        dlg = PerspectiveDialog(self, self.view.frame.copy())
        if dlg.exec() == QDialog.DialogCode.Accepted:
            cal = dlg.result_calib()
            if cal:
                self.calib.add(cal)
                self._refresh_cal_combo(); self._update_cal_status(); self.view.update()
                self.statusBar().showMessage(f"透视标定『{cal.name}』已创建", 5000)
            else:
                QMessageBox.warning(self, APP_NAME, "需要点满 4 个角。")
        self.view.frozen = was
        self.act_freeze.setChecked(was)

    def new_cross(self):
        if self.view.frame is None:
            QMessageBox.information(self, APP_NAME, "请先打开相机并对准直角坐标十字。")
            return
        was = self.view.frozen
        self.view.frozen = True
        dlg = CrossDialog(self, self.view.frame.copy())
        if dlg.exec() == QDialog.DialogCode.Accepted:
            cal = dlg.result_calib()
            if cal:
                self.calib.add(cal)
                self._refresh_cal_combo(); self._update_cal_status(); self.view.update()
                self.statusBar().showMessage(
                    f"十字标定『{cal.name}』: X={cal.um_per_px:.4f}  Y={cal.um_per_px_y:.4f} µm/px", 5000)
            else:
                QMessageBox.warning(self, APP_NAME, "需要 X、Y 两条线都画好。")
        self.view.frozen = was
        self.act_freeze.setChecked(was)

    def delete_calibration(self):
        name = self.cal_combo.currentText()
        if name and name != "(无标定 / 像素)":
            self.calib.remove(name)
            self._refresh_cal_combo(); self._update_cal_status()

    def _refresh_cal_combo(self):
        self.cal_combo.blockSignals(True)
        self.cal_combo.clear()
        self.cal_combo.addItem("(无标定 / 像素)")
        for c in self.calib.items:
            self.cal_combo.addItem(c.name)
        if self.calib.active:
            i = self.cal_combo.findText(self.calib.active)
            if i >= 0:
                self.cal_combo.setCurrentIndex(i)
        self.cal_combo.blockSignals(False)

    def _on_cal_selected(self, name):
        self.calib.active = None if name.startswith("(无") else name
        self.calib.save()
        self._update_cal_status()
        self.view.update()

    def _cur_width(self):
        if self.view.frame_size()[0]:
            return self.view.frame_size()[0]
        d = self.res_combo.currentData()
        return d[0] if d else 0

    def metric_mapper(self):
        """当前有效的 像素->µm 映射(线性或透视)，无标定返回 None。"""
        cal = self.calib.get_active()
        w = self._cur_width()
        if cal is None or not w:
            return None
        return cal.mapper(w)

    def _update_cal_status(self):
        cal = self.calib.get_active()
        w = self._cur_width()
        if cal and w:
            if cal.kind == "perspective":
                self._status_cal.setText(f"标定: {cal.name} [矩形/透视] @ {w}px")
            elif cal.kind == "xy":
                rsc = cal.ref_width / w if cal.ref_width else 1.0
                self._status_cal.setText(
                    f"标定: {cal.name} [十字] X={cal.um_per_px*rsc:.4f} Y={cal.um_per_px_y*rsc:.4f} µm/px")
            else:
                self._status_cal.setText(f"标定: {cal.name}  {cal.effective(w):.4f} µm/px @ {w}px")
        else:
            self._status_cal.setText("标定: 无 (显示像素)")

    # ---------- tools / style
    def set_tool(self, t):
        self.view.tool = t
        self.view.pending = []
        for name, btn in self.tool_btns.items():
            btn.setChecked(name == t)

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(*self.view.color), self, "选择线色")
        if c.isValid():
            self.view.color = (c.red(), c.green(), c.blue())
            self._refresh_color_btn()

    def _refresh_color_btn(self):
        r, g, b = self.view.color
        self.color_btn.setStyleSheet(f"background:rgb({r},{g},{b}); min-width:40px;")

    def _set_unit(self, idx):
        self.view.unit = "um" if idx == 0 else "mm"
        self.cfg["unit"] = self.view.unit; save_json(CFG_FILE, self.cfg)
        self.view.update(); self._refresh_items()

    # ---------- item list
    def _refresh_items(self):
        self.item_list.blockSignals(True)
        self.item_list.clear()
        m = self.metric_mapper()
        for i, it in enumerate(self.view.items):
            self.item_list.addItem(f"{i+1}. {TOOL_LABELS[it.kind]}: {it.measure(m, self.view.unit)}")
        if 0 <= self.view.selected < self.item_list.count():
            self.item_list.setCurrentRow(self.view.selected)
        self.item_list.blockSignals(False)

    def _list_select(self, row):
        self.view.selected = row
        self.view.update()

    def _del_sel(self):
        i = self.view.selected
        if 0 <= i < len(self.view.items):
            del self.view.items[i]; self.view.selected = -1
            self.view.update(); self._refresh_items()

    def _del_last(self):
        if self.view.items:
            self.view.items.pop(); self.view.selected = -1
            self.view.update(); self._refresh_items()

    def _clear(self):
        self.view.items = []; self.view.selected = -1
        self.view.update(); self._refresh_items()

    # ---------- misc
    def _on_cursor(self, s):
        self.statusBar().showMessage(s)

    def _tick(self):
        self._frames = 0

    def closeEvent(self, ev):
        self._stop_camera()
        save_json(CFG_FILE, self.cfg)
        super().closeEvent(ev)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setDesktopFileName("unilab-linux")   # 让 Wayland/GNOME 关联到 .desktop 的图标
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unilab.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    win = MainWindow()
    if os.path.exists(icon_path):
        win.setWindowIcon(QIcon(icon_path))
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
