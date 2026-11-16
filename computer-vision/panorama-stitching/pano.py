# -*- coding: utf-8 -*-
"""
多张图片合成全景图 —— 手写全链路实现
=====================================

课程：Python 视觉识别
流程（每一步都能单独导出图，供报告使用）：

    1. 特征提取      SIFT / ORB
    2. 特征匹配      BFMatcher (knn k=2) + Lowe's ratio test
    3. 单应性估计    cv2.findHomography(RANSAC) 剔除误匹配
    4. 图像配准      以中间图为参考平面，沿左右两侧累乘单应矩阵
    5. 透视变换      cv2.warpPerspective 投影到统一画布
    6. 图像融合      距离变换羽化权重 + 加权平均
    7. 后处理        裁剪黑边

另附 OpenCV 高层 API（cv2.Stitcher）的结果作为对照基线。

用法：
    python pano.py --images data                 # 手写 + 官方 API 都跑
    python pano.py --images data --method manual # 只跑手写
    python pano.py --images data --detector orb --scale 1.0 --dump-parts
"""

import argparse
import glob
import json
import math
import os
import re
import time

import cv2
import numpy as np


# --------------------------------------------------------------------------
# 0. 基础工具
# --------------------------------------------------------------------------
def imread_unicode(path):
    """cv2.imread 不支持中文路径，用 np.fromfile + imdecode 绕开。"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"读图失败: {path}")
    return img


def imwrite_unicode(path, img):
    """同上，写图也走 imencode，保证中文/空格路径可用。"""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"编码失败: {path}")
    buf.tofile(path)


def cylindrical_project(img, f_ratio=1.0):
    """柱面投影：把图像投到以焦距 f 为半径的圆柱面上展开。

    相机绕光心旋转拍摄时，平面投影下相邻图像会呈现明显的尺度差异
    （越靠边被拉伸越厉害），导致单应矩阵解出虚假的缩放并在累乘中
    不断放大。柱面投影把旋转转成近似纯平移，尺度自然归一。
    """
    h, w = img.shape[:2]
    f = max(f_ratio * w, 1.0)
    w_out = int(round(2 * f * math.atan(w / (2 * f))))
    h_out = h
    xs, ys = np.meshgrid(np.arange(w_out, dtype=np.float32),
                         np.arange(h_out, dtype=np.float32))
    theta = (xs - w_out / 2.0) / f
    map_x = f * np.tan(theta) + w / 2.0
    map_y = (ys - h_out / 2.0) / np.cos(theta) + h / 2.0
    out = cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0))
    # 采样坐标落在原图之外的像素是无效的，单独记一份掩码，
    # 否则投影产生的黑边会被后续当成真实画面参与拼接
    inside = ((map_x >= 0) & (map_x < w - 1) & (map_y >= 0) & (map_y < h - 1))
    valid = np.where(inside, 255, 0).astype(np.uint8)
    return out, valid


def natural_key(path):
    """自然序：1.jpg < 2.jpg < 10.jpg（字符串排序会把 10 排到 2 前面）。"""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def collect_images(source):
    """支持传文件夹、或用逗号分隔的多个文件，按自然序排列。"""
    if "," in source:
        files = [s.strip().strip('"') for s in source.split(",") if s.strip()]
    elif os.path.isdir(source):
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.bmp"):
            files.extend(glob.glob(os.path.join(source, ext)))
        files = sorted(set(files), key=natural_key)
    else:
        files = [source]
    if len(files) < 2:
        raise ValueError(f"至少需要 2 张图，当前只找到 {len(files)} 张")
    return files


def resize_keep_ratio(img, scale):
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def corners_of(h, w):
    """图像四角坐标，用于计算全景画布边界。"""
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)


def transform_scale(H):
    """从 3x3 变换矩阵的线性部分估算整体缩放倍数。"""
    det = np.linalg.det(H[:2, :2])
    return float(np.sqrt(abs(det))) if np.isfinite(det) else float("nan")


def largest_inner_rect(mask, max_side=512):
    """求二值掩码内面积最大的全 1 矩形，返回 (x0, y0, x1, y1)。

    用于裁掉拼接后参差不齐的边缘。在缩略图上用单调栈求解，
    再把比例映射回原尺寸，兼顾速度和精度。
    """
    h, w = mask.shape
    step = max(1, int(np.ceil(max(h, w) / max_side)))
    small = (mask[::step, ::step] > 0).astype(np.int32)
    sh, sw = small.shape

    heights = np.zeros(sw, dtype=np.int32)
    best_area, best = 0, (0, 0, sw, sh)
    for y in range(sh):
        heights = np.where(small[y] > 0, heights + 1, 0)
        stack = []                       # (起始列, 高度)
        for x in range(sw + 1):
            cur = heights[x] if x < sw else 0
            start = x
            while stack and stack[-1][1] >= cur:
                sx, sh_ = stack.pop()
                area = sh_ * (x - sx)
                if area > best_area:
                    best_area = area
                    best = (sx, y - sh_ + 1, x, y + 1)
                start = sx
            stack.append((start, cur))

    x0, y0, x1, y1 = best
    return x0 * step, y0 * step, min(x1 * step, w), min(y1 * step, h)


# --------------------------------------------------------------------------
# 1-3. 特征提取 / 匹配 / 单应性估计
# --------------------------------------------------------------------------
class PairMatcher:
    """负责单对图像之间的 特征 -> 匹配 -> 单应矩阵 全过程。"""

    def __init__(self, detector="sift", nfeatures=4000, ratio=0.75, ransac_thresh=5.0):
        self.detector_name = detector.lower()
        if self.detector_name == "sift":
            self.detector = cv2.SIFT_create(nfeatures=nfeatures)
            self.norm = cv2.NORM_L2
        elif self.detector_name == "orb":
            self.detector = cv2.ORB_create(nfeatures=nfeatures)
            self.norm = cv2.NORM_HAMMING
        else:
            raise ValueError("detector 只能是 sift 或 orb")
        self.ratio = ratio
        self.ransac_thresh = ransac_thresh

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, des = self.detector.detectAndCompute(gray, None)
        return kps, des

    def match(self, des1, des2):
        """knn 匹配 + Lowe's ratio test，保留"明显优于次优"的匹配。"""
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []
        bf = cv2.BFMatcher(self.norm)
        knn = bf.knnMatch(des1, des2, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)
        return good

    def estimate(self, kp1, kp2, good, flip=False, model="auto"):
        """求解两图之间的变换矩阵。

        model:
            homography  8 自由度透视变换，最通用但内点分布病态时会过拟合
            affine      6 自由度仿射
            similarity  4 自由度相似变换（旋转+等比缩放+平移），水平旋转拍摄最稳
            auto        先按 8 自由度求，尺度/正交性异常则自动降级为 similarity

        返回 (H 3x3, 内点掩码, 内点数, 实际使用的模型名)
        """
        if len(good) < 4:
            return None, None, 0, "none"
        if flip:
            src = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        else:
            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        if model == "similarity":
            return self._estimate_low_dof(src, dst, partial=True)
        if model == "affine":
            return self._estimate_low_dof(src, dst, partial=False)

        H8, mask8 = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
        if H8 is None:
            return None, None, 0, "failed"
        mask8 = mask8.ravel().astype(bool)

        if model == "homography":
            return self._refine_homography(src, dst, H8), mask8, int(mask8.sum()), "homography"

        # auto：检验 8 自由度解是否可信
        if self._is_well_conditioned(H8):
            return self._refine_homography(src, dst, H8), mask8, int(mask8.sum()), "homography"
        return self._estimate_low_dof(src, dst, partial=True)

    def _refine_homography(self, src, dst, H, iters=2):
        """RANSAC 只保证鲁棒，不保证精度；用内点做最小二乘精化，进一步压低重投影误差。"""
        for _ in range(iters):
            proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
            err = np.linalg.norm(proj - dst.reshape(-1, 2), axis=1)
            keep = err < self.ransac_thresh
            if keep.sum() < 6:
                break
            H_ref, _ = cv2.findHomography(src[keep], dst[keep], 0)
            if H_ref is None:
                break
            H = H_ref
        return H

    def _estimate_low_dof(self, src, dst, partial=True):
        if partial:
            M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                 ransacReprojThreshold=self.ransac_thresh)
            name = "similarity"
        else:
            M, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                          ransacReprojThreshold=self.ransac_thresh)
            name = "affine"
        if M is None:
            return None, None, 0, "failed"
        mask = inl.ravel().astype(bool) if inl is not None else np.ones(len(src), bool)
        # 用内点再做一轮最小二乘，把精度榨出来
        if mask.sum() >= 6:
            if partial:
                M2, _ = cv2.estimateAffinePartial2D(src[mask], dst[mask], method=cv2.LMEDS)
            else:
                M2, _ = cv2.estimateAffine2D(src[mask], dst[mask], method=cv2.LMEDS)
            if M2 is not None:
                M = M2
        H = np.vstack([M, [0, 0, 1]]).astype(np.float64)
        return H, mask, int(mask.sum()), name

    @staticmethod
    def _is_well_conditioned(H, scale_tol=0.15, ortho_tol=1.25):
        """判断单应矩阵的线性部分是否符合"相机绕光心旋转"的物理预期。

        绕光心水平旋转时，H 的 2x2 子矩阵应近似一个旋转矩阵：
        尺度接近 1、两列向量正交且等长。若明显偏离，说明求解被病态
        的内点分布带偏了（透视自由度在局部点集上过拟合）。
        """
        A = H[:2, :2]
        if not np.isfinite(A).all():
            return False
        det = np.linalg.det(A)
        if det <= 0:                       # 行列式为负 = 镜像，不可能是真实旋转
            return False
        scale = np.sqrt(det)
        c1, c2 = A[:, 0], A[:, 1]
        n1, n2 = np.linalg.norm(c1), np.linalg.norm(c2)
        len_ratio = max(n1, n2) / max(min(n1, n2), 1e-9)
        cos_angle = abs(float(np.dot(c1, c2)) / max(n1 * n2, 1e-9))
        return (abs(scale - 1.0) <= scale_tol
                and len_ratio <= ortho_tol
                and cos_angle <= 0.15)



def estimate_pair_gain(img1, img2, kp1, kp2, good, mask, radius=2):
    """用内点邻域的均值比估计两张图的亮度增益，返回 BGR 三通道中值。

    教室这类室内场景相机自动曝光会逐张变化，不校正的话接缝处会出现
    明显的明暗台阶。
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    ratios = []
    for m, ok in zip(good, mask):
        if not ok:
            continue
        x1, y1 = int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])
        x2, y2 = int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1])
        if not (radius <= x1 < w1 - radius and radius <= y1 < h1 - radius):
            continue
        if not (radius <= x2 < w2 - radius and radius <= y2 < h2 - radius):
            continue
        a = img1[y1 - radius:y1 + radius + 1, x1 - radius:x1 + radius + 1].reshape(-1, 3).mean(axis=0)
        b = img2[y2 - radius:y2 + radius + 1, x2 - radius:x2 + radius + 1].reshape(-1, 3).mean(axis=0)
        ratios.append(b / np.maximum(a, 1.0))
    if len(ratios) < 8:
        return np.ones(3, dtype=np.float32)
    return np.median(np.array(ratios, dtype=np.float32), axis=0)


def draw_match_visual(img1, kp1, img2, kp2, good, inlier_mask, path, max_draw=60):
    """把匹配结果画成连线图：内点绿线、外点红线，报告直接可用。"""
    inlier_matches, outlier_matches = [], []
    for i, m in enumerate(good):
        (inlier_matches if inlier_mask[i] else outlier_matches).append(m)

    inlier_matches = sorted(inlier_matches, key=lambda m: m.distance)[:max_draw]
    outlier_matches = sorted(outlier_matches, key=lambda m: m.distance)[:max_draw]
    vis = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None,
                          matchColor=(0, 220, 0), singlePointColor=None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if outlier_matches:
        vis = cv2.drawMatches(img1, kp1, img2, kp2, outlier_matches, vis,
                              matchColor=(0, 0, 255), singlePointColor=None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    imwrite_unicode(path, vis)


# --------------------------------------------------------------------------
# 4-7. 全景拼接主流程
# --------------------------------------------------------------------------
def build_panorama(images, img_masks, names, args, out_dir):
    t_start = time.time()
    matcher = PairMatcher(args.detector, args.nfeatures, args.ratio, args.ransac)
    n = len(images)

    print(f"\n[1] 特征提取 ({matcher.detector_name.upper()}, nfeatures={args.nfeatures})")
    feats = []
    for img, name in zip(images, names):
        kps, des = matcher.detect(img)
        feats.append((kps, des))
        print(f"    {name:<24} keypoints={len(kps):>6}")

    print("\n[2-3] 相邻图像匹配 + RANSAC 变换估计")
    raw_pairs = []
    for i in range(n - 1):
        kp1, des1 = feats[i]
        kp2, des2 = feats[i + 1]
        good = matcher.match(des1, des2)
        if len(good) < 4:
            raise RuntimeError(
                f"{names[i]} 与 {names[i+1]} 匹配点只有 {len(good)} 个，重叠区域可能不足")
        raw_pairs.append({"i": i, "good": good})

    def solve_all(model_name):
        """对每一对相邻图像求双向变换矩阵。"""
        solved = []
        for p in raw_pairs:
            i = p["i"]
            kp1, _ = feats[i]
            kp2, _ = feats[i + 1]
            H_fwd, mask, inl, used = matcher.estimate(kp1, kp2, p["good"],
                                                      model=model_name)
            H_bwd, _, _, _ = matcher.estimate(kp1, kp2, p["good"], flip=True,
                                              model=model_name)
            if H_fwd is None or H_bwd is None:
                raise RuntimeError(f"{names[i]} 与 {names[i+1]} 变换矩阵求解失败")
            solved.append({"i": i, "H_fwd": H_fwd, "H_bwd": H_bwd,
                           "mask": mask, "inliers": inl, "model": used})
        return solved

    pairs = solve_all(args.model)
    # auto 模式下只要有一对判为病态，就整体统一降级，避免不同自由度混用导致尺度不一致
    if args.model == "auto" and any(p["model"] != "homography" for p in pairs):
        print("    [!] 检测到病态单应（尺度偏离 1 / 线性部分不正交），整体降级为 similarity")
        pairs = solve_all("similarity")

    for p in pairs:
        i = p["i"]
        kp1, _ = feats[i]
        kp2, _ = feats[i + 1]
        rate = p["inliers"] / max(len(raw_pairs[i]["good"]), 1) * 100
        sc = transform_scale(p["H_fwd"])
        p["scale"] = sc
        p["good"] = raw_pairs[i]["good"]
        print(f"    {names[i]:<13} -> {names[i+1]:<13} "
              f"good={len(p['good']):>5}  inliers={p['inliers']:>5} ({rate:5.1f}%)  "
              f"model={p['model']:<10} scale={sc:.3f}")
        if args.dump_parts:
            draw_match_visual(images[i], kp1, images[i + 1], kp2, p["good"], p["mask"],
                              os.path.join(out_dir, f"matches_{i:02d}.jpg"))

    # ---- 以中间图为参考平面，左右两侧累乘变换矩阵 ----
    mid = n // 2
    Hs = [None] * n
    Hs[mid] = np.eye(3, dtype=np.float64)
    for i in range(mid, n - 1):                    # 向右：H_{i+1 -> mid}
        Hs[i + 1] = Hs[i] @ pairs[i]["H_bwd"]
    for i in range(mid - 1, -1, -1):               # 向左：H_{i -> mid}
        Hs[i] = Hs[i + 1] @ pairs[i]["H_fwd"]
    print(f"\n[4] 参考平面取第 {mid + 1} 张（{names[mid]}），左右累乘变换矩阵完成")

    # ---- 曝光补偿：逐张对齐到参考图的亮度 ----
    gains = [np.ones(3, dtype=np.float32) for _ in images]
    if args.exposure:
        for i in range(mid, n - 1):
            g = estimate_pair_gain(images[i], images[i + 1], feats[i][0], feats[i + 1][0],
                                   pairs[i]["good"], pairs[i]["mask"])
            gains[i + 1] = gains[i] / np.maximum(g, 1e-3)
        for i in range(mid - 1, -1, -1):
            g = estimate_pair_gain(images[i], images[i + 1], feats[i][0], feats[i + 1][0],
                                   pairs[i]["good"], pairs[i]["mask"])
            gains[i] = gains[i + 1] * g
        images = [np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)
                  for img, g in zip(images, gains)]
        print("[*] 曝光补偿增益：" + "  ".join(
            f"{nm}:{g.mean():.3f}" for nm, g in zip(names, gains)))


    # ---- 计算全局画布 ----
    all_pts = []
    for img, H in zip(images, Hs):
        h, w = img.shape[:2]
        all_pts.append(cv2.perspectiveTransform(corners_of(h, w), H))
    all_pts = np.concatenate(all_pts, axis=0)
    x_min, y_min = np.floor(all_pts.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_pts.max(axis=0).ravel()).astype(int)
    canvas_w, canvas_h = x_max - x_min, y_max - y_min
    shift = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    print(f"[5] 画布尺寸 {canvas_w} x {canvas_h}，"
          f"总像素 {canvas_w * canvas_h / 1e6:.1f} M")

    # ---- 逐张 warp + 融合 ----
    print(f"[6] 透视变换 + 融合（{args.blend}）")
    warped_size = (canvas_w, canvas_h)
    weights = []                    # 每张图的有效权重（uint8 存储，省内存）
    if args.blend == "seam":
        # 第一遍：只算权重，用于确定每个像素"归哪张图"
        for idx, (img, H) in enumerate(zip(images, Hs)):
            H_full = shift @ H
            valid = cv2.warpPerspective(img_masks[idx], H_full,
                                        warped_size, flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            w = cv2.distanceTransform(valid, cv2.DIST_L2, 5)
            if w.max() > 0:
                w = (w / w.max() * 255).astype(np.uint8)
            else:
                w = w.astype(np.uint8)
            weights.append(w)
        w_stack = np.stack(weights, axis=0)
        label = w_stack.argmax(axis=0).astype(np.uint8)      # 每个像素归属的图号

        # 接缝线 = label 的边界；只在接缝两侧窄带内做过渡，其余硬切，杜绝鬼影
        edge = np.zeros(label.shape, dtype=np.uint8)
        edge[:, :-1] |= (label[:, :-1] != label[:, 1:]).astype(np.uint8)
        edge[:-1, :] |= (label[:-1, :] != label[1:, :]).astype(np.uint8)
        band = cv2.dilate(edge, np.ones((15, 15), np.uint8), iterations=1)
        band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 5.0)
        band = np.clip(band, 0, 1)[..., None]
        print(f"    接缝像素 {int(edge.sum())} 个，过渡带宽 {int((band > 0.01).sum())} 像素")

        num = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        den = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)
        for idx, (img, H) in enumerate(zip(images, Hs)):
            H_full = shift @ H
            warped = cv2.warpPerspective(img, H_full, warped_size,
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            alpha = (label == idx).astype(np.float32)[..., None]
            alpha = alpha * (1 - band) + band * (weights[idx].astype(np.float32)[..., None] / 255.0)
            num += warped.astype(np.float32) * alpha
            den += alpha
            if args.dump_parts:
                imwrite_unicode(os.path.join(out_dir, f"warped_{idx:02d}.jpg"), warped)
            print(f"    warp + seam: {names[idx]}")
        pano = np.clip(num / np.maximum(den, 1e-6), 0, 255).astype(np.uint8)
        wsum = den[..., 0]
    elif args.blend == "gcut":
        # GraphCut 接缝搜索：沿着重叠区里"颜色差异最小"的路径切分，
        # 让接缝主动避开幕布边缘、黑板框这类高对比结构。
        warped_list, mask_list = [], []
        for idx, (img, H) in enumerate(zip(images, Hs)):
            H_full = shift @ H
            warped = cv2.warpPerspective(img, H_full, warped_size, flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            valid = cv2.warpPerspective(img_masks[idx], H_full,
                                        warped_size, flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warped_list.append(warped)
            mask_list.append(valid)
            if args.dump_parts:
                imwrite_unicode(os.path.join(out_dir, f"warped_{idx:02d}.jpg"), warped)
            print(f"    warp: {names[idx]}")

        print("    运行 GraphCut 接缝搜索 ...")
        try:
            # GraphCut 的图结构按像素建节点，全画布直接跑会吃光内存，
            # 因此先缩小 4 倍求接缝，再把掩码放大回原尺寸。
            step = 4
            sw, sh = max(warped_size[0] // step, 1), max(warped_size[1] // step, 1)
            small_imgs = [cv2.resize(w, (sw, sh), interpolation=cv2.INTER_AREA)
                          for w in warped_list]
            small_masks = [cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST)
                           for m in mask_list]
            try:
                finder = cv2.detail_GraphCutSeamFinder("COST_COLOR")
                out_masks = finder.find(small_imgs, [(0, 0)] * len(small_imgs), small_masks)
            except Exception:
                # GraphCut 在部分 OpenCV 版本/掩码形状下会触发内部断言，
                # 退到动态规划接缝搜索，效果相当且更稳定
                finder = cv2.detail_DpSeamFinder("COLOR")
                out_masks = finder.find(small_imgs, [(0, 0)] * len(small_imgs), small_masks)
            if out_masks is None:                     # 某些版本原地修改
                out_masks = small_masks
            out_masks = [m.get() if hasattr(m, "get") else m for m in out_masks]
            mask_list = [cv2.resize(np.asarray(m), warped_size,
                                    interpolation=cv2.INTER_NEAREST) for m in out_masks]
            print(f"    接缝搜索完成（{type(finder).__name__}，1/{step} 分辨率求解）")
        except Exception as exc:                      # 环境不支持时回退
            print(f"    [!] GraphCut 不可用（{exc}），回退为几何分界")
            mask_list = [np.where(m > 0, 255, 0).astype(np.uint8) for m in mask_list]

        num = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        den = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)
        fnum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        fden = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)
        for idx, warped in enumerate(warped_list):
            alpha = (mask_list[idx] > 0).astype(np.float32)
            alpha = cv2.GaussianBlur(alpha, (0, 0), args.seam_soft)[..., None]   # 软化接缝
            num += warped.astype(np.float32) * alpha
            den += alpha
            # 同一批数据再算一份"宽羽化"结果，只取其低频
            soft = cv2.GaussianBlur((mask_list[idx] > 0).astype(np.float32), (0, 0), 12.0)[..., None]
            fnum += warped.astype(np.float32) * soft
            fden += soft
        pano = np.clip(num / np.maximum(den, 1e-6), 0, 255).astype(np.uint8)
        wsum = den[..., 0]

        if args.exposure:
            # 低频校正：接缝保证细节不重影，但两侧曝光仍有残余差异，
            # 表现为幕布上的一条亮度台阶。用宽羽化结果的低频替换掉即可。
            feather_img = fnum / np.maximum(fden, 1e-6)
            sigma = max(canvas_w, canvas_h) / 120.0
            low_seam = cv2.GaussianBlur(pano.astype(np.float32), (0, 0), sigma)
            low_feather = cv2.GaussianBlur(feather_img, (0, 0), sigma)
            pano = np.clip(pano.astype(np.float32) - low_seam + low_feather,
                           0, 255).astype(np.uint8)
            print(f"    低频校正完成（sigma={sigma:.1f}）")
    else:
        acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        wsum = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        for idx, (img, H) in enumerate(zip(images, Hs)):
            H_full = shift @ H
            warped = cv2.warpPerspective(img, H_full, warped_size,
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            valid = cv2.warpPerspective(img_masks[idx], H_full,
                                        warped_size, flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # 距离变换：越靠近图像中心权重越大，边缘趋近 0 —— 接缝自然过渡
            weight = cv2.distanceTransform(valid, cv2.DIST_L2, 5).astype(np.float32)
            acc += warped.astype(np.float32) * weight[..., None]
            wsum += weight
            if args.dump_parts:
                imwrite_unicode(os.path.join(out_dir, f"warped_{idx:02d}.jpg"), warped)
            print(f"    warp + feather: {names[idx]}")
        pano = np.clip(acc / np.maximum(wsum, 1e-6)[..., None], 0, 255).astype(np.uint8)
    print("[6] 融合完成")

    # ---- 裁掉无效边缘 ----
    if args.crop != "none":
        mask_all = (wsum > 1e-6).astype(np.uint8)
        if args.crop == "inner":
            x0, y0, x1, y1 = largest_inner_rect(mask_all)
        else:                                        # bbox：只按外接框裁
            nz = np.where(mask_all > 0)
            y0, y1 = nz[0].min(), nz[0].max() + 1
            x0, x1 = nz[1].min(), nz[1].max() + 1
        pano = pano[y0:y1, x0:x1]
        print(f"[7] 裁剪方式 {args.crop} -> {pano.shape[1]} x {pano.shape[0]}")

    elapsed = time.time() - t_start

    # ---- 报告数据 ----
    report = {
        "method": "manual (SIFT/ORB + ratio test + RANSAC + warpPerspective + feather blend)",
        "detector": matcher.detector_name,
        "nfeatures": args.nfeatures,
        "ratio": args.ratio,
        "ransac_thresh_px": args.ransac,
        "transform_model": args.model,
        "models_used": sorted({p["model"] for p in pairs}),
        "blend": args.blend,
        "exposure_compensation": bool(args.exposure),
        "scale": args.scale,
        "reference_image": names[mid],
        "canvas": [int(canvas_w), int(canvas_h)],
        "output_size": [int(pano.shape[1]), int(pano.shape[0])],
        "elapsed_sec": round(elapsed, 2),
        "images": [
            {
                "name": name,
                "size": [int(img.shape[1]), int(img.shape[0])],
                "keypoints": len(kps),
            }
            for name, img, (kps, _) in zip(names, images, feats)
        ],
        "pairs": [
            {
                "pair": f"{names[p['i']]} -> {names[p['i'] + 1]}",
                "good_matches": len(p["good"]),
                "ransac_inliers": p["inliers"],
                "inlier_ratio": round(p["inliers"] / max(len(p["good"]), 1), 4),
                "model": p["model"],
                "transform_scale": round(p["scale"], 4),
                "H": np.round(p["H_fwd"], 5).tolist(),
            }
            for p in pairs
        ],
    }
    return pano, report


def build_with_opencv_api(images):
    """OpenCV 高层 Stitcher，作为对照基线。"""
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    t0 = time.time()
    status, pano = stitcher.stitch(images)
    cost = time.time() - t0
    status_map = {
        cv2.Stitcher_OK: "OK",
        cv2.Stitcher_ERR_NEED_MORE_IMGS: "ERR_NEED_MORE_IMGS（重叠不足）",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "ERR_HOMOGRAPHY_EST_FAIL（单应估计失败）",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "ERR_CAMERA_PARAMS_ADJUST_FAIL（相机参数调整失败）",
    }
    return (pano if status == cv2.Stitcher_OK else None), status_map.get(status, str(status)), cost


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="多张图片合成全景图（手写全链路）")
    ap.add_argument("--images", default="data", help="图片文件夹，或单张图片路径")
    ap.add_argument("--output", default="output", help="输出目录")
    ap.add_argument("--method", default="both", choices=["manual", "opencv", "both"])
    ap.add_argument("--projection", default="cylindrical", choices=["cylindrical", "plane"],
                    help="cylindrical=柱面投影（默认，消除旋转带来的尺度漂移），plane=直接用平面投影")
    ap.add_argument("--focal", type=float, default=1.0,
                    help="柱面投影焦距，单位是图像宽度的倍数，手机常见 0.8~1.2")
    ap.add_argument("--detector", default="sift", choices=["sift", "orb"])
    ap.add_argument("--model", default="auto",
                    choices=["auto", "homography", "affine", "similarity"],
                    help="变换模型：auto 自动判断是否降级，homography 为经典 8 自由度单应")
    ap.add_argument("--nfeatures", type=int, default=4000)
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe's ratio，越小筛得越狠")
    ap.add_argument("--ransac", type=float, default=5.0, help="RANSAC 重投影阈值（像素）")
    ap.add_argument("--scale", type=float, default=0.5, help="处理前缩放，1.0 为原图")
    ap.add_argument("--crop", default="inner", choices=["inner", "bbox", "none"],
                    help="inner=取最大内接矩形（边缘最整齐，默认），bbox=外接框去黑边，none=不裁")
    ap.add_argument("--seam-soft", type=float, default=6.0,
                    help="接缝过渡半径（像素），越大越平滑但过大可能产生重影")
    ap.add_argument("--exposure", type=int, default=1, help="1=做曝光补偿，0=关闭")
    ap.add_argument("--blend", default="gcut", choices=["gcut", "seam", "feather"],
                    help="gcut=GraphCut颜色接缝（默认），seam=几何接缝+窄带过渡，feather=羽化加权平均")
    ap.add_argument("--dump-parts", action="store_true", help="导出匹配图/逐张 warp 图")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    files = collect_images(args.images)
    names = [os.path.basename(f) for f in files]
    print(f"共 {len(files)} 张输入图：{', '.join(names)}")

    images = [resize_keep_ratio(imread_unicode(f), args.scale) for f in files]
    # 统一分辨率：输入图尺寸不一致时，柱面投影的焦距会各算各的，导致尺度不一致
    base_w = max(im.shape[1] for im in images)
    if any(im.shape[1] != base_w for im in images):
        fixed = []
        for im, nm in zip(images, names):
            if im.shape[1] != base_w:
                s = base_w / im.shape[1]
                interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
                im = cv2.resize(im, (base_w, int(round(im.shape[0] * s))), interpolation=interp)
                print(f"    [i] {nm} 尺寸不一致，已缩放到 {im.shape[1]}x{im.shape[0]}")
            fixed.append(im)
        images = fixed
    print(f"处理分辨率：{images[0].shape[1]} x {images[0].shape[0]}")

    img_masks = [np.full(im.shape[:2], 255, dtype=np.uint8) for im in images]
    if args.projection == "cylindrical":
        projected = [cylindrical_project(im, args.focal) for im in images]
        images = [p[0] for p in projected]
        img_masks = [p[1] for p in projected]
        print(f"柱面投影：f={args.focal:.2f}·W，投影后 {images[0].shape[1]} x {images[0].shape[0]}")
        print(f"           有效区域占比 {img_masks[0].mean() / 255 * 100:.1f}%")

    summary = {"inputs": names, "scale": args.scale}

    if args.method in ("manual", "both"):
        pano, report = build_panorama(images, img_masks, names, args, args.output)
        pano_path = os.path.join(args.output, "pano_manual.jpg")
        imwrite_unicode(pano_path, pano)
        summary["manual"] = report
        summary["manual"]["output_file"] = pano_path
        print(f"\n[手写结果] {pano_path}  ({pano.shape[1]}x{pano.shape[0]})")

    if args.method in ("opencv", "both"):
        pano_api, status, cost = build_with_opencv_api(images)
        summary["opencv_stitcher"] = {"status": status, "elapsed_sec": round(cost, 2)}
        if pano_api is not None:
            api_path = os.path.join(args.output, "pano_opencv.jpg")
            imwrite_unicode(api_path, pano_api)
            summary["opencv_stitcher"]["output_file"] = api_path
            summary["opencv_stitcher"]["output_size"] = [int(pano_api.shape[1]), int(pano_api.shape[0])]
            print(f"[官方API] {api_path}  ({pano_api.shape[1]}x{pano_api.shape[0]}, {cost:.2f}s)")
        else:
            print(f"[官方API] 失败：{status}")

    report_path = os.path.join(args.output, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n报告数据：{report_path}")


if __name__ == "__main__":
    main()
