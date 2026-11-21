"""透视相机与 AABB 视锥剔除核心实现。

相机位于原点，朝 -Z 方向观察，+Y 向上。投影矩阵在构造时计算并缓存，
之后所有状态只读，因此并发投影/剔除安全，且重复投影结果逐位一致。
"""

from __future__ import annotations

import math

try:
    import numpy as _np
except ImportError:  # pragma: no cover - numpy 不可用时退化为纯 Python
    _np = None

__all__ = ["Camera"]


def _finite_float(name, value):
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限数值, 得到 {value!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name} 必须是有限数值, 得到 {value!r}")
    return v


class Camera:
    """透视相机：近远裁剪、NDC 投影与 AABB/点视锥剔除。"""

    __slots__ = (
        "_fov", "_near", "_far", "_aspect",
        "_tan_half", "_f", "_f_over_aspect",
        "_matrix", "_planes",
    )

    def __init__(self, fov, near, far, aspect=1.0):
        fov = _finite_float("fov", fov)
        near = _finite_float("near", near)
        far = _finite_float("far", far)
        aspect = _finite_float("aspect", aspect)
        if not 0.0 < fov < 180.0:
            raise ValueError(f"fov 必须在 (0, 180) 开区间内, 得到 {fov!r}")
        if near <= 0.0:
            raise ValueError(f"near 必须为正数, 得到 {near!r}")
        if far <= near:
            raise ValueError(f"far 必须大于 near, 得到 near={near!r}, far={far!r}")
        if aspect <= 0.0:
            raise ValueError(f"aspect 必须为正数, 得到 {aspect!r}")

        self._fov = fov
        self._near = near
        self._far = far
        self._aspect = aspect
        self._tan_half = math.tan(math.radians(fov) * 0.5)
        self._f = 1.0 / self._tan_half
        self._f_over_aspect = self._f / aspect
        # 标准右手系透视投影矩阵（行主序 4x4 元组），构造时缓存，之后只读。
        self._matrix = (
            (self._f_over_aspect, 0.0, 0.0, 0.0),
            (0.0, self._f, 0.0, 0.0),
            (0.0, 0.0, (far + near) / (near - far),
             2.0 * far * near / (near - far)),
            (0.0, 0.0, -1.0, 0.0),
        )
        t = self._tan_half
        ta = t * aspect
        # 六个视锥平面：normal·p + d >= 0 表示在内侧（法线无需归一化）。
        self._planes = (
            ((0.0, 0.0, -1.0), -near),   # near: -z >= near
            ((0.0, 0.0, 1.0), far),      # far:   z >= -far
            ((-1.0, 0.0, -ta), 0.0),     # left
            ((1.0, 0.0, -ta), 0.0),      # right
            ((0.0, -1.0, -t), 0.0),      # top
            ((0.0, 1.0, -t), 0.0),       # bottom
        )

    # -- 只读属性 --------------------------------------------------------
    @property
    def fov(self):
        return self._fov

    @property
    def near(self):
        return self._near

    @property
    def far(self):
        return self._far

    @property
    def aspect(self):
        return self._aspect

    @property
    def projection_matrix(self):
        """缓存的 4x4 透视投影矩阵（元组的元组，行主序）。"""
        return self._matrix

    # -- 投影 ------------------------------------------------------------
    def project(self, point):
        """把相机坐标系下的点投影到 NDC，返回 (x, y) 元组。

        点在近裁剪面之前、远裁剪面之后或相机背后时返回 None；
        坐标维度不为 3 或含非有限值时抛出 ValueError。
        """
        try:
            x, y, z = point
        except (TypeError, ValueError) as exc:
            raise ValueError(f"point 必须是三元坐标, 得到 {point!r}") from exc
        x = _finite_float("x", x)
        y = _finite_float("y", y)
        z = _finite_float("z", z)
        w = -z
        if w < self._near or w > self._far:
            return None
        inv_w = 1.0 / w
        return (x * self._f_over_aspect * inv_w, y * self._f * inv_w)

    # -- 视锥剔除 --------------------------------------------------------
    def cull(self, items):
        """对视锥剔除，返回可见项的下标列表（int）。

        每项可以是：
        - 三元点 ``(x, y, z)``；
        - 六元 AABB ``(minx, miny, minz, maxx, maxy, maxz)``；
        - 二元组 ``((minx, miny, minz), (maxx, maxy, maxz))``。

        与视锥相交即视为可见。坐标含非有限值时抛出 ValueError。
        本方法不修改任何共享状态，可安全并发调用。
        """
        if items is None:
            raise ValueError("items 不能为 None")
        if _np is not None:
            if isinstance(items, _np.ndarray):
                return self._cull_array(items)
            if not isinstance(items, list):
                items = list(items)
            if not items:
                return []
            arr = self._try_fast_array(items)
            if arr is not None:
                return self._cull_array(arr)
            return [i for i, it in enumerate(items) if self._visible_one(it)]
        items = list(items)
        return [i for i, it in enumerate(items) if self._visible_one(it)]

    # -- 内部实现 --------------------------------------------------------
    @staticmethod
    def _try_fast_array(items):
        """把同构的点/AABB 列表快速转成 (N, 3)/(N, 6) 数组，失败返回 None。"""
        import itertools
        try:
            width = len(items[0])
        except TypeError:
            return None
        if width == 2:
            try:
                arr = _np.asarray(items, dtype=float)
            except (TypeError, ValueError):
                return None
            return arr if arr.ndim == 3 and arr.shape[1:] == (2, 3) else None
        if width not in (3, 6) or len(items[-1]) != width:
            return None
        count = len(items) * width
        try:
            flat = _np.fromiter(
                itertools.chain.from_iterable(items), dtype=float, count=count)
        except (TypeError, ValueError):
            return None
        if flat.size != count:
            return None
        return flat.reshape(len(items), width)

    def _cull_array(self, arr):
        arr = _np.asarray(arr, dtype=float)
        if arr.ndim == 3 and arr.shape[1:] == (2, 3):
            arr = arr.reshape(arr.shape[0], 6)
        if arr.ndim != 2 or arr.shape[1] not in (3, 6):
            raise ValueError(
                f"每项必须是三元点或六元 AABB, 得到形状 {arr.shape!r}")
        if arr.shape[0] and not _np.isfinite(arr).all():
            raise ValueError("坐标必须全部为有限数值")
        # 相机位于原点朝 -Z，六个视锥平面只含轴向分量，
        # 因此剔除可全部用 (N,) 列运算完成（与 _planes 的平面测试等价）。
        near = self._near
        far = self._far
        t = self._tan_half
        ta = t * self._aspect
        if arr.shape[1] == 3:
            x = arr[:, 0]
            y = arr[:, 1]
            z = arr[:, 2]
            mask = (z <= -near) & (z >= -far)
            mask &= _np.abs(x) <= -ta * z
            mask &= _np.abs(y) <= -t * z
        else:
            minx = arr[:, 0]
            miny = arr[:, 1]
            minz = arr[:, 2]
            maxx = arr[:, 3]
            maxy = arr[:, 4]
            maxz = arr[:, 5]
            # 正顶点法：对每个平面取沿法线最远的顶点做剔除测试。
            mask = (minz <= -near) & (maxz >= -far)
            mask &= -minx - ta * minz >= 0.0
            mask &= maxx - ta * minz >= 0.0
            mask &= -miny - t * minz >= 0.0
            mask &= maxy - t * minz >= 0.0
        return mask.nonzero()[0].tolist()

    def _visible_one(self, item):
        try:
            n = len(item)
        except TypeError as exc:
            raise ValueError(f"剔除项必须是点或 AABB, 得到 {item!r}") from exc
        if n == 3:
            x, y, z = item
            return self._point_visible(
                _finite_float("x", x), _finite_float("y", y),
                _finite_float("z", z))
        if n == 6:
            vals = [_finite_float(f"aabb[{k}]", v) for k, v in enumerate(item)]
            return self._aabb_visible(vals[:3], vals[3:])
        if n == 2:
            lo, hi = item
            lx, ly, lz = lo
            hx, hy, hz = hi
            mins = [_finite_float("min", lx), _finite_float("min", ly),
                    _finite_float("min", lz)]
            maxs = [_finite_float("max", hx), _finite_float("max", hy),
                    _finite_float("max", hz)]
            return self._aabb_visible(mins, maxs)
        raise ValueError(f"剔除项必须是三元点或六元 AABB, 得到 {item!r}")

    def _point_visible(self, x, y, z):
        for (nx, ny, nz), d in self._planes:
            if nx * x + ny * y + nz * z + d < 0.0:
                return False
        return True

    def _aabb_visible(self, mins, maxs):
        for (nx, ny, nz), d in self._planes:
            px = maxs[0] if nx >= 0.0 else mins[0]
            py = maxs[1] if ny >= 0.0 else mins[1]
            pz = maxs[2] if nz >= 0.0 else mins[2]
            if nx * px + ny * py + nz * pz + d < 0.0:
                return False
        return True
