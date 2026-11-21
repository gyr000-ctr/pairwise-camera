"""camera 核心 API（初始未实现）。"""


class Camera:
    def __init__(self, fov, near, far):
        """构造"""
        raise NotImplementedError

    def project(self, point):
        """投影"""
        raise NotImplementedError

    def cull(self, points):
        """剔除"""
        raise NotImplementedError
