import os
import pickle
import numpy as np
import cv2

from common.dataset import Dataset
from common.trajectory import Trajectory


# Количество ключевых точек на кадр - несколько сотен как рекомендовано в задании
MAX_FEATURES  = 500
# Порог для ratio test Lowe - стандартное значение 0.75
RATIO_THRESH  = 0.75
# Порог RANSAC при поиске фундаментальной матрицы в пикселях
FUND_THRESH   = 3.0
# Максимально допустимая ошибка репроекции для 3D точек (10px как в задании)
REPROJ_THRESH = 10.0
# Порог RANSAC для PnP
PNP_THRESH    = 8.0
# Окно для матчинга соседних опорных кадров между собой
MATCH_WINDOW  = 8
# Минимальное количество инлаеров чтобы считать пару кадров хорошей
MIN_INLIERS   = 8


# Строю матрицу камеры K из параметров интринсики
def _build_K(intr):
    return np.array([[intr.fx, 0.,      intr.cx],
                     [0.,      intr.fy, intr.cy],
                     [0.,      0.,      1.     ]], dtype=np.float64)


def _pose_to_Rt(mat4):
    # Поза хранится как world-from-camera (4x4)
    # OpenCV ожидает обратное преобразование: camera-from-world
    # R_cam = R_world.T,  t_cam = -R_cam @ t_world
    Rw = mat4[:3, :3]
    tw = mat4[:3, 3:4]
    Rc = Rw.T
    tc = -Rc @ tw
    return Rc, tc


def _Rt_to_pose(Rc, tc):
    # Обратное преобразование: из системы камеры обратно в world-from-camera
    # нужно чтобы сохранить позу в правильном формате
    Rw = Rc.T
    tw = (-Rw @ tc).ravel()
    mat = np.eye(4)
    mat[:3, :3] = Rw
    mat[:3, 3]  = tw
    return mat


def _proj_matrix(Rc, tc, K):
    # Матрица проекции P = K * [R | t], размер 3x4
    return K @ np.hstack([Rc, tc])


# Функции для сохранения keypoints в pickle
# cv2.KeyPoint нельзя сериализовать напрямую, поэтому храним как список кортежей
def _kps_to_list(kps):
    return [(k.pt, k.size, k.angle, k.response, k.octave, k.class_id)
            for k in kps]


def _list_to_kps(lst):
    return [cv2.KeyPoint(x=p[0], y=p[1], size=s, angle=a,
                         response=r, octave=o, class_id=c)
            for p, s, a, r, o, c in lst]


def _load_features(cache_path, img_path, orb):
    # Загружаю из кэша если уже считал - не хочу каждый раз пересчитывать ORB
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            kps_data, descs = pickle.load(f)
        return _list_to_kps(kps_data), descs

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [], None

    kps, descs = orb.detectAndCompute(img, None)
    if not kps:
        return [], None

    # Оставляю только самые сильные точки по response
    if len(kps) > MAX_FEATURES:
        order = np.argsort([-k.response for k in kps])[:MAX_FEATURES]
        kps   = [kps[i] for i in order]
        descs = descs[order]

    # Сохраняю в кэш
    with open(cache_path, 'wb') as f:
        pickle.dump((_kps_to_list(kps), descs), f)
    return kps, descs


def _match_ratio(d1, d2):
    # Ratio test по Lowe: берём только те совпадения где лучший матч
    # значительно лучше второго - отсеивает неоднозначные совпадения
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return []
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(d1, d2, k=2)
    return [m for pair in raw if len(pair) == 2
            for m, n in [pair] if m.distance < RATIO_THRESH * n.distance]


def _filter_fund(kps1, kps2, matches):
    # Фильтрую матчи через фундаментальную матрицу + RANSAC
    # Это убирает геометрически неверные совпадения
    if len(matches) < MIN_INLIERS:
        return []
    p1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    p2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    _, mask = cv2.findFundamentalMat(
        p1, p2, cv2.FM_RANSAC,
        ransacReprojThreshold=FUND_THRESH, confidence=0.999)
    if mask is None:
        return []
    return [m for m, f in zip(matches, mask.ravel()) if f]


# Union-Find для построения треков
# Трек = одна и та же 3D точка видимая на нескольких кадрах
class _UF:
    def __init__(self):
        self.p = {}

    def add(self, x):
        if x not in self.p:
            self.p[x] = x

    def find(self, x):
        # Path compression для ускорения
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _build_tracks(inliers_dict, known_ids):
    # Объединяю совпадения с разных пар кадров в единые треки
    # Если точка m на кадре 1 совпала с n на кадре 2, а n совпала с k на кадре 3
    # то m, n, k - это один трек (одна 3D точка)
    uf = _UF()
    for (fi, fj), pairs in inliers_dict.items():
        for ki, kj in pairs:
            na, nb = (fi, ki), (fj, kj)
            uf.add(na); uf.add(nb)
            uf.union(na, nb)

    groups = {}
    for node in uf.p:
        groups.setdefault(uf.find(node), {})[node[0]] = node[1]

    # Оставляю только треки где есть хотя бы 2 опорных кадра - иначе нельзя триангулировать
    return [t for t in groups.values()
            if sum(1 for fid in t if fid in known_ids) >= 2]


def _triangulate_track(track, known_ids, kps_dict, proj_dict, Rt_dict, K):
    # Триангулирую 3D точку по двум опорным кадрам из трека
    known_in = [fid for fid in track if fid in known_ids]
    if len(known_in) < 2:
        return None

    fa, fb = known_in[0], known_in[1]
    pa = np.float64(kps_dict[fa][track[fa]].pt).reshape(2, 1)
    pb = np.float64(kps_dict[fb][track[fb]].pt).reshape(2, 1)

    hom = cv2.triangulatePoints(proj_dict[fa], proj_dict[fb], pa, pb)
    w = hom[3, 0]
    if abs(w) < 1e-9:
        return None
    pt3d = (hom[:3] / w).ravel()

    # Проверка хейральности - точка должна быть перед камерой (z > 0)
    for fid in (fa, fb):
        Rc, tc = Rt_dict[fid]
        if float((Rc @ pt3d + tc.ravel())[2]) <= 0:
            return None

    # Проверяю ошибку репроекции на всех опорных кадрах трека
    # Если больше 10px - точка плохая, выбрасываю (шаг 5 из задания)
    for fid in known_in:
        Rc, tc  = Rt_dict[fid]
        rvec, _ = cv2.Rodrigues(Rc)
        proj, _ = cv2.projectPoints(pt3d.reshape(1, 3), rvec, tc, K, None)
        err = np.linalg.norm(
            np.array(kps_dict[fid][track[fid]].pt) - proj.ravel())
        if err > REPROJ_THRESH:
            return None

    return pt3d


def _solve_pnp(pts3d, pts2d, K):
    # PnP: по набору 3D точек и их 2D проекциям нахожу позу камеры
    # RANSAC + уточнение методом Левенберга-Марквардта
    pts3d = np.array(pts3d, dtype=np.float64)
    pts2d = np.array(pts2d, dtype=np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, None,
        reprojectionError=PNP_THRESH,
        iterationsCount=1000,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE)

    if not ok or inliers is None or len(inliers) < 6:
        return None, 0

    # Уточняю на инлаерах
    idx = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(
        pts3d[idx], pts2d[idx], K, None, rvec, tvec)

    Rc, _ = cv2.Rodrigues(rvec)
    return _Rt_to_pose(Rc, tvec), len(idx)


def estimate_trajectory(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Кэш дескрипторов храню в папке вывода чтобы не пересчитывать при повторных запусках
    cache_dir = os.path.join(out_dir, '_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Шаг 1: загружаю метаданные датасета
    from common.intrinsics import Intrinsics
    intr        = Intrinsics.read(Dataset.get_intrinsics_file(data_dir))
    K           = _build_K(intr)
    rgb_list    = Dataset.read_dict_of_lists(Dataset.get_rgb_list_file(data_dir))
    known_traj  = Trajectory.read(Dataset.get_known_poses_file(data_dir))
    known_ids   = set(known_traj.keys())
    all_ids     = sorted(rgb_list.keys())
    unknown_ids = [fid for fid in all_ids if fid not in known_ids]

    # Считаю матрицы проекции для всех опорных кадров заранее
    proj_dict = {}
    Rt_dict   = {}
    for fid, pose in known_traj.items():
        Rc, tc         = _pose_to_Rt(pose)
        proj_dict[fid] = _proj_matrix(Rc, tc, K)
        Rt_dict[fid]   = (Rc, tc)

    # Шаг 1: ORB дескрипторы для всех кадров (опорных и неизвестных)
    orb = cv2.ORB_create(
        nfeatures=MAX_FEATURES, scaleFactor=1.2, nlevels=8,
        edgeThreshold=31, WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31)

    kps_dict   = {}
    descs_dict = {}
    for fid in all_ids:
        cache    = os.path.join(cache_dir, f'{fid}.pkl')
        img_path = os.path.join(data_dir, rgb_list[fid])
        kps_dict[fid], descs_dict[fid] = _load_features(cache, img_path, orb)

    # Шаг 2: матчинг между опорными кадрами
    # Использую скользящее окно - каждый кадр матчу только с ближайшими MATCH_WINDOW кадрами
    known_list   = sorted(known_ids)
    inliers_dict = {}
    for i, fi in enumerate(known_list):
        for fj in known_list[i + 1: i + 1 + MATCH_WINDOW]:
            matches = _match_ratio(descs_dict[fi], descs_dict[fj])
            inliers = _filter_fund(kps_dict[fi], kps_dict[fj], matches)
            if len(inliers) >= MIN_INLIERS:
                inliers_dict[(fi, fj)] = [
                    (m.queryIdx, m.trainIdx) for m in inliers]

    # Шаг 3: строю треки через Union-Find
    tracks = _build_tracks(inliers_dict, known_ids)

    # Шаги 4-5: триангулирую 3D точки и фильтрую по ошибке репроекции
    # frame_to_3d[fid][kp_idx] = pt3d - быстрый поиск 3D точки по кадру и индексу keypoint
    frame_to_3d = {}
    for track in tracks:
        pt3d = _triangulate_track(
            track, known_ids, kps_dict, proj_dict, Rt_dict, K)
        if pt3d is None:
            continue
        for fid, kp_idx in track.items():
            frame_to_3d.setdefault(fid, {})[kp_idx] = pt3d

    # Шаг 6: для каждого неизвестного кадра нахожу позу через PnP
    trajectory = dict(known_traj)

    for fid in unknown_ids:
        if descs_dict[fid] is None or len(descs_dict[fid]) == 0:
            continue

        pts3d = []
        pts2d = []
        seen  = set()  # чтобы не добавлять одну и ту же 3D точку дважды

        # Матчу неизвестный кадр с каждым опорным у которого есть 3D точки
        for ref in known_list:
            if ref not in frame_to_3d:
                continue
            matches = _match_ratio(descs_dict[ref], descs_dict[fid])
            inliers = _filter_fund(kps_dict[ref], kps_dict[fid], matches)
            ref_map = frame_to_3d[ref]
            for m in inliers:
                if m.queryIdx not in ref_map:
                    continue
                pt3d = ref_map[m.queryIdx]
                key  = id(pt3d)
                if key in seen:
                    continue
                seen.add(key)
                pts3d.append(pt3d)
                pts2d.append(kps_dict[fid][m.trainIdx].pt)

        if len(pts3d) < 6:
            continue

        pose, _ = _solve_pnp(pts3d, pts2d, K)
        if pose is not None:
            trajectory[fid] = pose

    # Шаг 7: сохраняю все позы в all_poses.txt
    Trajectory.write(Dataset.get_result_poses_file(out_dir), trajectory)
