import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
import scipy.sparse as sp
from sklearn.ensemble import RandomTreesEmbedding


def info_gain(x, y, k=None):
    num_d = len(y)
    num_ck = {}
    num_fi_ck = {}
    num_nfi_ck = {}
    for xi, yi in zip(x.data, y):
        num_ck[yi] = num_ck.get(yi, 0) + 1
        for index, xii in enumerate(xi.data):
            if index not in num_fi_ck:
                num_fi_ck[index] = {}
                num_nfi_ck[index] = {}
            if yi not in num_fi_ck[index]:
                num_fi_ck[index][yi] = 0
                num_nfi_ck[index][yi] = 0
            if not xii == 0:
                num_fi_ck[index][yi] = num_fi_ck[index].get(yi) + 1
            else:
                num_nfi_ck[index][yi] = num_nfi_ck[index].get(yi) + 1
    num_fi = {}
    for fi, dic in num_fi_ck.items():
        num_fi[fi] = sum(dic.values())
    num_nfi = dict([(fi, num_d - num) for fi, num in num_fi.items()])
    HD = 0
    for ck, num in num_ck.items():
        p = float(num) / num_d
        HD = HD - p * np.math.log(p, 2)
    IG = {}
    for fi in num_fi_ck.keys():
        POS = 0
        for yi, num in num_fi_ck[fi].items():
            p = (float(num) + 0.0001) / (num_fi[fi] + 0.0001 * len(dic))
            POS = POS - p * np.math.log(p, 2)

        NEG = 0
        for yi, num in num_nfi_ck[fi].items():
            p = (float(num) + 0.0001) / (num_nfi[fi] + 0.0001 * len(dic))
            NEG = NEG - p * np.math.log(p, 2)
        p = float(num_fi[fi]) / num_d
        IG[fi] = round(HD - p * POS - (1 - p) * NEG, 4)
    IG = sorted(IG.items(), key=lambda d: d[1], reverse=True)
    if k == None:
        return IG
    else:
        return IG[0:k]


class InformationGain(TransformerMixin):
    def __init__(self, normalize=True, idf=False, norm="l2", binary=False, dtype=np.float32):
        pass

    def fit(self, X, y=None, verbose=False):
        def _calIg():
            entropy_x_set = 0
            entropy_x_not_set = 0
            for c in classCnt:
                probs = classCnt[c] / float(featureTot)
                entropy_x_set = entropy_x_set - probs * np.log(probs)
                probs = (classTotCnt[c] - classCnt[c]) / float(tot - featureTot)
                entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
            for c in classTotCnt:
                if c not in classCnt:
                    probs = classTotCnt[c] / float(tot - featureTot)
                    entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
            return entropy_before - ((featureTot / float(tot)) * entropy_x_set
                                     + ((tot - featureTot) / float(tot)) * entropy_x_not_set)

        tot = X.shape[0]
        classTotCnt = {}
        entropy_before = 0
        for i in y:
            if i not in classTotCnt:
                classTotCnt[i] = 1
            else:
                classTotCnt[i] = classTotCnt[i] + 1
        for c in classTotCnt:
            probs = classTotCnt[c] / float(tot)
            entropy_before = entropy_before - probs * np.log(probs)

        nz = X.T.nonzero()
        pre = 0
        classCnt = {}
        featureTot = 0
        information_gain = []
        for i in range(0, len(nz[0])):
            if (i != 0 and nz[0][i] != pre):
                for notappear in range(pre + 1, nz[0][i]):
                    information_gain.append(0)
                ig = _calIg()
                information_gain.append(ig)
                pre = nz[0][i]
                classCnt = {}
                featureTot = 0
            featureTot = featureTot + 1
            yclass = y[nz[1][i]]
            if yclass not in classCnt:
                classCnt[yclass] = 1
            else:
                classCnt[yclass] = classCnt[yclass] + 1
        ig = _calIg()
        information_gain.append(ig)

        return np.asarray(information_gain)

    def transform(self, X, y):
        def _entropy(values):
            counts = np.bincount(values)
            probs = counts[np.nonzero(counts)] / float(len(values))
            return - np.sum(probs * np.log(probs))

        def _information_gain(feature, y):
            feature_set_indices = np.nonzero(feature)[0]
            feature_not_set_indices = [i for i in feature_range if i not in feature_set_indices]
            entropy_x_set = _entropy(y[feature_set_indices])
            entropy_x_not_set = _entropy(y[feature_not_set_indices])

            return entropy_before - (((len(feature_set_indices) / float(feature_size)) * entropy_x_set)
                                     + ((len(feature_not_set_indices) / float(feature_size)) * entropy_x_not_set))

        feature_size = X.shape[0]
        feature_range = range(0, feature_size)
        entropy_before = _entropy(y)
        information_gain_scores = []

        for feature in X.T:
            information_gain_scores.append(_information_gain(feature, y))
        return information_gain_scores


class PredefinedFeatureSelection(TransformerMixin):
    def __init__(self):
        self.features = [(4098, 5), (4099, 5), (10247, 5), (4107, 5), (4109, 5), (11608, 5), (4114, 5), (4115, 5),
                         (5465, 5), (4121, 5), (4125, 5), (10273, 5), (4132, 5), (11458, 5), (4139, 5), (4143, 5),
                         (10291, 5), (5470, 5), (4150, 5), (4152, 5), (10298, 5), (9226, 5), (3424, 5), (11617, 5),
                         (1036, 5), (4173, 5), (6224, 5), (9209, 5), (4192, 5), (4193, 5), (11622, 5), (4198, 5),
                         (4211, 5), (4213, 5), (10359, 5), (4216, 5), (10370, 5), (4227, 5), (4238, 5), (7535, 5),
                         (11632, 5), (1054, 5), (8379, 5), (8390, 5), (8394, 5), (8404, 5), (8417, 5), (9937, 5),
                         (8425, 5), (8426, 5), (5501, 5), (8432, 5), (2305, 5), (8450, 5), (10499, 5), (8453, 5),
                         (10503, 5), (8476, 5), (2341, 5), (11655, 5), (10543, 5), (10552, 5), (10553, 5), (10571, 5),
                         (334, 5), (3469, 5), (336, 5), (6200, 5), (4436, 5), (2390, 5), (2391, 5), (2396, 5),
                         (11665, 5), (2409, 5), (2411, 5), (3474, 5), (376, 5), (11670, 5), (8590, 5), (411, 5),
                         (9220, 5), (10243, 5), (429, 5), (433, 5), (440, 5), (442, 5), (446, 5), (3489, 5), (458, 5),
                         (463, 5), (8657, 5), (4567, 5), (473, 5), (474, 5), (475, 5), (4574, 5), (482, 5), (483, 5),
                         (488, 5), (8682, 5), (8683, 5), (5977, 5), (3426, 5), (503, 5), (8697, 5), (506, 5), (8701, 5),
                         (8708, 5), (517, 5), (4616, 5), (4622, 5), (11693, 5), (528, 5), (3502, 5), (535, 5),
                         (8731, 5), (4637, 5), (4640, 5), (546, 5), (548, 5), (8742, 5), (8743, 5), (8744, 5),
                         (8750, 5), (8751, 5), (7603, 5), (8756, 5), (4662, 5), (8768, 5), (8769, 5), (8771, 5),
                         (10821, 5), (8774, 5), (10828, 5), (11704, 5), (8796, 5), (606, 5), (607, 5), (609, 5),
                         (11341, 5), (10870, 5), (10182, 5), (10876, 5), (637, 5), (641, 5), (645, 5), (646, 5),
                         (647, 5), (652, 5), (654, 5), (660, 5), (675, 5), (10916, 5), (686, 5), (693, 5), (694, 5),
                         (700, 5), (713, 5), (714, 5), (715, 5), (11029, 5), (11033, 5), (11040, 5), (11042, 5),
                         (11082, 5), (11085, 5), (11088, 5), (3556, 5), (11103, 5), (11108, 5), (11117, 5), (11146, 5),
                         (3565, 5), (9120, 5), (9122, 5), (9132, 5), (9134, 5), (11188, 5), (955, 5), (9148, 5),
                         (501, 5), (9154, 5), (963, 5), (964, 5), (970, 5), (972, 5), (976, 5), (978, 5), (9174, 5),
                         (9175, 5), (985, 5), (9180, 5), (9183, 5), (993, 5), (11239, 5), (11240, 5), (11241, 5),
                         (1004, 5), (11245, 5), (9202, 5), (1011, 5), (11252, 5), (11253, 5), (11254, 5), (4330, 5),
                         (11258, 5), (9386, 5), (1022, 5), (11266, 5), (11268, 5), (11269, 5), (11270, 5), (1031, 5),
                         (1032, 5), (1033, 5), (11274, 5), (11275, 5), (11276, 5), (11277, 5), (11279, 5), (11285, 5),
                         (11286, 5), (11287, 5), (11292, 5), (11294, 5), (1058, 5), (9254, 5), (11306, 5), (11307, 5),
                         (11315, 5), (11317, 5), (11318, 5), (11322, 5), (11323, 5), (11324, 5), (2012, 5), (11328, 5),
                         (11329, 5), (11331, 5), (11332, 5), (11333, 5), (11335, 5), (11338, 5), (11340, 5), (11343, 5),
                         (11344, 5), (5203, 5), (11352, 5), (11353, 5), (5210, 5), (11356, 5), (9309, 5), (11361, 5),
                         (11362, 5), (11364, 5), (11366, 5), (9319, 5), (11368, 5), (11373, 5), (9327, 5), (11377, 5),
                         (11378, 5), (11379, 5), (11384, 5), (9337, 5), (11455, 5), (11388, 5), (11389, 5), (9344, 5),
                         (11394, 5), (11395, 5), (11396, 5), (1917, 5), (11399, 5), (11401, 5), (11403, 5), (11404, 5),
                         (11405, 5), (11406, 5), (11407, 5), (11409, 5), (11410, 5), (11413, 5), (3223, 5), (5273, 5),
                         (11418, 5), (11419, 5), (11420, 5), (11426, 5), (5286, 5), (11432, 5), (11433, 5), (11434, 5),
                         (9757, 5), (5298, 5), (11444, 5), (11452, 5), (5309, 5), (11454, 5), (9410, 5), (5316, 5),
                         (11809, 5), (11465, 5), (11466, 5), (11467, 5), (11470, 5), (11471, 5), (3282, 5), (11479, 5),
                         (11481, 5), (11482, 5), (11487, 5), (11488, 5), (11489, 5), (11490, 5), (11492, 5), (11814, 5),
                         (5350, 5), (5351, 5), (5352, 5), (11498, 5), (11499, 5), (11500, 5), (5358, 5), (5359, 5),
                         (11507, 5), (5364, 5), (11519, 5), (5376, 5), (5377, 5), (5379, 5), (5382, 5), (11528, 5),
                         (11530, 5), (11531, 5), (7436, 5), (11535, 5), (11536, 5), (11537, 5), (11541, 5), (11542, 5),
                         (11544, 5), (11545, 5), (11546, 5), (11551, 5), (11553, 5), (11554, 5), (11483, 5), (11556, 5),
                         (11557, 5), (5415, 5), (11565, 5), (11566, 5), (5424, 5), (11569, 5), (5428, 5), (7478, 5),
                         (11575, 5), (11577, 5), (11579, 5), (11581, 5), (11586, 5), (3395, 5), (3396, 5), (5445, 5),
                         (3398, 5), (11591, 5), (11592, 5), (3401, 5), (11597, 5), (3406, 5), (5456, 5), (11601, 5),
                         (11602, 5), (3413, 5), (11607, 5), (5464, 5), (11609, 5), (11612, 5), (11614, 5), (11616, 5),
                         (11618, 5), (11619, 5), (11620, 5), (3430, 5), (11623, 5), (11626, 5), (7531, 5), (5485, 5),
                         (11631, 5), (5488, 5), (11633, 5), (3442, 5), (3443, 5), (7542, 5), (11639, 5), (7544, 5),
                         (3450, 5), (7548, 5), (11645, 5), (11646, 5), (11647, 5), (3463, 5), (11657, 5), (7565, 5),
                         (3471, 5), (11664, 5), (3473, 5), (11666, 5), (11667, 5), (11777, 5), (3478, 5), (7576, 5),
                         (7577, 5), (11677, 5), (11678, 5), (11679, 5), (7584, 5), (7585, 5), (11682, 5), (11683, 5),
                         (582, 5), (7590, 5), (11691, 5), (3501, 5), (11694, 5), (11695, 5), (3504, 5), (11699, 5),
                         (11700, 5), (7605, 5), (11702, 5), (7608, 5), (11710, 5), (11711, 5), (11712, 5), (7619, 5),
                         (11717, 5), (11719, 5), (588, 5), (3530, 5), (3531, 5), (3533, 5), (7630, 5), (3536, 5),
                         (3538, 5), (11731, 5), (11732, 5), (11740, 5), (11742, 5), (11743, 5), (11453, 5), (11747, 5),
                         (11748, 5), (11749, 5), (11753, 5), (11754, 5), (11756, 5), (11757, 5), (11758, 5), (11760, 5),
                         (3569, 5), (11763, 5), (11765, 5), (11766, 5), (11768, 5), (11769, 5), (3579, 5), (3584, 5),
                         (3585, 5), (11778, 5), (11781, 5), (3590, 5), (11786, 5), (3595, 5), (11789, 5), (11791, 5),
                         (11793, 5), (11798, 5), (11802, 5), (11803, 5), (11804, 5), (3620, 5), (11813, 5), (3622, 5),
                         (11819, 5), (11820, 5), (11821, 5), (11824, 5), (11826, 5), (10163, 5), (11828, 5), (11829, 5),
                         (11830, 5), (11831, 5), (11832, 5), (11834, 5), (11838, 5), (11843, 5), (11844, 5), (11845, 5),
                         (11442, 5), (11851, 5), (3662, 5), (11857, 5), (9810, 5), (9811, 5), (9816, 5), (11867, 5),
                         (11869, 5), (3683, 5), (9829, 5), (3686, 5), (3699, 5), (959, 5), (4032, 5), (9860, 5),
                         (1985, 5), (9874, 5), (9876, 5), (1701, 5), (11548, 5), (1712, 5), (9843, 5), (1718, 5),
                         (9916, 5), (11787, 5), (1730, 5), (9923, 5), (1735, 5), (11835, 5), (9935, 5), (1745, 5),
                         (1746, 5), (1747, 5), (9940, 5), (1754, 5), (1755, 5), (9951, 5), (1760, 5), (9962, 5),
                         (1773, 5), (1775, 5), (1778, 5), (1795, 5), (1800, 5), (1806, 5), (1807, 5), (11858, 5),
                         (5422, 5), (1819, 5), (4060, 5), (1835, 5), (4061, 5), (11859, 5), (5939, 5), (5940, 5),
                         (5953, 5), (1860, 5), (11574, 5), (5968, 5), (5969, 5), (5974, 5), (5976, 5), (11701, 5),
                         (10086, 5), (10088, 5), (1899, 5), (11520, 5), (1913, 5), (11242, 5), (1924, 5), (6022, 5),
                         (6023, 5), (1934, 5), (10128, 5), (6033, 5), (1944, 5), (1945, 5), (1950, 5), (3397, 5),
                         (1954, 5), (1955, 5), (11590, 5), (10149, 5), (1958, 5), (10152, 5), (1961, 5), (1962, 5),
                         (1966, 5), (1967, 5), (1970, 5), (1971, 5), (1972, 5), (1012, 5), (4031, 5), (4635, 5),
                         (4033, 5), (4034, 5), (1987, 5), (10180, 5), (4037, 5), (1990, 5), (4042, 5), (4044, 5),
                         (4046, 5), (4047, 5), (4049, 5), (11257, 5), (1018, 5), (3408, 5), (4066, 5), (4068, 5),
                         (4069, 5), (5457, 5), (9213, 5), (3411, 5), (4086, 5), (1705, 5), (10234, 5), (4092, 5),
                         (4095, 5), (4184, 4), (3432, 4), (10361, 4), (8383, 4), (10510, 4), (2348, 4), (10558, 4),
                         (2414, 4), (377, 4), (2436, 4), (8592, 4), (422, 4), (8618, 4), (4576, 4), (4597, 4),
                         (8710, 4), (4617, 4), (8714, 4), (8726, 4), (8764, 4), (8767, 4), (3519, 4), (667, 4),
                         (687, 4), (736, 4), (5483, 4), (8672, 4), (9119, 4), (3568, 4), (965, 4), (9187, 4), (1023, 4),
                         (11308, 4), (11311, 4), (7690, 4), (11411, 4), (9379, 4), (9392, 4), (5339, 4), (11522, 4),
                         (11524, 4), (5404, 4), (11603, 4), (3425, 4), (11624, 4), (11733, 4), (8785, 4), (3638, 4),
                         (11521, 4), (7709, 4), (9761, 4), (9774, 4), (3643, 4), (11836, 4), (3668, 4), (9867, 4),
                         (7833, 4), (1714, 4), (9909, 4), (9917, 4), (1768, 4), (1769, 4), (9197, 4), (1869, 4),
                         (10079, 4), (10121, 4), (1005, 4), (2382, 4), (10208, 4), (10214, 4), (10246, 3), (327, 3),
                         (4563, 3), (512, 3), (5212, 3), (1002, 3), (11309, 3), (9293, 3), (11390, 3), (11736, 3),
                         (11815, 3), (3697, 3), (7817, 3), (1789, 3), (8008, 3), (10075, 3), (8589, 2), (8600, 2),
                         (9407, 2), (3464, 2), (3479, 2), (11734, 2), (8026, 2), (7497, 2), (3613, 2)]
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        indics = np.array([idx for idx, freq in self.features if freq >= 5], dtype=np.int)
        indics.sort()
        # indics = indics[:200]
        print("selecting %d features " % len(indics))
        return X[:, indics].toarray()


class TopRandomTreesEmbedding(BaseEstimator,TransformerMixin):
    def __init__(self, k=100,n_estimators=20, max_depth=10):
        self.k = k
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, X, y):
        self._rtree = RandomTreesEmbedding(n_estimators=self.n_estimators, max_depth=self.max_depth,sparse_output=False) #sparse_output=False,,sparse_output=False
        self._rtree.fit(X, y)
        non_zero_indics = np.nonzero(self._rtree.feature_importances_)[0]
        important_indics = self._rtree.feature_importances_.argsort()[::-1][:self.k]
        self.important_indices = np.intersect1d(important_indics,non_zero_indics)
        return self

    def transform(self, X):
        return X[:,self.important_indices].toarray()

        # indics = np.nonzero(self._rtree.feature_importances_)[0]
        # return X[:, indics].toarray()

        # important_indics = self._rtree.feature_importances_.argsort()[::-1][:self.k]
        # return X[:,important_indics].toarray()

import os

if __name__ == "__main__":
    X = np.random.randint(10, 1000, (120, 2000))
    y = np.concatenate((np.ones(80, dtype=np.int), 2 * np.ones(40, dtype=np.int)))
    info_gain = InformationGain()
    X_transform = sp.csr_matrix(info_gain.transform(X, y)[0])
    # print(X_transform.shape)
    print(X_transform)
    # x = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]]
    # y = [1, 2, 3, 2, 2]
    # IG = info_gain(X, y, 3)
    # for k, v in IG:
    #     print(k, v)
