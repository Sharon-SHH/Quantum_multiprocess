import strawberryfields as sf
from strawberryfields.utils import random_interferometer
from strawberryfields import ops
from strawberryfields import RemoteEngine
import numpy as np
from numpy import linalg

class SamplingSetting(object):
    def __init__(self, device='simulon_gaussian'):
        self.device = device

    def mean_photo_number(self, B):
        A = np.block([[0 * B, B.T], [B, 0 * B]])
        w, v = linalg.eig(B)
        w[np.abs(w) < 1e-6] = 0
        num_sv = len(np.nonzero(w)[0])
        m0 = num_sv * np.sinh(1.0) * np.sinh(1.0) / 4
        parameters = sf.decompositions.bipartite_graph_embed(A, m0)
        sqz_param = np.array(parameters[0])
        mpn = np.sum(np.sinh(sqz_param) * np.sinh(sqz_param)) / 8
        return mpn

    def run_device(self, B):
        '''B: a 4*4 matrix of correlation'''
        #eng = sf.RemoteEngine(self.device)
        eng = sf.Engine(self.device)
        prog = sf.Program(8)
        mean_photon_n = self.mean_photo_number(B)
        # print(B)
        with prog.context as q:
            ops.BipartiteGraphEmbed(B, mean_photon_per_mode=mean_photon_n, edges=True) | q
            ops.MeasureFock() | q

        res = eng.run(prog)
        return res