from ColorUtils import *

from DataLoader import *

sigL = [0.0283, 0.133, 4.336]
sigRG = [0.0536, 0.494]
sigBY = [0.0392, 0.386]

omegaL = [0.921, 0.105, -0.108]
omegaRG = [0.488, 0.330]
omegaBY = [0.531, 0.371]



def intermediate_kernel(l=5, sig=2.0):
    """\
    Gaussian Kernel Creator via given length and sigma
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(1) * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)


def get_spatial_kernels():
    L_filters = []
    RG_filters = []
    BY_filters = []

    L_filter = np.zeros((5,5))
    RG_filter = np.zeros((5,5))
    BY_filter = np.zeros((5,5))


    for sig in sigL:
        L_filters.append(intermediate_kernel(5, sig))

    for sig in sigRG:
        RG_filters.append(intermediate_kernel(5, sig))

    for sig in sigBY:
        BY_filters.append(intermediate_kernel(5, sig))


    for i in range(2):
        L_filter = L_filter + L_filters[i]*omegaL[i]
        RG_filter = RG_filter + RG_filters[i] * omegaRG[i]
        BY_filter = BY_filter + BY_filters[i] * omegaBY[i]

    L_filter = L_filter + L_filters[2]*omegaL[2]

    L_filter = L_filter / np.sum(L_filter)
    RG_filter = RG_filter / np.sum(RG_filter)
    BY_filter = BY_filter / np.sum(BY_filter)

    return L_filter, RG_filter, BY_filter


