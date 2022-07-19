from scipy.ndimage import median_filter as blur
import numpy as np

def cond(data, threshold, blur_size=5, value=1, cond='>'):

    if blur_size > 0:
        bdata = blur(data, size=blur_size)
    else:
        bdata=data
    if cond == '>': w = np.where(bdata > threshold)
    if cond == '>=': w = np.where(bdata >= threshold)
    if cond == '<': w = np.where(bdata < threshold)
    if cond == '<=': w = np.where(bdata <= threshold)
    if cond == '==': w = np.where(bdata == threshold)

    return make_mask(bdata, w, value=value)


def make_mask(data, w_cond, value=1):

    aux = data*0
    aux[w_cond] = value

    return aux #, aux.astype(boolean)


