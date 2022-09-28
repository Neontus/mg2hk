# running unit tests

import os
import numpy as np

to_test = [
    '20220626_040436_3620108077',
    '20220721_053919_3620108077',
    '20220722_222826_3620108076',
    '20220724_012227_3620110077',
    '20220730_125009_3620112077'
]

blur_to_test = np.arange(20, 40, 5)
n_to_test = np.arange(0.1, 0.25, 0.05)

# os.system("python coordalignunittest.py -o {} -b {} -n{}".format(to_test[0], blur_to_test[0], n_to_test[0]))

for n in n_to_test:
    for b in blur_to_test:
        os.system("python coordalignunittest.py -o {} -b {} -n{}".format(to_test[0], b, n))