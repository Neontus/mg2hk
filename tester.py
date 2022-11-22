# running unit tests

import os
import numpy as np

to_test = [
    '20221008_030317_3620108077',
#     '20220924_021937_3620110077'
    '20220626_040436_3620108077',
    '20220721_053919_3620108077',
    '20220722_222826_3620108076',
    '20220724_012227_3620110077',
    '20220730_125009_3620112077'
]

init_values = [
    [0.21, 0.39, 20],
    [0.21, 0.12, 20],
    [0.15, 0.10, 30],
    [0.21, 0.29, 20],
    [0.21, 0.12, 20]
]

a_to_test = np.arange(0.19, 0.25, 0.02)
i_to_test = np.arange(0.10, 0.5, 0.05)
blur_to_test = np.arange(20, 30, 5)

# for i in to_test:
#     os.system("python unitalign.py -o {} -a {} -i {} -b {}".format(i, 0.21, 0.12, 25))

# for i, _id in enumerate(to_test):
#     os.system("python unitalign.py -o {} -a {} -i {} -b {}".format(_id, init_values[i][0], init_values[i][1], init_values[i][2]))

# os.system("python coordalignunittest.py -o {} -b {} -n{}".format(to_test[0], blur_to_test[0], n_to_test[0]))
for img in to_test:
    for a in a_to_test:
        for i in i_to_test:
            for b in blur_to_test:
                os.system("python unitalign.py -o {} -a {} -i{} -b{}".format(img, a, i, b))


# for test in to_test:
#     os.system("python ransacunit.py -o {} -b {} -n{}".format(test, 30, 0.10))

# os.system("python ransacunit.py -o {} -b {} -n{} -r{}".format(to_test[1], 20, 0.21, 0.35))
#
# for r in r_to_test:
#     os.system("python ransacunit.py -o {} -b {} -n{} -r{}".format(to_test[0], 20, 0.21, r))
#
# os.system("python ransacunit.py -o {} -b {} -n{} -r{}".format(to_test[0], 20, 0.21, 0.38))
# os.system("python ransacunit.py -o {} -b {} -n{} -r{}".format(to_test[1], 20, 0.21, 0.31))
# os.system("python ransacunit.py -o {} -b {} -n{} -r{}".format(to_test[4], 20, 0.21, 0.12))

