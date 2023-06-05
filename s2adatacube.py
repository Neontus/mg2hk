# new stuff
import re
with open('/Users/jkim/Downloads/query.txt', 'r') as f:
   n = f.readlines()
   f.close()
obsids = [re.findall(r'\/(\d*_\d*_\d*)\/', o) for o in n][0]

import os
for obs in np.unique(obsids):
   try:
      os.system('python3 s2aunit.py -o {}'.format(obs))
   except:
      print("error: ", obs)