import os

lobsid = [
	'20230103_194208_3610108077',
	'20230404_030531_3400109477',
	'20230401_161451_3400609477',
	'20140815_223609_3880012196'
]

for obs in lobsid:
	os.system('python3 s2aunit.py -o {}'.format(obs))

# new stuff
import re
with open('/Users/jkim/Downloads/query.txt', 'r') as f:
   n = f.readlines()
   f.close()
obsids = [re.findall(r'\/(\d*_\d*_\d*)\/', o) for o in n][0]

import os
for a in np.unique(obsids):
	try:
		os.system('python3 s2adatacubeunit.py -o {}'.format(a))
	except:
		print("error: ", a)