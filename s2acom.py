import os

lobsid = [
	'20230103_194208_3610108077',
	'20230404_030531_3400109477',
	'20230401_161451_3400609477',
	'20140815_223609_3880012196'
]

for obs in lobsid:
	os.system('python3 s2aunit.py -o {}'.format(obs))