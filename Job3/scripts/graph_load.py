from __future__ import print_function
from sys import argv
import re
import subprocess
import time
prog = re.compile('#define LAG_THRESH (\d+)')
jobN = re.compile('Your job (\d+)')
ctime = re.compile('Cuda time \(ms\):   (\d+).(\d+) \( + (\d+).(\d+)')
for thresh in range(2000,2001):
	header = open('./include/matRedux.h','r')
	header_data = ''
	for line in header:
		res = prog.match(line)
		if res:
			header_data = header_data + '#define LAG_THRESH '+str(thresh)+'\n'
		else:
			header_data = header_data + line
	header.close()
	header = open('./include/matRedux.h','w')
	header.write(header_data)
	header.close()
	token = jobN.match(subprocess.check_output(['qsub','./scripts/matRedux.pbs'])).group(1)
	pbs_outfile = '../../matRedux.pbs.o'+token
	print('Sleeping for five minutes...',end='')
	time.sleep(5*60)
	print('Done.')
	pbsfile = open(pbs_outfile,'r')
	for line in pbsfile:
		res = ctime.match(line)
		if res:
			crtime = res.group(1)+'.'+res.group(2)
			cmtime = res.group(3)+'.'+res.group(4)
			break
	crtime = double(crtime)
	cmtime = double(cmtime)
	print(crtime)
	print(cmtime)

	
	
