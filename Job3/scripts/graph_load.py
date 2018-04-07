from __future__ import print_function
from sys import argv
import sys
import re
import subprocess
import time
import pickle
prog = re.compile('#define LAG_THRESH (\d+)')
jobN = re.compile('Your job (\d+)')
ctime = re.compile('Cuda time [(]ms[)]:   (\d+)[.](\d+) [(] \+ (\d+).(\d+)')
redtimes = []
memtimes = []
for thresh in range(1,1024):
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
	subprocess.call(['./scripts/genMat','
	token = jobN.match(subprocess.check_output(['qsub','./scripts/matRedux.pbs'])).group(1)
	pbs_outfile = '../../matRedux.pbs.o'+token
	print('Sleeping for five minutes...',end='')
	sys.stdout.flush()
	time.sleep(5*60)
	print('Done.')
	sys.stdout.flush()
	pbsfile = open(pbs_outfile,'r')
	for line in pbsfile:
		res = ctime.match(line)
		if res:
			crtime = res.group(1)+'.'+res.group(2)
			cmtime = res.group(3)+'.'+res.group(4)
			break
	crtime = float(crtime)
	cmtime = float(cmtime)
	print(str(thresh)+':'+str(ctime)+','+str(cmtime))
	redtimes.append(crtime)
	memtimes.append(cmtime)
with open('threshold_times.pkl','w') as f:
	pickle.dump([redtimes,memtimes],f)
