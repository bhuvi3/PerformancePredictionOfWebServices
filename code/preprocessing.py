#"""
data = open('../data/qws_preprocessed.csv')
first_line = data.readline()
country_dict = {}
ip_dict = {}
wsdl_dict = {}
provider_dict = {}
c = 0
i = 0
w = 0
p = 0
lines = data.readlines()
for line in lines:
	line = line[:-1]
	toks = line.split(',')
	if country_dict.has_key(toks[8]) == False:
		c += 1
		country_dict[toks[8]] = c
	#
	if country_dict.has_key(toks[3]) == False:
		c += 1
		country_dict[toks[3]] = c
	#
	if ip_dict.has_key(toks[2]) == False:
		i += 1
		ip_dict[toks[2]] = i
	#
	if wsdl_dict.has_key(toks[6]) == False:
		w += 1
		wsdl_dict[toks[6]] = w
	#
	if provider_dict.has_key(toks[7]) == False:
		p += 1
		provider_dict[toks[7]] = p
	#
#
data.close()

new_data = open('../data/qws_preprocessed_complete.csv','w')
new_data.write(first_line)
new_lines = []
for line in lines:
	line = line[:-1]
	toks = line.split(',')
	toks[0] = 'U' + str(toks[0])
	toks[1] = 'WS' + str(toks[1])
	toks[3] = 'C' + str(country_dict[toks[3]])
	toks[8] = 'C' + str(country_dict[toks[8]])
	toks[2] = 'IP' + str(ip_dict[toks[2]])
	toks[6] = 'WSDL' + str(wsdl_dict[toks[6]])
	toks[7] = 'WSP' + str(provider_dict[toks[7]])
	new_line = ','.join([str(x) for x in toks])
	new_data.write(new_line + '\n')
#
new_data.close()
"""
#
data = open('../data/qws_preprocessed_complete.csv')
new_data = open('../data/qws_preprocessed_complete_normalized.csv','w')
first_line = data.readline()
new_data.write(first_line)
lines = data.readlines()
new_lines = []
for line in lines:
	line = line[:-1]
	toks = line.split(',')
	toks[0] = 'A' + str(toks[0])
	toks[1] = 'A' + str(toks[1])
	toks[3] = 'A' + str(toks[3])
	toks[8] = 'A' + str(toks[8])
	new_line = ','.join([str(x) for x in toks])
	new_data.write(new_line + '\n')
#
new_data.close()
data.close()
#
#"""
