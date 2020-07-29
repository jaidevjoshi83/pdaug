

file_name = 'all_against_all_Neg.tsv'
Sep = ','


file  = open('New_all_Neg.csv')
lines = file.readlines()[1:]

index = []

for line in lines:
	index.append(line.split(Sep)[0])

abs = list(set(index))

outfile = open('outfile.csv', 'w')
outfile.write("index"+Sep.join(abs)+'\n')

for i  in abs:
	outfile.write(i),
	for j in abs:
		for line in lines:
			if i+Sep+j in line:
				outfile.write(Sep+line.split(Sep)[2].strip('\n'))
			else:
				pass

	outfile.write("\n")

outfile.close()





