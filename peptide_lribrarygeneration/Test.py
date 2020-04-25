f = open('jai_1.fasta')

lines = f.readlines()

flines = []

for line in lines:
    if '>' in line:
        pass
    else:
        flines.append(line.strip('\n'))

flines = "".join(flines)


print flines
