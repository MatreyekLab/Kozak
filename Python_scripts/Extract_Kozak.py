## Usage note: Pass the fast filename (with .fastq attached) as the first argument
## eg. "python3 Extract_Kozak.py IDT_STD.assembled.fastq"

import os
import sys
import csv
import codecs

infile_name = sys.argv[1]  ## File name should be the first argument

header_list = []
sequence_list = []
qscore_list = []


index = 1

## Take all of the lines of the fastq and separate them out by category

with open(infile_name,'rt') as csvfile:
	csvfile = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in csvfile:
		if index == 1:
			header_list.append(row)
			index += 1
			continue
		if index == 2:
			sequence_list.append(row)
			index += 1
			continue	
		if index == 3:
			index += 1
			continue
		if index == 4:
			qscore_list.append(row)
			index = 1
			continue


## Print out lists to make sure everything looks right

#print(len(header_list))
#print(len(sequence_list))
#print(len(qscore_list))

## Now go through the lists and extract the info one wants

read_class = []
kozak_sequence = []

for x in range(0,len(sequence_list)):
	temp_sequence = str(sequence_list[x])
	temp_index = temp_sequence.find("CGCAACTACAC")
	potential_kozak = temp_sequence[(temp_index+11):(temp_index+20)]
	if potential_kozak == "TCCAGCTCC":
		read_class.append("Template")
		kozak_sequence.append("NA")
	elif potential_kozak[6:9] == "ATG":
		read_class.append("Kozak")
		kozak_sequence.append(potential_kozak[0:6])
	else:
		read_class.append("Other")
		kozak_sequence.append("NA")

#print(read_class)
#print(kozak_sequence)

## Now write everything out in a new csv file

outfile_name = str(infile_name[:len(infile_name)-6])+".tsv"
outfile = codecs.open(outfile_name, "w", "utf-8", "replace")

# Write the header
#outfile.write("{}\t{}\t{}\t{}\t{}\n".format("header","sequence","qscore","read_class","kozak_sequence"))
outfile.write("{}\t{}\n".format("read_class","kozak_sequence"))

for i in range(0,len(header_list)):
	#outfile.write("{}\t{}\t{}\t{}\t{}\n".format(header_list[i], sequence_list[i], qscore_list[i], read_class[i], kozak_sequence[i]))
	outfile.write("{}\t{}\n".format(read_class[i], kozak_sequence[i]))