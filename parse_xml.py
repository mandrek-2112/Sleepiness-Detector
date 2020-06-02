# import necessary packages
import argparse
import re

# parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to iBug 300-W data split XML file")
ap.add_argument("-t", "--output", required=True,
	help="path output data split XML file")
args = vars(ap.parse_args())

# set integer indices for eyes according to iBUG 300-W annotations
# keeping in mind that indices start from 0
LANDMARKS = set(list(range(36, 48)))

# to get lines starting with 'part name' and followed by any number
PART = re.compile("part name='[0-9]+'")

# load the contents of the original XML file and open the output file
# for writing
print("[RESULT] Opening input XML file.")
rows = open(args["input"]).read().strip().split("\n")
print("[RESULT] Opening output XML file.")
output = open(args["output"], "w")

# begin parsing
print("[RESULT] Parsing input XML file.")

# loop over the rows of the input cml file
for row in rows:
	# check to see if the current line satisfies regex
	parts = re.findall(PART, row)

	# if no lines found
	if len(parts) == 0:
		output.write("{}\n".format(row))

	# if lines found
	else:
		# parse out the name of the attribute from the row
		attr = "name='"
		i = row.find(attr)
		j = row.find("'", i + len(attr) + 1)
		name = int(row[i + len(attr):j])

		# if within eye range, write to file
		if name in LANDMARKS:
			output.write("{}\n".format(row))

# close the output file
output.close()
print("[RESULT] New XML file generated.")