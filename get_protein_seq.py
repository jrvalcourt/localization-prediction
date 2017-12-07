import sys
import os
import urllib.request
import gzip
import numpy as np
import pickle

# length of the N-terminal sequence we consider
seq_len = 100

# how many examples of the given localization do we need in order to consider it?
min_loc_count = 100

pickle_dir = 'data/'
x_data_filename = os.path.join(pickle_dir, 'X.p.gz')
y_data_filename = os.path.join(pickle_dir, 'Y.p.gz')
names_data_filename = os.path.join(pickle_dir, 'names.p.gz')
features_filename = os.path.join(pickle_dir, 'features.p.gz')

aa2int = {'A' : 0,
          'C' : 1,
          'D' : 2,
          'E' : 3,
          'F' : 4,
          'G' : 5,
          'H' : 6,
          'I' : 7,
          'K' : 8,
          'L' : 9,
          'M' : 10,
          'N' : 11,
          'P' : 12,
          'Q' : 13,
          'R' : 14,
          'S' : 15,
          'T' : 16,
          'U' : 17,
          'V' : 18,
          'W' : 19,
          'X' : 20,
          'Y' : 21,
          '-' : 22}
loc_counts = {}

num_aa = len(aa2int)

# returns the protein sequence
def get_sequence(prot_id):
    url = 'http://rest.ensembl.org/sequence/id/' + prot_id + \
          '?content-type=text/x-fasta'  # ;type=protein'
    response = urllib.request.urlopen(url)
    fasta = response.read().decode('utf-8')
    e = fasta.split()
    return ''.join(e[1:])

# generate a lookup table that maps gene -> protein.
# we consider only the first protein associated with a
# given gene.
gene2prot = {}
with open(sys.argv[2]) as fin:
    header = fin.readline()
    for line in fin:
        e = line.split()
        gene_id = e[0]
        if len(e) < 3:
            continue
        if not gene_id in gene2prot:
            gene2prot[gene_id] = e[2]

# how many records do we have?
gene_count = 0
with open(sys.argv[1]) as fin:
    header = fin.readline()
    for line in fin:
        e = line.split('\t')
        gene_id = e[0]
        if gene_id in gene2prot:
            gene_count += 1
        else:
            sys.stderr.write("Skipping {}\n".format(gene_id))
    print('\rCounted {:d} genes'.format(gene_count))

# if we've already stored the sequence data, fetch it.
# otherwise, initialize an array to hold it
print('\rGetting data...', end='')
fetch_x_data = False
if os.path.exists(x_data_filename):
    x_data = pickle.load(gzip.open(x_data_filename))
else:
    fetch_x_data = True
    x_data = np.zeros((gene_count, seq_len, num_aa), dtype='int8')
names_data = np.empty((gene_count), dtype='string_')

# run through the file again to get sequence data and count the
# number of times each location is observed
with open(sys.argv[1]) as fin:
    header = fin.readline()
    count = 0
    for line in fin:
        print('\r{:0.2f}%             '.format(count / gene_count * 100), end='')
        sys.stdout.flush()
        e = line.split('\t')
        gene_id = e[0]
        if not gene_id in gene2prot:
            continue

        # if we don't yet have the sequence data, get it from ensembl
        # and store it in a onehot representation
        if fetch_x_data:
            aa_seq = '{s:{c}<{n}}'.format(
                    s=get_sequence(gene2prot[gene_id])[:seq_len], 
                    n=seq_len, c='-')
            for jj, c in enumerate(aa_seq):
                x_data[count, jj, aa2int[c]] = 1

        # get the list of locations for this protein.
        # we consider all levels of confidence of "Approved" and above
        # with equal weight.
        # on this loop through the file, we're just counting how many
        # times each location comes up
        locs = [y for x in e[3:6] for y in x.split(';') if not y == '']
        for l in locs:
            if not l in loc_counts:
                loc_counts[l] = 1
            else:
                loc_counts[l] += 1

        # store the name of the protein
        names_data[count] = gene2prot[gene_id]
        count += 1
print('\rDone.            ')

# we only want to consider locations that have at least a minimum number
# of counts
final_locs = [x for x in loc_counts if loc_counts[x] > min_loc_count]

# plus one to have an 'uncertain' category
num_localizations = len(final_locs) + 1

# store the names of the localization categories
pickle.dump(np.array(final_locs, dtype='string_'), 
            gzip.open(features_filename, 'wb'))

# run through the file one more time, this time to 
# get the localizations for each protein
y_data = np.zeros((gene_count, num_localizations), dtype='int8')
with open(sys.argv[1]) as fin:
    header = fin.readline()
    count = 0
    for line in fin:
        print('\r{:0.2f}%           '.format(count / gene_count * 100), end='')
        sys.stdout.flush()
        e = line.split('\t')
        gene_id = e[0]
        if not gene_id in gene2prot:
            continue

        # get the locations and store them iff they're in
        # the list of "final" locations that occurred enough
        # times
        locs = [y for x in e[3:6] for y in x.split(';') 
                                  if not y == '' 
                                  if y in final_locs]
        for l in locs:
            y_data[count, final_locs.index(l)] = 1
        if len(locs) == 0:
            y_data[count, -1] = 1

        count += 1
print('\rDone.            ')

# we did all the work of fetching it, so store the data
if fetch_x_data:
    pickle.dump(x_data, gzip.open(x_data_filename, "wb"))
pickle.dump(y_data,     gzip.open(y_data_filename, "wb"))
pickle.dump(names_data, gzip.open(names_data_filename, "wb"))

# make sure there's only a single one in each onehot entry
summed = np.sum(x_data, 2)
assert np.all(summed == np.ones(summed.shape, dtype='int8'))

# make sure there's at least one location per protein
summed2 = np.sum(y_data, 1)
assert np.all(summed2 >= np.ones(summed2.shape, dtype='int8'))
