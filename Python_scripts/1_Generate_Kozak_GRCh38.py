#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Packages
import os
import numpy as np
import pandas as pd

# Custom Functions

# Amino Acid Translator
def amino_acids(Sequence, iinput):
    # Please note that this uses the python indexing.  0 is reading frame 1, 1 is reading frame 2, and 2 is reading frame 3 for iinput.
    i = int(iinput)
    Peptide = []
    codon_dict = {'TTT':'F', 'TTC':'F', 'TTA':'L', 'TTG':'L', 'CTT':'L', 'CTC':'L', 'CTA':'L', 'CTG':'L',
                  'ATT':'I', 'ATC':'I', 'ATA':'I', 'ATG':'M', 'GTT':'V', 'GTC':'V', 'GTA':'V', 'GTG':'V',
                  'TCT':'S', 'TCC':'S', 'TCA':'S', 'TCG':'S', 'CCT':'P', 'CCC':'P', 'CCA':'P', 'CCG':'P',
                  'ACT':'T', 'ACC':'T', 'ACA':'T', 'ACG':'T', 'GCT':'A', 'GCC':'A', 'GCA':'A', 'GCG':'A',
                  'TAT':'Y', 'TAC':'Y', 'TAA':'*', 'TAG':'*', 'CAT':'H', 'CAC':'H', 'CAA':'Q', 'CAG':'Q',
                  'AAT':'N', 'AAC':'N', 'AAA':'K', 'AAG':'K', 'GAT':'D', 'GAC':'D', 'GAA':'E', 'GAG':'E',
                  'TGT':'C', 'TGC':'C', 'TGA':'*', 'TGG':'W', 'CGT':'R', 'CGC':'R', 'CGA':'R', 'CGG':'R',
                  'AGT':'S', 'AGC':'S', 'AGA':'R', 'AGG':'R', 'GGT':'G', 'GGC':'G', 'GGA':'G', 'GGG':'G',
                  # entries for N nucleotides
                  'TTN':'X', 'CTN':'L', 'ATN':'X', 'GTN':'V', 'TCN':'S', 'CCN':'P', 'ACN':'T', 'GCN':'A',
                  'TAN':'X', 'CAN':'X', 'AAN':'X', 'GAN':'X', 'TGN':'X', 'CGN':'R', 'AGN':'X', 'GGN':'G',
                  'TNT':'X', 'TNC':'X', 'TNA':'X', 'TNG':'X', 'CNT':'X', 'CNC':'X', 'CNA':'X', 'CNG':'X',
                  'ANT':'X', 'ANC':'X', 'ANA':'X', 'ANG':'X', 'GNT':'X', 'GNC':'X', 'GNA':'X', 'GNG':'X',
                  'NTT':'X', 'NTC':'X', 'NTA':'X', 'NTG':'X', 'NCT':'X', 'NCC':'X', 'NCA':'X', 'NCG':'X',
                  'NAT':'X', 'NAC':'X', 'NAA':'X', 'NAG':'X', 'NGT':'X', 'NGC':'X', 'NGA':'X', 'NGG':'X',
                  'NNT':'X', 'NNA':'X', 'NNC':'X', 'NNG':'X', 'NTN':'X', 'NAN':'X', 'NCN':'X', 'NGN':'X',
                  'TNN':'X', 'ANN':'X', 'CNN':'X', 'GNN':'X', 'NNN':'X',
                  # entries for Y nucleotides
                  'TTY':'F', 'CTY':'L', 'ATY':'I', 'GTY':'V', 'TCY':'S', 'CCY':'P', 'ACY':'T', 'GCY':'A',
                  'TAY':'Y', 'CAY':'H', 'AAY':'N', 'GAY':'D', 'TGY':'C', 'CGY':'R', 'AGY':'S', 'GGY':'G',
                  'TYT':'X', 'TYC':'X', 'TYA':'X', 'TYG':'X', 'CYT':'X', 'CYC':'X', 'CYA':'X', 'CYG':'X',
                  'AYT':'X', 'AYC':'X', 'AYA':'X', 'AYG':'X', 'GYT':'X', 'GYC':'X', 'GYA':'X', 'GYG':'X',
                  'YTT':'X', 'YTC':'X', 'YTA':'L', 'YTG':'L', 'YCT':'X', 'YCC':'X', 'YCA':'X', 'YCG':'X',
                  'YAT':'X', 'YAC':'X', 'YAA':'X', 'YAG':'X', 'YGT':'X', 'YGC':'X', 'YGA':'X', 'YGG':'X',
                  'YYT':'X', 'YYA':'X', 'YYC':'X', 'YYG':'X', 'YTY':'X', 'YAY':'X', 'YCY':'X', 'YGY':'X',
                  'TYY':'X', 'AYY':'X', 'CYY':'X', 'GYY':'X', 'YYY':'X',
                  # entries R nucleotides
                  'TTR':'L', 'CTR':'L', 'ATR':'X', 'GTR':'V', 'TCR':'S', 'CCR':'P', 'ACR':'T', 'GCR':'A',
                  'TAR':'*', 'CAR':'Q', 'AAR':'K', 'GAR':'E', 'TGR':'X', 'CGR':'R', 'AGR':'R', 'GGR':'G',
                  'TRT':'X', 'TRC':'X', 'TRA':'*', 'TRG':'X', 'CRT':'X', 'CRC':'X', 'CRA':'X', 'CRG':'X',
                  'ART':'X', 'ARC':'X', 'ARA':'X', 'ARG':'X', 'GRT':'X', 'GRC':'X', 'GRA':'X', 'GRG':'X',
                  'RTT':'X', 'RTC':'X', 'RTA':'L', 'RTG':'L', 'RCT':'X', 'RCC':'X', 'RCA':'X', 'RCG':'X',
                  'RAT':'X', 'RAC':'X', 'RAA':'X', 'RAG':'X', 'RGT':'X', 'RGC':'X', 'RGA':'X', 'RGG':'X',
                  'RRT':'X', 'RRA':'X', 'RRC':'X', 'RRG':'X', 'RTR':'X', 'RAR':'X', 'RCR':'X', 'RGR':'X',
                  'TRR':'X', 'ARR':'X', 'CRR':'X', 'GRR':'X', 'RRR':'X',
                  # entries V nucleotides
                  'TTV':'X', 'CTV':'L', 'ATV':'X', 'GTV':'V', 'TCV':'S', 'CCV':'P', 'ACV':'T', 'GCV':'A',
                  'TAV':'X', 'CAV':'X', 'AAV':'X', 'GAV':'X', 'TGV':'X', 'CGV':'R', 'AGV':'X', 'GGV':'G',
                  'TVT':'X', 'TVC':'X', 'TVA':'X', 'TVG':'X', 'CVT':'X', 'CVC':'X', 'CVA':'X', 'CVG':'X',
                  'AVT':'X', 'AVC':'X', 'AVA':'X', 'AVG':'X', 'GVT':'X', 'GVC':'X', 'GVA':'X', 'GVG':'X',
                  'VTT':'X', 'VTC':'X', 'VTA':'X', 'VTG':'X', 'VCT':'X', 'VCC':'X', 'VCA':'X', 'VCG':'X',
                  'VAT':'X', 'VAC':'X', 'VAA':'X', 'VAG':'X', 'VGT':'X', 'VGC':'X', 'VGA':'X', 'VGG':'X',
                  'VVT':'X', 'VVA':'X', 'VVC':'X', 'VVG':'X', 'VTV':'X', 'VAV':'X', 'VCV':'X', 'VGV':'X',
                  'TVV':'X', 'AVV':'X', 'CVV':'X', 'GVV':'X', 'VVV':'X',
                  # entries B nucleotides
                  'TTB':'X', 'CTB':'L', 'ATB':'X', 'GTB':'V', 'TCB':'S', 'CCB':'P', 'ACB':'T', 'GCB':'A',
                  'TAB':'X', 'CAB':'X', 'AAB':'X', 'GAB':'X', 'TGB':'X', 'CGB':'R', 'AGB':'X', 'GGB':'G',
                  'TBT':'X', 'TBC':'X', 'TBA':'X', 'TBG':'X', 'CBT':'X', 'CBC':'X', 'CBA':'X', 'CBG':'X',
                  'ABT':'X', 'ABC':'X', 'ABA':'X', 'ABG':'X', 'GBT':'X', 'GBC':'X', 'GBA':'X', 'GBG':'X',
                  'BTT':'X', 'BTC':'X', 'BTA':'X', 'BTG':'X', 'BCT':'X', 'BCC':'X', 'BCA':'X', 'BCG':'X',
                  'BAT':'X', 'BAC':'X', 'BAA':'X', 'BAG':'X', 'BGT':'X', 'BGC':'X', 'BGA':'X', 'BGG':'X',
                  'BBT':'X', 'BBA':'X', 'BBC':'X', 'BBG':'X', 'BTB':'X', 'BAB':'X', 'BCB':'X', 'BGB':'X',
                  'TBB':'X', 'ABB':'X', 'CBB':'X', 'GBB':'X', 'BBB':'X'}
    while i < len(Sequence):
        if len(Sequence[i:i+3]) == 3:
            Peptide.append(codon_dict[Sequence[i:i+3]])
        i = i+3
    return(''.join(Peptide))

# Find Index of max length seq
def Find_Index(List, MaxLength):
    for index, seq in enumerate(List):
        if len(seq) == MaxLength:
            return index

# read text file
def readmyfile(filename):
    with open(filename, 'r') as MyData:
        lines = []
        for fileline in MyData:
            lines.append(fileline.rstrip())
    return(lines)

# find kozak through exact string match between RNA translation and protein
def find_kozak_method_1(translation,readframe,rnaseq):
    rf = readframe-1
    possible_kozak = []
    for prot in prot_sequences:
        if translation.find(prot) != -1:
            possible_kozak.append(rnaseq[translation.find(prot)*3+rf-6:translation.find(prot)*3+rf])
            break #this will cause us to only call the first match.  Some may have multiple matches.
                  #however, since this would mean 25,262,185,494 checks per reading frame at time of
                  #writing, I am reducing to limit computational load.  However, note that removing
                  #will break downstream code.
    return(possible_kozak)

# find kozak through through 'kozak' for longest string in any rf
def find_kozak_method_2(translation1, # reading frame 1
                        translation2, # reading frame 2
                        translation3, # reading frame 3
                        rnaseq):
    # set up intermediate lists
    maxseqlens = []
    # set up rf1
    seqs1 = translation1.split('*')
    seqlens1 = []   
    for seq in seqs1:
        seqlens1.append(len(seq))
    maxseqlens.append(max(seqlens1))
    # set up rf2
    seqs2 = translation2.split('*')
    seqlens2 = []
    for seq in seqs2:
        seqlens2.append(len(seq))
    maxseqlens.append(max(seqlens2))
    # set up rf3
    seqs3 = translation3.split('*')
    seqlens3 = []
    for seq in seqs3:
        seqlens3.append(len(seq))
    maxseqlens.append(max(seqlens3))
    
    # case of rf1 has max
    if maxseqlens.index(max(maxseqlens)) == 0:
        longest_string = seqs1[seqlens1.index(max(seqlens1))]
        if longest_string.find('M') == -1:
            return(np.nan) # no methionine, aka no translation
        else:
            longest_string_translated = longest_string[longest_string.find('M'):]
            return(rnaseq[translation1.find(longest_string_translated)*3+0-6:translation1.find(longest_string_translated)*3+0])
    
    # case of rf2 has max
    elif maxseqlens.index(max(maxseqlens)) == 1:
        longest_string = seqs2[seqlens2.index(max(seqlens2))]
        if longest_string.find('M') == -1:
            return(np.nan) # no methionine, aka no translation
        else:
            longest_string_translated = longest_string[longest_string.find('M'):]
            return(rnaseq[translation2.find(longest_string_translated)*3+1-6:translation2.find(longest_string_translated)*3+1])
    
    # case of rf3 has max
    elif maxseqlens.index(max(maxseqlens)) == 2:
        longest_string = seqs3[seqlens3.index(max(seqlens3))]
        if longest_string.find('M') == -1:
            return(np.nan) # no methionine, aka no translation
        else:
            longest_string_translated = longest_string[longest_string.find('M'):]
            return(rnaseq[translation3.find(longest_string_translated)*3+2-6:translation3.find(longest_string_translated)*3+2])
    
    # return NAN if something didn't work otherwise
    else:
        return(np.nan)
    
# merge outputs for each reading frame of finding kozak through method 1
def unify_method_1(rf1,rf2,rf3):
    if str(rf1) != '[]':
        return(str(rf1).replace('[','').replace(']','').replace("'",''))
    elif str(rf2) != '[]':
        return(str(rf2).replace('[','').replace(']','').replace("'",''))
    elif str(rf3) != '[]':
        return(str(rf3).replace('[','').replace(']','').replace("'",''))
    else:
        return(np.nan)

# extract NM identifier in block 1
def extract_nm(input):
    return(input[input.find('>')+1:input.find(' ')])

# extract NM identifier in block 2
def extract_nm2(input):
    return(input[:input.find('(')])

# outputs location of mutations around kozak sequence
def find_location(mut):
    if mut[2] == '+':
        return('intron')
    elif mut[2] ==  '-':
        return('intron')
    else:
        return('kozak')
    
# outputs old kozak sequence and updated mutated kozak sequence.  Supplies appropriate error or 
# 'errors' if excluded situations or failures are encountered.
def mutate_kozak(m1_kozak, m2_kozak, mut_location, mut_type, mut_details):
    # return error:no_kozak_found if no kozak from method 1 or method 2.
    if pd.isnull(m1_kozak) == True:
        if pd.isnull(m2_kozak) == True:
            return('error:no_kozak_found','error:no_kozak_found')
    # return error:intron_mutation if mutation in intron.  We will not be
    # handling such cases due to complexity and scope.
    if mut_location == 'intron':
        return('error:intron_mutation','error:intron_mutation')
    # return error:microsatellite if mutation is a microsatellite.  We will
    # not be handling such cases due to complexity and scope.
    if mut_type == 'Microsatellite':
        return('error:microsatellite','error:microsatellite')
    # making variables to simplify reading of rest of this function.
    position = int(mut_details[0:2]) # make sure this is a number!
    oldnt = mut_details[2]
    # handle case of duplication.  There is only one at type of writing, but
    # will be generalizable in case more found in future.  In principle, would
    # be same for a deletion, but would require going back into the RNA to collect.
    if mut_type == 'Duplication':
        if pd.isnull(m1_kozak) == False:
            if m1_kozak[position] == oldnt:
                return(m1_kozak,m1_kozak[-5:position] + 2*newnt + m1_kozak[position+1:])
        elif pd.isnull(m2_kozak) == False:
            if m2_kozak[position] == oldnt:
                return(m2_kozak,m2_kozak[-5:position] + 2*newnt + m2_kozak[position+1:])
        else:
            # return error:mutation_not_match_kozak when mutation doesn't match either method 1 or 2 kozak.
            return('error:mutation_not_match_kozak','error:mutation_not_match_kozak')
    # making variables to simplify reading of rest of this function.      
    newnt = mut_details[4]
    if mut_type == 'single nucleotide variant':
        if pd.isnull(m1_kozak) == False:
            if m1_kozak[position] == oldnt:
                if position == -1:
                    return(m1_kozak, m1_kozak[-6:position] + newnt)
                else:
                    return(m1_kozak, m1_kozak[-6:position] + newnt + m1_kozak[position+1:])
            elif pd.isnull(m2_kozak) == False:
                if m2_kozak[position] == oldnt:
                    if position == -1:
                        return(m2_kozak, m2_kozak[-6:position] + newnt)
                    else:
                        return(m2_kozak, m2_kozak[-6:position] + newnt + m2_kozak[position+1:])
                else:
                    return('error:mutation_not_match_kozak','error:mutation_not_match_kozak')
            else:
                return('error:mutation_not_match_kozak','error:mutation_not_match_kozak')
        elif pd.isnull(m2_kozak) == False:
            if m2_kozak[position] == oldnt:
                if position == -1:
                    return(m2_kozak, m2_kozak[-6:position] + newnt)
                else:
                    return(m2_kozak, m2_kozak[-6:position] + newnt + m2_kozak[position+1:])
            else:
                return('error:mutation_not_match_kozak','error:mutation_not_match_kozak')
        else:
            # return error:mutation_not_match_kozak when mutation doesn't match either method 1 or 2 kozak.
            return('error:mutation_not_match_kozak','error:mutation_not_match_kozak')
    else:
        # output if everything else failed
        return('error:function_failed','error:function_failed')
    
### Block 0 ###

# Below parses the RNA and Protein file from
# https://www.ncbi.nlm.nih.gov/genome/guide/human/ for GRCh38, and prepares
# a background list (info about the sequence data) and a sequence file (the actual sequence data).
# Use the Refseq Transcripts and Refseq Proteins downloads in this situation.
# --> https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GRCh38_latest_rna.fna.gz
# --> https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GRCh38_latest_protein.faa.gz
# For later, we will also need the variant summary text file from Clinvar.  This can be found at:
# https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/archive/
# select the variant_summary_(date).txt.gz with the most current date for this.
# On mac, after downloading, go to terminal and use: gunzip (drag file after space into terminal) to unzip each of these files.
# On windows, go to terminal and use: tar -xvzf (filepath) to unzip each of these files
# make sure files are in a folder titled 'data' at same directory position as this script

print('This script will take some time to run.')
print('Statements will be printed throughout to update on pprogress.')
print('Estimated run time:')
print('Block 1: 15 hours')
print('Block 2: 10 minutes')
print('Block 3: 1 minute')

GRCH38_rna = readmyfile('data/GRCh38_latest_rna.fna')
GRCH38_prot = readmyfile('data/GRCh38_latest_protein.faa')
clinvar_dat = pd.read_csv('data/variant_summary_2025-02.txt', sep='\t', header = 0, low_memory = False)
print('Data ingested.')

# makes output folder
if not os.path.exists('output'):
    os.makedirs('output')

### Block 1 ###

## Code Block 1 is designed to generate KOZAK data by utilizing the data from GRCh38.

print('Beginning Block 1: Generating Kozaks')
i = 0
rna_headers = []
rna_sequences = []
temp_sequence = []
while i < len(GRCH38_rna):
    if GRCH38_rna[i][0] == '>':
        rna_sequences.append(''.join(temp_sequence))
        temp_sequence = []
        rna_headers.append(GRCH38_rna[i])
    else:
        temp_sequence.append(GRCH38_rna[i])
    i +=1
rna_sequences.append(''.join(temp_sequence))
rna_sequences = rna_sequences[1:] # remove the empty first row that was generated in seqs
print('RNA Sequences ready.')

i = 0
prot_headers = []
prot_sequences = []
temp_sequence = []
while i < len(GRCH38_prot):
    if GRCH38_prot[i][0] == '>':
        prot_sequences.append(''.join(temp_sequence))
        temp_sequence = []
        prot_headers.append(GRCH38_prot[i])
    else:
        temp_sequence.append(GRCH38_prot[i])
    i +=1
prot_sequences.append(''.join(temp_sequence))
prot_sequences = prot_sequences[1:] # remove the empty first row that was generated in seqs
print('Protein Sequences ready.')

# Garbage Collection
del(temp_sequence)
del(GRCH38_rna)
del(GRCH38_prot)

kozak_df = pd.DataFrame({
    'grch38_headers': rna_headers,
    'grch38_rna_seqs': rna_sequences})

        ## Section 2: Find Kozak Sequences ##

## Approach 1
# Here, we will attempt to find the KOZAK sequence by translating the RNA sequences and
# cross referencing them with the protein dataset.

# create translations in each reading frame
kozak_df['reading_frame_1'] = kozak_df.apply(lambda x: amino_acids(x.grch38_rna_seqs,0), axis = 1)
kozak_df['reading_frame_2'] = kozak_df.apply(lambda x: amino_acids(x.grch38_rna_seqs,1), axis = 1)
kozak_df['reading_frame_3'] = kozak_df.apply(lambda x: amino_acids(x.grch38_rna_seqs,2), axis = 1)
print('reading frames translated')

# find string matches
kozak_df['rf1_m1_kozaks'] = kozak_df.apply(lambda x: find_kozak_method_1(x.reading_frame_1,
                                                                         1,
                                                                         x.grch38_rna_seqs),axis = 1)
print('Reading frame 1 method 1 done.')
kozak_df['rf2_m1_kozaks'] = kozak_df.apply(lambda x: find_kozak_method_1(x.reading_frame_2,
                                                                         2,
                                                                         x.grch38_rna_seqs),axis = 1)
print('Reading frame 2 method 1 done.')
kozak_df['rf3_m1_kozaks'] = kozak_df.apply(lambda x: find_kozak_method_1(x.reading_frame_3,
                                                                         3,
                                                                         x.grch38_rna_seqs), axis = 1)
print('Reading frame 3 method 1 done.')

kozak_df['m1_kozak'] = kozak_df.apply(lambda x: unify_method_1(x.rf1_m1_kozaks,
                                                               x.rf2_m1_kozaks,
                                                               x.rf3_m1_kozaks),axis = 1)
print('Method 1 done.')

## Approach 2
# This approach searches for the longest protein sequence in each reading frame, and finds
# the kozak for that sequence.  This is less optimal as an approach, but it's included to be comprehensive.

kozak_df['m2_kozak'] = kozak_df.apply(lambda x: find_kozak_method_2(x.reading_frame_1,
                                                                    x.reading_frame_2,
                                                                    x.reading_frame_3,
                                                                    x.grch38_rna_seqs), axis = 1)
print('method 2 done.')

# pull out nm number, which will be used later to merge with clinvar data
kozak_df['nm'] = kozak_df.apply(lambda x: extract_nm(x.grch38_headers), axis = 1)

# replace field that's entirely space (or empty) with NaN
kozak_df = kozak_df.replace('', np.nan, regex = True)

kozak_df.to_csv('output/block_1_output.csv')
print('Block 1 output saved as output/block_1_output.csv')

# Garbage Collection
del(rna_headers)
del(rna_sequences)
del(prot_headers)
del(prot_sequences)
del(i)

### Block 2 ###

## Code Block 2 is designed to set up the dataframe of clinvar variants we will look at the kozak mutations of.

print('Beginning Block 2: Isolating Kozak Mutations in Clinvar')

# To start, we will need to remove all GRCh37 entries from the data, which are found in the Assembly column.
clinvar_dat = clinvar_dat[clinvar_dat['Assembly'] == 'GRCh38']

# Now, we will remove all columns that are extraneous for our purposes.
clinvar_dat = clinvar_dat[['Type','Name','GeneID','GeneSymbol',
                           'HGNC_ID','ClinicalSignificance','Chromosome',
                           'Start','Stop','Oncogenicity']]

# The two main columns that we will want to start with are Type and Name.
# Type contains mutation type (indel/SNV/etc.)
# Name contains the actual mutation information that we will want to get at.
# e.g. NM_014630.3(ZNF592):c.3136G>A (p.Gly1046Arg)
# :c. comes right before the mutation
# next is the position(s) that are affected (3136)
# then the change(s) (G --> A)
# what we will do first is make a new column off of the name column with the mutation data after :c.,
# after which we will isolate out mutations affecting the kozak positions.
clinvar_dat['mutation'] = clinvar_dat.apply(lambda x: x.Name[x.Name.find(':c.')+3:], axis = 1)
clinvar_dat = clinvar_dat[clinvar_dat['mutation'].str.startswith(('-1A','-1T','-1C','-1G','-2A','-2T','-2C','-2G',
                                                                  '-3A','-3T','-3C','-3G','-4A','-4T','-4C','-4G',
                                                                  '-5A','-5T','-5C','-5G','-6A','-6T','-6C','-6G',
                                                                  '-1-','-1+','-2-','-2+','-3-','-3+','-4-','-4+',
                                                                  '-5-','-5+','-6+'))]
# To simplify later steps, we will identify which mutations affect introns here.
clinvar_dat['mutation_location'] = clinvar_dat.apply(lambda x: find_location(x.mutation), axis = 1)

clinvar_dat['nm'] = clinvar_dat.apply(lambda x: extract_nm2(x.Name), axis = 1)

# replace field that's entirely space (or empty) with NaN
clinvar_dat = clinvar_dat.replace('', np.nan, regex = True)

clinvar_dat.to_csv('output/block_2_output.csv')
print('Block 2 output saved as output/block_2_output.csv')

### Block 3 ###

## Code Block 3 will merge the two prior dataframes, and apply kozak mutations.

print('Beginning Block 3: Merging the Data and Apply Mutation')

# To merge the data, we will use the NM values.
# We will be focusing on using the keys in the clinvar_dat frame.
# This will provide a smaller dataframe to work within.
final_dat = pd.merge(kozak_df, clinvar_dat, how = 'right', on =  ['nm'])

# replace field that's entirely space (or empty) with NaN
# especially important here - empty cells cause failure
final_dat = final_dat.replace('', np.nan, regex = True)

final_dat[['old_kozak','new_kozak']] = final_dat.apply(lambda x: mutate_kozak(x.m1_kozak,
                                                                              x.m2_kozak,
                                                                              x.mutation_location,
                                                                              x.Type,
                                                                              x.mutation),
                                                       axis = 'columns',
                                                       result_type = 'expand')

# Save output
final_dat.to_csv('output/clinvar_kozak_mutations.csv')
print('Block 3 output saved as output/clinvar_kozak_mutations.csv')

