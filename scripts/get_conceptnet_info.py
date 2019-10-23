import sys 
import os
import json
import argparse
import requests
import time
#import causalchains.utils.data_utils as du
import pickle
from predpatt import PredPatt, load_conllu



CAUSES="rel=/r/Causes"
PREREQ="rel=/r/HasPrerequisite"
CAUSE_DESIRE="rel=/r/CausesDesire"
SUBEVENT="rel=/r/HasSubevent"
URL="http://api.conceptnet.io/query?"
START = "start=/c/en/"
END ="end=/c/en/"
ANY ="node=/c/en/"

def load_vocab(filename):
    #load vocab from json file
    with open(filename, 'rb') as fi:
        voc = pickle.load(fi)
    return voc

def query_cn(query):
    obj = requests.get(query + "&limit=1000")
    time.sleep(1.0) #So we don't get kicked out
    return obj.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', type=str, help='Directory with decomp event json files' )
    parser.add_argument('--outfile', type=str)
    args = parser.parse_args()

    evocab = load_vocab(args.vocab)
    verbs = [x.split('->')[0] for x in evocab.itos[2:] if x.split('->')[1] in ['nsubj', 'dobj'] and len(x.split('->')) ==2]
    verbs = [i for n, i in enumerate(verbs) if i not in verbs[:n]] #remove duplicates

    verbs=verbs[200:300] #TESTING
        
    output_writer = open(args.outfile, 'w')

    for i, v in enumerate(verbs):
        print("Processing Verb {} out of {}".format(i, len(verbs)))

        causes_query=URL + END + v + '&' + CAUSES
        prereq_query=URL + START + v + '&' + PREREQ
        causes_desire_query=URL + END + v + '&' + CAUSE_DESIRE
        subevent_query = URL + ANY + v + '&' + SUBEVENT

        causes_res=query_cn(causes_query)
        json.dump(causes_res, output_writer)
        output_writer.write('\n')

        prereq_res=query_cn(prereq_query)
        json.dump(prereq_res, output_writer)
        output_writer.write('\n')
       
        causes_des_res=query_cn(causes_desire_query)
        json.dump(causes_des_res, output_writer)
        output_writer.write('\n')
      
        subevent_res=query_cn(subevent_query)
        json.dump(subevent_res, output_writer)
        output_writer.write('\n')










#        num_processed +=1

#    output_writer.close()
    
#    for line in decomp_lines_json_chunk:
#        json.dump(line, output_writer)
#        output_writer.write('\n')
    output_writer.close()


            
            
