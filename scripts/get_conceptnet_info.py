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

def query_main(args):
    print("QUERYING CONCEPT NET")
    evocab = load_vocab(args.vocab)
    verbs = [x.split('->')[0] for x in evocab.itos[2:] if x.split('->')[1] in ['nsubj', 'dobj'] and len(x.split('->')) ==2]
    verbs = [i for n, i in enumerate(verbs) if i not in verbs[:n]] #remove duplicates

    output_writer = open(args.outfile, 'w')

    for i, v in enumerate(verbs):
        print("Processing Verb {} out of {}".format(i, len(verbs)))

        causes_query=URL + END + v + '&' + CAUSES
        prereq_query=URL + START + v + '&' + PREREQ
        causes_desire_query=URL + END + v + '&' + CAUSE_DESIRE

        causes_res=query_cn(causes_query)
        json.dump(causes_res, output_writer)
        output_writer.write('\n')

        prereq_res=query_cn(prereq_query)
        json.dump(prereq_res, output_writer)
        output_writer.write('\n')
       
        causes_des_res=query_cn(causes_desire_query)
        json.dump(causes_des_res, output_writer)
        output_writer.write('\n')
      

    output_writer.close()

def convert_main(args):
    print("CONVERTING TO REQUIRED JSON")
    with open(args.vocab, 'r') as fi:
        queries = fi.readlines()
    queries = [json.loads(x) for x in queries]

    event_dict = {}
    for query_result in queries:
        query = query_result['@id'].split('?')[1]
        print(query)
        query_1 = query.split('&')[0]
        query_2 = query.split('&')[1]
        if query_1.split('=')[0] == 'rel':
            rel = query_1.split('/')[-1]
            predicate = query_2.split('/')[-1]
        else:
            predicate = query_1.split('/')[-1]
            rel = query_2.split('/')[-1]


        assert rel in ['Causes', 'HasPrerequisite', 'CausesDesire']
        if predicate not in event_dict:
            event_dict[predicate] = []

        if rel == 'Causes' or rel == 'CausesDesire': #for these relations, look at start node
            othernode = 'start'
        elif rel == 'HasPrerequisite': #for this relation, look at end node
            othernode = 'end'

        for edge in query_result['edges']:
            node = edge[othernode]
            label = " ".join(node['term'].split('/')[-1].split("_"))
            event_dict[predicate].append((label, edge['weight']))

    with open(args.outfile, 'w') as fi:
        json.dump(event_dict, fi)

            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', type=str, help='Vocab if --query is on, ELSE, use for the inputfile, for converting results to new JSON format' )
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--query', action='store_true')
    args = parser.parse_args()

    if args.query:
        query_main(args)
    else:
        convert_main(args)



            
            
