import sys 
import os
import json
import argparse
import spacy
from predpatt import PredPatt, load_conllu
import xml.etree.ElementTree as ET

#Convert copa into json with e1, preve1_intext, allprev, e1_text, alt1, alt1_text, alt2, alt2_text, correct=[1 or 2], asks-for


def predpatt2text(predicate): #Convert predpatt Predicate object to text
    token_list = predicate.tokens
    for arg in predicate.arguments:
        token_list = token_list + arg.tokens
    token_list = sorted(token_list, key=lambda tok: tok.position)
    return " ".join([x.text for x in token_list])


def get_events_and_text(sent):
    """
    sent is a spacy parsed sentence (parsed through the default English spacy pipeline)
    Extract the events and the text of the events from a line of COPA
    """
    text = sent.text
    sorels = ['nsubj', 'dobj', 'iobj']
    outputs = []
    pp = PredPatt.from_sentence(text)
    events = pp.events
    for event in events:
        position = event.position
        args =event.arguments
        event_rels = {}
        for a in args:
            head = a.root
            govrel = head.gov_rel
            event_rels[govrel] = head
        lemma = sent[position].lemma_
        if 'nsubj' in event_rels:
            e1 = lemma + '->nsubj'
            e1_text = predpatt2text(event)
        elif 'dobj' in event_rels:
            e1 =lemma + '->dobj'
            e1_text = predpatt2text(event)
        elif 'iobj' in event_rels:
            e1 =lemma + '->iobj'
            e1_text = predpatt2text(event)
        else:
            e1 =lemma + '->nsubj'
            e1_text = predpatt2text(event)

        outputs.append({'e1':e1, 'e1_text':e1_text})
    return outputs

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CopaProc')
    parser.add_argument('--copafile', type=str)
    parser.add_argument('--outfile', type=str)
    args = parser.parse_args()

    tree = ET.parse(args.copafile)
    root = tree.getroot()
    nlp = spacy.load('en')

    outfi = open(args.outfile, 'w')

    for item in root.findall('item'):
        asks_for = item.attrib['asks-for']
        correct_ans = int(item.attrib['most-plausible-alternative'])

        premise_text = item.find('p').text
        a1_text = item.find('a1').text
        a2_text = item.find('a2').text
        
        premise_events = get_events_and_text(nlp(premise_text))
        a1_events = get_events_and_text(nlp(a1_text))
        a2_events = get_events_and_text(nlp(a2_text))

        if premise_events and a1_events and a2_events:
            print("Writing Instance!")
            output = {'premise_e1': premise_events[0]['e1'] , 'premise_e1_text': premise_events[0]['e1_text'],
                      'a1_e1': a1_events[0]['e1']  , 'a1_e1_text': a1_events[0]['e1_text'], 
                      'a2_e1': a2_events[0]['e1'], 'a2_e1_text':a2_events[0]['e1_text'],
                      'asks-for':asks_for, 'correct_ans':correct_ans}
            json.dump(output, outfi)
            outfi.write('\n')
            

    outfi.close()
    


            
            
