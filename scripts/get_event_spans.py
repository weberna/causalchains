# Takes in a directory with decompevent.json files, a directory with conll files and outputs a training data file
import sys 
import os
import json
from predpatt import PredPatt, load_conllu

#usage: python decomp2train_format.py input.decompevent.json input.txt.conll output.json

def parse_conll_filename(fname): #taken from rrudinger's preprocessing scripts
    fname = fname.rstrip(".conllu")
    fname = fname.split("/")[-1]
    doc_id = ".".join(fname.split(".")[-2::])
    book_file_name = ".".join(fname.split(".")[:-2])
    genre = book_file_name.split(":")[0]
    book = book_file_name.split(":")[1:]
    return genre, book, doc_id

def parse_decomp_filename(fname): #taken from rrudinger's preprocessing scripts
    fname = fname.split(".event.decomp.json")[0]
    book = fname.split(":")[-1]
    genre = fname.split(":")[0].split("/")[-1]
    return genre, book

def predpatt2text(predicate): #Convert predpatt Predicate object to text
    token_list = predicate.tokens
    for arg in predicate.arguments:
        token_list = token_list + arg.tokens
    token_list = sorted(token_list, key=lambda tok: tok.position)
    return " ".join([x.text for x in token_list])


def concat_single_chunk(json_chunk_list): #Convert a list of sentential json obj to a single event chain, in a single dict
    #Filter out the following
    #-Non realis events
    event_chain = []
    event_texts = []
    for line in json_chunk_list:    
        events = line['syntactic-events']
        texts = line['event_text']
        fact = line['fact-predictions']
        pheads=line['predicate-head-idxs']
        assert len(pheads) == len(texts) and len(texts) <= len(events)  #Single predicate sometimes expanded into multiple predicates, in this case, just take the first event

        for i in range(len(texts)):
            if fact[i] == "pos":
                event_chain.append(events[i])
                event_texts.append(texts[i])

    assert len(event_chain) == len(event_texts)
    outdict = {"event_chain":event_chain, "event_texts":event_texts}
    return outdict
            

def convert_to_train(concat_chunk):
    instances = []
    for i in range(len(concat_chunk['event_chain'])-1):
        event_1 = concat_chunk['event_chain'][i]
        event_2 = concat_chunk['event_chain'][i+1]
        text_1 = concat_chunk['event_texts'][i]
        instances.append({'e1':event_1, 'e2':event_2, 'e1_text':text_1})
    return instances
        
        
    


if __name__ == "__main__":
    decompdir = sys.argv[1].rstrip("/")
    conlldir= sys.argv[2].rstrip("/")
    outfile = sys.argv[3]

#    decompfile = sys.argv[1]
#    conllfile= sys.argv[2]
#    outfile = sys.argv[3]

    output_writer = open(outfile, 'w')

    num_books = len(os.listdir(decompdir))
    num_processed = 0

    for decompfi in os.listdir(decompdir): #For each book
        decompfile = os.path.join(decompdir, decompfi)
        with open(decompfile, 'r') as decomp_fi:
            decomp_lines = decomp_fi.readlines()
   
        currgenre, currbook = parse_decomp_filename(decompfile)
        print("Processing {} of Genre: {}, Progress {}/{} ({} %)".format(currbook, currgenre, num_processed, num_books, num_processed/(num_books*1.0)))
        decomp_lines_json = [json.loads(x) for x in decomp_lines]

        book_conll_files = [fi for fi in os.listdir(conlldir) if parse_conll_filename(fi)[1][0] == currbook]

        for conllfi in book_conll_files: #For each chunk in the book
            conllfile = os.path.join(conlldir, conllfi)
            genre, book, doc_id = parse_conll_filename(conllfile)
            conll_iter = load_conllu(conllfile)
            decomp_lines_json_chunk = [x for x in decomp_lines_json if x['doc-id']==doc_id] #get the lines associated with this chunk
            line_idx = 0 #Where we are in the decomp json file

            for sent_id, parse in conll_iter:
                sent_id = int(sent_id.split('_')[1])

                if line_idx >= len(decomp_lines_json_chunk):
                    break

                if decomp_lines_json_chunk[line_idx]['sent-id'] == sent_id:
                    json_line = decomp_lines_json_chunk[line_idx]
                    ppat = PredPatt(parse)
                    pred_heads = json_line['predicate-head-idxs']
                    event_text = []
                    for head in pred_heads:
                        predicate = ppat.event_dict[ppat.tokens[head]]
                        pred_text = predpatt2text(predicate)
                        event_text.append(pred_text)
                    json_line['event_text'] = event_text
                    json_line['sprl-predictions'] = []
                    line_idx += 1
                        
    
#    print(list(zip(*concat_single_chunk(decomp_lines_json_chunk).values())))
            instances = convert_to_train(concat_single_chunk(decomp_lines_json_chunk))

            for inst in instances:
                json.dump(inst, output_writer)
                output_writer.write('\n')

        num_processed +=1

    output_writer.close()
    
#    for line in decomp_lines_json_chunk:
#        json.dump(line, output_writer)
#        output_writer.write('\n')
#    output_writer.close()


            
            
