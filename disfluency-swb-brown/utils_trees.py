import os
import re

import tb

def get_span_tagged_text(text):
    counter = 1
    new_text = []
    for line in text.split("\n"):
        if "CODE" in line:
            if counter == 4:
                new_text.append("( (CODE <SEP>))")
                counter = 0
            else:
                new_text.append("( (CODE ))")
            counter += 1
        else:
            new_text.append(line)
    new_text = "\n".join(new_text)

    return new_text

def get_brown_span_tagged_text(text):
    counter = 1
    new_text = []
    for line in text.split("\n"):
        if "( (S" in line:
            if counter == 2:
                new_text.append("( (CODE <SEP>))\n( (S")
                counter = 0
            else:
                new_text.append("( (CODE ))\n( (S")
            counter += 1
        else:
            new_text.append(line)
    new_text = "\n".join(new_text)

    return new_text

def copy_files_over(in_filepaths, out_dir, withTags, isBrown):

    for filepath in in_filepaths:

        # initialize the contents for each file
        contents = ""

        # read the input file
        with open(filepath) as f:
            contents = f.read()

        if withTags:
            if isBrown:
                contents = get_brown_span_tagged_text(contents)
            else:
                contents = get_span_tagged_text(contents)

        # write to the output file
        filename = filepath.split("/")[-1]
        with open(os.path.join(out_dir, filename), mode="w") as f:
            f.write(contents)


# adapted from tb.py
def get_disfluent_leaf_nodes(tree):
    """Yields the terminal or leaf nodes of tree."""
    def visit(node):
        if isinstance(node, list):
            for child in node[1:]:
                yield from visit(child)
        else:
            yield node
    yield from visit(tree) 

# from: https://stackoverflow.com/questions/2158395/flatten-an-irregular-arbitrarily-nested-list-of-lists/2158532#2158532
from collections.abc import Iterable
def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

# adapted from tb.py
def is_period_in_subtree(tree):
    if "." in flatten(tree):
        return True

# adapted from tb.py, yields leaf nodes
def get_fluent_leaf_nodes(tree):
    """Yields the terminal or leaf nodes of tree."""
    def visit(node):
        if isinstance(node, list):
            for child in node[1:]:
                if any(x in ["EDITED", "INTJ", "PRN"] for x in child):
                    if is_period_in_subtree(child):
                        yield "."
                else:
                    yield from visit(child)
        else:
            yield node
    yield from visit(tree) 

# this function is used for getting (our way of) formatting transcripts from the trees
def get_clean_transcript_from_tree_file(filepath, get_disfluent):
    tb_file = tb.read_file(filepath)
    if get_disfluent:
        s = list(get_disfluent_leaf_nodes(tb_file))
    else:  # get the fluent leaf nodes
        s = list(get_fluent_leaf_nodes(tb_file))
    
    r = []
    for i in s:
        i = i.replace("E_S", ".")
        i = i.replace("N_S", ".")
        i = i.replace("MUMBLEx", "")
        i = i.replace(",", "")
        i = i.replace("[", "")
        i = i.replace("+", "")
        i = i.replace("]", "")
        i = i.replace("\\", "")
        i = re.sub(r"\*.*\*-[0-9]","",i,flags=re.IGNORECASE)
        i = re.sub(r".*\*-[0-9]","",i,flags=re.IGNORECASE)
        i = re.sub(r"\*-[0-9]","",i,flags=re.IGNORECASE)
        i = i.replace("*?*", "")
        i = i.replace("*", "")
        i = re.sub(r"\d+","",i,flags=re.IGNORECASE)
        r.append(i)

    def capitalize_after_period(text):
        return re.sub(r'(?<=\. )\w', lambda m: m.group().upper(), text)

    def capitalize_after_period_and_SEP(text):
        return re.sub(r'(?<=\. <SEP> )\w', lambda m: m.group().upper(), text)

    r = " ".join(r)
    
    r = r.replace(" n't", "n't")
    r = re.sub(r"(\w+)\s+(')", r"\1\2", r)
    r = r.replace(" .", ".")  # to turn multiple periods into 1 period
    r = re.sub(r'\s+',' ',r)
    r = r.replace(" .", ".")
    r = capitalize_after_period(r)
    r = capitalize_after_period_and_SEP(r)
    r = r.replace("<SEP>.", "<SEP>")
    r = re.sub(r'\.+','.',r)
    r = r.replace(". ?.", "?")
    r = r.replace("?.", "?")
    r = r.replace(" ?", "?")
    r = re.sub(r"^\s+\.","",r)  # to clean up " ." at the beginning
    r = r.strip()
    r = r[0].upper() + r[1:]
    return r


def get_consistent_transcript(filepath, get_disfluent, isBrown):
    r = ""
    r = get_clean_transcript_from_tree_file(filepath=filepath, get_disfluent=get_disfluent)
    r = r.replace("?","")
    
    if isBrown:
        r = r.replace("`", "")
        r = r.replace("\"", "")
        r = r.replace("\'", "")
        r = r.replace(";", "")
        r = r.replace("!", "")
        r = re.sub(r'\s+',' ',r)
    
    return r