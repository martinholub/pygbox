import os
import json

def load_metadata(tif_file):
    """ Parse metadata from a text file.
    """

    txt_file = os.path.splitext(tif_file)[0] + ".txt"
    metadata = {}

    # Extratc header
    lines = extract_header(txt_file)

    headline = lines[0].split("\t")[:-1]
    name = headline[0].split(":")[-1].strip()

    metadata = {
        "name" : name,
        "date" : headline[1],
        "time" : headline[2],
    }

    n_dims = 0
    n_frames = 1
    for row in lines[5:]:
        n_dims += 1
        parts = [l.strip() for l in row.split(":")]
        metadata[parts[0]] = int(parts[1])
        n_frames *= int(parts[1])

    metadata["ndims"] = n_dims
    metadata["nframes"] = n_frames

    # Extract order of repetitions
    repeat_order = extract_repeat_order(txt_file)
    metadata["order"] = list(repeat_order.keys())
    metadata["n_repeats"] = list(repeat_order.values())

    return metadata

def extract_header(file):
    headlines = []
    n_empty = 0
    with open(txt_file, "r") as f:
        for line in f:
            if line == "\n":
                n_empty += 1
                if n_empty > 1:
                    break
                else:
                    continue
            else:
                n_empty = 0
                headlines.append(line)
    return headlines

def extract_repeat_order(file):
    # most likely it is just inverse of the order in metadata file!
    with open(txt_file, "r") as f:
        lines = f.read().splitlines()

    lines = [l.lstrip() for l in lines]
    good_starts = ("Repeat ", "XY Positions ", "Move Channel ")
    good_lines = [l.startswith(good_starts) for l in lines]
    lines_sel = [l for l,tf in zip(lines, good_lines) if tf]

    repeats = {}
    is_xy = 0
    for line in lines_sel:
        line = line.split()
        if line[2] == "XY":
            is_xy = 1
            continue
        elif line[0] == "Repeat" or line[0] == "XY":
            if not is_xy:
                variable = line[1]
                value = line[3] if variable != "Z" else line[6]
            else:
                variable = line[0]
                value = line[-1].replace("(","").replace(")","")
                is_xy = 0
        elif line[0] == "Move":
            if "CH" not in repeats.keys(): repeats["CH"] = 0
            repeats["CH"] += 1

        repeats[variable] = int(value)

    return repeats

def dump_json(data, save_path):
    root, ext = os.path.splitext(save_path)
    if not ext.endswith(".json"):
        save_path = root + ".json"

    with open(save_path, 'w') as wf:
        json.dump(data, wf, sort_keys = False, indent = 4)
        print("Saved data dict to {}".format(save_path))

def load_json(fpath):
    root, ext = os.path.splitext(fpath)
    if not ext.endswith(".json"):
        fpath = root + ".json"
    with open(fpath, 'r') as jsn:
        content =  json.load(jsn)
    return content
