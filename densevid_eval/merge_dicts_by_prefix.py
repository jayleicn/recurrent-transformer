import json
import glob
import argparse


def merge_dicts(list_dicts):
    merged_dict = list_dicts[0].copy()
    for i in range(1, len(list_dicts)):
        merged_dict.update(list_dicts[i])
    return merged_dict


def save_json_pretty(data, file_path):
    """save formatted json, use this one for some json config files"""
    with open(file_path, "w") as f:
        f.write(json.dumps(data, indent=4, sort_keys=True))


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def merge_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--template", type=str,
                        help="path template for glob.glob, all files with the same template will be merged")
    parser.add_argument("-o", "--output", type=str, help="path to the output")
    args = parser.parse_args()

    print("args.template {}".format(args.template))
    prefix_filepaths = glob.glob(args.template)  # list of filepaths
    print("Loading {} files:\n{}".format(len(prefix_filepaths), "\n".join(prefix_filepaths)))
    merged_dict = merge_dicts([load_json(e) for e in prefix_filepaths])

    save_json_pretty(merged_dict, args.output)


if __name__ == '__main__':
    merge_main()
