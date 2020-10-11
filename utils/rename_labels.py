#!/usr/bin/env python3

import argparse
import fnmatch
import os

description_text = """\
!!!Buggy Script! USE AT YOUR OWN RISK!!!

Use this script to rename all your labeled image from either tracking.py or labeler.py.

REMINDER: Edit this script to ensure your label in the script match the label text files. And DO NOT 
    Re-run this script once it's done.
"""

epilog_text = """\
    example:
        ./rename_labels.py [folder] rename labels in [folder]
"""

parser = argparse.ArgumentParser(
    description=description_text,
    epilog=epilog_text,
    formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument("folder", type=str,
                    help="Folder path")

args = parser.parse_args()


def rename(directory):
    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, "*.txt"):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                is_modified = False
                lines = f.readlines()
                new_lines = []

                for line in lines[0:]:
                    if "s" in line:
                        new_lines.append(line.replace("s", "skystone"))
                    elif "r" in line:
                        new_lines.append(line.replace("r", "stone"))
                if len(new_lines) > 0:
                    with open(filepath, "w") as f2:
                        f2.write("".join(new_lines))
                        is_modified = True
            print("Current File: " + filename + ", modified?:" + str(is_modified))


if __name__ == "__main__":
    rename(args.folder)
