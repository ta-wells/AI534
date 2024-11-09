#!/usr/bin/env python3

__author__ = "Liang Huang"

'''
Usage: cat income.test.predicted | python3 validate.py
OR:    python validate.py income.test.predicted              (this might work better on Windows)

Note: this code works for both Python 3 and Python 2.
'''

import sys

def error(msg, example=False, formatting_ok=False):
    print("ERROR:", msg)
    if not formatting_ok:
        print("PLEASE CHECK YOUR CODE AND REFER TO income.dev.txt FOR THE FORMAT.")
    else:
        print("PLEASE DOUBLE CHECK YOUR BINARIZATION AND K-MEANS CODE.")
    if example:
        print("FOR YOUR INFO, EACH LINE SHOULD BE LIKE THE FOLLOWING:")
        print("37, Private, Bachelors, Separated, Other-service, White, Male, 70, England, <=50K")
    exit(1)

def truncate(s):
    return s if len(s) < 150 else "%s ...(line too long; truncated)... %s" % (s[:100], s[-20:])

if __name__ == "__main__":
    infile = sys.stdin if len(sys.argv) == 1 else open(sys.argv[1]) # either standard input or first command-line argument

    pos = 0

    i = 0 # just in case of empty file
    for i, line in enumerate(infile, 5999):
        fields = line.strip().split(",")
        if i == 5999:
            if fields != ['id', 'age', 'sector', 'edu', 'marriage', 'occupation', 'race', 'sex', 'hours', 'country', 'target']:
                error("header does not match the format.", True)
            continue # skip header line
        if len(fields) != 11:
            error("each line should have 11 fields (separated by ',') but line %d got %d field(s):\n%s."% (i, len(fields), truncate(line)), True)
        last = fields[-1].upper()
        if last not in [">50K", "<=50K"]:
            error("the last field of each line should be either '>50K' or '<=50K' but line %d got '%s'." % (i, last), True)
        first = int(fields[0])
        if first != i:
            error("the first field id should match the line number but line %d got '%d'." % (i, first), True)
        pos += last == ">50K"

    if i != 6999:
        error("the number of lines should be 1000 but your file has %d line(s)." % i+1-6000)


    # formatting OK

    print("Your file has passed the formatting test! :)")
        
    pos_rate = pos / 10. # 100%
    print("Your positive rate is %.1f%%." % pos_rate)

    if pos_rate > 30:
        error("Your positive rate seems too high (should be similar to train and dev).", formatting_ok=True)

    if pos_rate < 10:
        error("Your positive rate seems too low (should be similar to train and dev).", formatting_ok=True)

    print("Your positive rate seems reasonable.\nThis does not guarantee that your prediction accuracy will be good though.")
