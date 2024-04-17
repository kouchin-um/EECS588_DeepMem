#!/usr/bin/python3

import sys
import re
import argparse
import random
import os

change_pointer = False

def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
            description="Input arguments")

    parser.add_argument("graph",
            type=str,
            help="The graph file name")

    parser.add_argument("output",
            type=str,
            help="The output file name")

    parser.add_argument("-o", "--object", 
            type=str, required=True, choices=["_EPROCESS", "_ETHREAD"],
            help="The type of the object to be mutated [_EPROCESS/_ETHREAD]")
    
    parser.add_argument("-s", "--size", 
            type=int, required=True,
            help="The number of bytes to be mutated in total")
    
    return parser

def mutate_addr(addr: str) -> str:
    addr_list = addr.split(',')
    p = random.randrange(len(addr_list))
    
    pos = random.randrange(2, len(addr_list[p]))
    if pos%2:
        pos = pos - 1
    addr_list[p] = addr_list[p][:pos] + rf"{random.randrange(256):02x}" + addr_list[p][pos+2:]
    return ",".join(addr_list)

def mutate_data(data: str) -> str:
    data_bytes = data.strip().split(' ')
    pos = random.randrange(len(data_bytes))
    data_bytes[pos] = str(random.randrange(256))
    return " ".join(data_bytes)

def mutate_line(line: str, size: int) -> str:
    data = line.strip().split('\t')
    # data[1:4] and data[6] can be mutated
    index = 0
    data_start = [0]

    # data[1:4]
    for i in range(1,5):
        if change_pointer:
            if data[i] != '':
                index = index + 4 * (data[i].count(',') + 1)
            data_start.append(index)
        else:
            data_start.append(0)

    # data[6] (but the size is given by data[5])
    index = index + int(data[5])
    data_start.append(index)

    # data_start size is 6
    for time in range(size):
        pos = random.randrange(data_start[0], data_start[5])
        for i in range(len(data_start) - 1):
            if data_start[i] <= pos < data_start[i+1]:
                if i == 4:
                    data[6] = mutate_data(data[6])
                else:
                    data[i+1] = mutate_addr(data[i+1])
                break
    
    return "\t".join(data)

if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()

    graph_file = open(args.graph, "r")
    lines = graph_file.readlines()

    cand_lines = []

    for i in range(len(lines)):
        lines[i] = lines[i].strip()
        if args.object in lines[i]:
            cand_lines.append(i)

    targets = [ cand_lines[i] for i in [random.randrange(len(cand_lines)) for j in range(args.size)] ]
    targets.sort()

    current_size = 0
    for i in range(len(targets)):
        j = targets[i]
        current_size = current_size + 1
        if i == len(targets)-1 or targets[i+1] != targets[i]:
            lines[j] = mutate_line(lines[j], current_size)
            current_size = 0

    output_file = open(args.output, "w")
    for line in lines:
        print(line, file=output_file)
    output_file.close()

    graph_file.close()