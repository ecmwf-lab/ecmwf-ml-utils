#!/bin/env python3
import logging
import os
import re
import sys

LOGGER = logging.getLogger(__name__)


def parse_slurm_nodes(input=None) -> list[str]:
    if input is None:
        # No info about nodes provided. Try to read the env variable.
        input = os.environ.get("SLURM_JOB_NODELIST")

    if input is None:
        # Still no values, use 'localhost'
        input = "localhost"
        LOGGER.warn(
            f"Not in a slurm environement? Using mockup value SLURM_JOB_NODELIST={input}"
        )

    if "[" not in input:
        # Only one node, return it as a list
        return [input]

    m = re.match(r"(.+)\[(.+)\]", input)
    base = m.group(1)
    suffixes = m.group(2)
    assert "[" not in suffixes, f"Multiple [] not supported in {input}"

    output = []
    for suffix in suffixes.split(","):
        if "-" not in suffix:
            output.append(base + suffix)
            continue

        start, end = suffix.split("-")
        start, end = int(start), int(end)
        for i in range(start, end + 1):
            output.append(base + str(i))

    return output


def test():
    test_one("aa45-11", ["aa45-11"])
    test_one("aa45-[11,22]", ["aa45-11", "aa45-22"])
    test_one("aa45-[11-13]", ["aa45-11", "aa45-12", "aa45-13"])
    test_one("aa45-[11-13,22]", ["aa45-11", "aa45-12", "aa45-13", "aa45-22"])


def test_one(input, ref):
    print(f"{input} -> {ref}")
    output = parse_slurm_nodes(input)
    assert output == ref, (input, output, ref)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        test()
        exit()

    input = sys.argv[1]
    output = parse_slurm_nodes(input)
    print(output)
