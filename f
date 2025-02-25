#!/usr/bin/env python
from fed.cli import parse, main


if __name__ == '__main__':
    args = parse()
    main(args)