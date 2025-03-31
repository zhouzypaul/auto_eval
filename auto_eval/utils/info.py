def print_red(message):
    print(f"\033[91m{message}\033[0m")  # red color


def print_yellow(message):
    print(f"\033[33m{message}\033[0m")  # darker yellow color


def print_obvious(message):
    print("*" * 80)
    print(message)
    print("*" * 80)
