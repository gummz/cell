def function1():
    pass


def function2():
    pass


if __name__ == '__main__':
    to_output = [function1, function2]
    for output in to_output:
        output()
