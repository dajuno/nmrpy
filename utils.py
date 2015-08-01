import sys


def progressbar(t, tend):
    # print(int(t/tend*10))
    amtDone = t/tend
    sys.stdout.write("\rsolve: [{0:50s}] {1}/{2}".format('#' * int(amtDone *
                     50), t, tend))
    sys.stdout.flush()
