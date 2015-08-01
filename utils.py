import sys


def progressbar(t, tend, s='progress'):
    # print(int(t/tend*10))
    amtDone = t/tend
    sys.stdout.write("\r{0}: [{1:50s}] {2:.3f}/{3:.1f}".format(s, '#' * int(amtDone *
                     50), t, tend))
    sys.stdout.flush()
