from argparse import ArgumentParser
from os.path import abspath

class SubArgParser(ArgumentParser):
    """Subclass of ArgumentParser with default values"""
    def __init__(self, *args, **kwargs):
        super(SubArgParser, self).__init__(*args, **kwargs)
        self._add_args()

    def _add_args(self):
        """Populate ArgParser object with default values"""

        #   todo: work with all the arguments of the pipeline

        self.description = "Genome in a Box (GBOX) Analysis Toolkit"
        self.add_argument("--version", action = "version", version = "%(prog)s 0.1.0")

        self.add_argument("-d", "--datadir", action = "store",
                        default = "./", help = "Path to data directory [default='./']",
                        type = abspath)
