import __init__ as cls
import run_embedding as emb
from arg_parser import *

if __name__ == '__main__':
    # args.split = 0.8
    # args.run_embedding = True
    # emb.run_main()
    #TODO here>> figure out how to run gcn mlp and softmax
    args.ensemble = True
    args.report_performance = True
    args.split = 0.6
    cls.run_main()