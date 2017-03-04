# from tools import build_graph
# import pandas as pd
# train_df = pd.read_csv("../../data/training_set.txt", sep=" ",header=None)
# train_df.columns = ["source","target","label"]
#
# g = build_graph("../../")
# node_degree = pd.DataFrame(index= g.vs["name"],data = {"indegree":g.indegree(),"outdegree":g.outdegree()})
# node_degree.to_csv("node_degree.csv",header=0)
