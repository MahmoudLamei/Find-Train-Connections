import networkx as nx
import matplotlib.pyplot as plt
import numpy as py
import pandas as pd
import heapq
from IPython.display import display
from datetime import datetime, timedelta

schedule = pd.read_csv('mini-schedule.csv')
df = schedule.head(6)
print(df)