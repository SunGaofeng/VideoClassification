from .model import regist_model, get_model
from .attention_cluster import AttentionCluster
from .nextvlad import NEXTVLAD
from .tsn import TSN
from .stnet import STNET

# regist models
regist_model("AttentionCluster", AttentionCluster)
regist_model("NEXTVLAD", NEXTVLAD)
regist_model("TSN", TSN)
regist_model("STNET", STNET)

