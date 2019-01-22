from .model import regist_model, get_model
from .attention_cluster import AttentionCluster
from .nextvlad import NEXTVLAD

# regist models
regist_model("AttentionCluster", AttentionCluster)
regist_model("NEXTVLAD", NEXTVLAD)

