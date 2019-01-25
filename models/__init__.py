from .model import regist_model, get_model
from .attention_cluster import AttentionCluster
from .nextvlad import NEXTVLAD
from .attention_lstm import AttentionLSTM

# regist models
regist_model("AttentionCluster", AttentionCluster)
regist_model("NEXTVLAD", NEXTVLAD)
regist_model("AttentionLSTM", AttentionLSTM)

