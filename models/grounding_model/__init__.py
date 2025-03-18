from .modal_encoder import CrossModalEncoder
from .query_decoder import QueryDecoder
from .classifier import TemporalSampling, SpatialActivation

    
def build_encoder(cfg):
    return CrossModalEncoder(cfg)

def build_decoder(cfg):
    return QueryDecoder(cfg)

def build_TemporalSampling(width):
    return TemporalSampling(width)

def build_SpatialActivation(width, vobsize):
    return SpatialActivation(width, vobsize)