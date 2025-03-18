import torch
from torch import nn
from .net_utils import MLP
from .vision_model import build_vis_encoder
from .language_model import build_text_encoder
from .grounding_model import build_encoder, build_decoder, build_TemporalSampling, build_SpatialActivation
from utils.misc import NestedTensor
from .vidswin.video_swin_transformer import vidswin_model
import json
import torch

def precision_recall(predicted_labels, true_labels):
    try:
        predicted_set = set(predicted_labels)
        true_set = set(true_labels)

        intersection_size = len(predicted_set.intersection(true_set))
    
        if len(predicted_set) == 0:
            precision = 0
        else:
            precision = intersection_size / len(predicted_set)
        if len(true_set) == 0:
            recall = 0
        else:
            recall = intersection_size / len(true_set)
    except Exception as e:
        print(e)
        precision, recall = 0, 0
    return precision, recall

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

class TASTVGNet(nn.Module):

    def __init__(self, cfg):
        super(TASTVGNet, self).__init__()
        self.cfg = cfg.clone()
        self.max_video_len = cfg.INPUT.MAX_VIDEO_LEN
        self.use_attn = cfg.SOLVER.USE_ATTN
        
        self.use_aux_loss = cfg.SOLVER.USE_AUX_LOSS  # use the output of each transformer layer
        self.use_actioness = cfg.MODEL.TASTVG.USE_ACTION
        self.query_dim = cfg.MODEL.TASTVG.QUERY_DIM

        self.vis_encoder = build_vis_encoder(cfg)
        vis_fea_dim = self.vis_encoder.num_channels

        hidden_dim = cfg.MODEL.TASTVG.HIDDEN

        self.text_encoder = build_text_encoder(cfg)
        self.s_temporal_clas = build_TemporalSampling(hidden_dim)
        self.t_temporal_clas = build_TemporalSampling(hidden_dim)
        self.s_spatial_clas = build_SpatialActivation(hidden_dim, cfg.DATASET.APP_NUM)
        self.t_spatial_clas = build_SpatialActivation(hidden_dim, cfg.DATASET.MOT_NUM)
        self.ground_encoder = build_encoder(cfg)
        self.ground_decoder = build_decoder(cfg)
        
        self.input_proj = nn.Conv2d(vis_fea_dim, hidden_dim, kernel_size=1)
        self.temp_embed = MLP(hidden_dim, hidden_dim, 2, 2, dropout=0.3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.vid = vidswin_model("video_swin_t_p4w7", "video_swin_t_p4w7_k400_1k")
        self.input_proj2 = nn.Conv2d(768, hidden_dim, kernel_size=1)
        for param in self.vid.parameters():
            param.requires_grad = False

        self.action_embed = MLP(hidden_dim, hidden_dim, 1, 2, dropout=0.3)

        self.ground_decoder.time_embed2 = self.action_embed

        # add the iteration anchor update
        self.ground_decoder.decoder.bbox_embed = self.bbox_embed

        self.verb_label = load_json(cfg.DATA_DIR + "/annos/train.json")
        self.verb_label2 = load_json(cfg.DATA_DIR + "/annos/test.json")
        self.theta = 0.45 if cfg.DATASET.NAME == 'VidSTG' else 0.7

    def forward(self, videos, texts, targets, iteration_rate=-1):
        """
        Arguments:
            videos  (NestedTensor): N * C * H * W, N = sum(T) 
            durations : batch video length
            texts   (NestedTensor]):
            targets (list[TargetTensor]): ground-truth
        Returns: 
        """
        # Extract Visual Feature
        vis_outputs, vis_pos_embed = self.vis_encoder(videos)
        vis_res_features, vis_mask, vis_durations = vis_outputs.decompose()  
        vis_features = self.input_proj(vis_res_features)  
        vis_outputs = NestedTensor(vis_features, vis_mask, vis_durations)
        with torch.no_grad():
            vid_features_all = self.vid(videos.tensors, len(videos.tensors))
        vid_features = self.input_proj2(vid_features_all['3'])

        # Extract Textual Feature
        info_key = str(targets[0]['item_id']) if self.cfg.DATASET.NAME == 'VidSTG' else targets[0]['vid']
        subject = self.verb_label[info_key]['sub'] if self.training else self.verb_label2[info_key]['sub']
        texts = [subject + " " + texts[0]]
        text_outputs, text_cls = self.text_encoder(texts, vis_features.device)

        # Multimodal Feature Fusion
        encoded_info = self.ground_encoder(videos=vis_outputs, vis_pos=vis_pos_embed, texts=text_outputs, vid=vid_features)

        l = vid_features.size(-1) * vid_features.size(-2)
        f_vid_features = encoded_info['encoded_feature'][-l:].permute(1, 2, 0).reshape(vid_features.size()).detach()
        f_vis_features = encoded_info['encoded_feature'][:l].permute(1, 2, 0).reshape(vid_features.size()).detach()
        f_text_cls = encoded_info['encoded_feature'][l:-l].mean(1).unsqueeze(0).detach()
        
        # Text-guided Temporal Sampling
        logits_f_m = self.t_temporal_clas(f_vid_features, f_text_cls)
        logits_f_a = self.s_temporal_clas(f_vis_features, f_text_cls)
        # Frame Sampling
        att_sequences = (logits_f_m.sigmoid()  + logits_f_a.sigmoid()) / 2
        choose_index = torch.nonzero(att_sequences > self.theta).squeeze().tolist()
        choose_index = [choose_index] if isinstance(choose_index, int) else choose_index
        choose_index = choose_index or torch.nonzero(att_sequences > 0).squeeze().tolist()   

        # Attribute-aware Spatial Activation
        logits_r_m, att_map_t = self.t_spatial_clas(f_vid_features[choose_index], f_text_cls[:,:1])
        logits_r_a, att_map_s = self.s_spatial_clas(f_vis_features[choose_index], f_text_cls[:,:1])

        # Generating Object Queries
        init_tempral_query = (encoded_info['encoded_feature'][-l:].permute(1, 0, 2)[choose_index] * att_map_t.unsqueeze(2)).mean((0, 1))
        init_spatial_query = (encoded_info['encoded_feature'][:l].permute(1, 0, 2)[choose_index] * att_map_s.unsqueeze(2)).mean((0, 1))

        # Query-based Decoding
        outputs_pos, outputs_time = self.ground_decoder(encoded_info=encoded_info, vis_pos=vis_pos_embed, isq=init_spatial_query, itq=init_tempral_query)

        if iteration_rate < 0:
            choose_index = torch.nonzero((self.action_embed(outputs_time)[-1].squeeze().sigmoid()>0.5).int()).squeeze().tolist()
            choose_index = [choose_index] if isinstance(choose_index, int) else choose_index
            choose_index = choose_index or torch.nonzero(att_sequences > 0).squeeze().tolist()  

            logits_r_a, att_map_s = self.s_spatial_clas(f_vis_features[choose_index], f_text_cls[:,:1])
            logits_r_m, att_map_t = self.t_spatial_clas(f_vid_features[choose_index], f_text_cls[:,:1])
           
            init_tempral_query = (encoded_info['encoded_feature'][-l:].permute(1, 0, 2)[choose_index] * att_map_t.unsqueeze(2)).mean((0, 1))
            init_spatial_query = (encoded_info['encoded_feature'][:l].permute(1, 0, 2)[choose_index] * att_map_s.unsqueeze(2)).mean((0, 1))
            outputs_pos, outputs_time = self.ground_decoder(encoded_info=encoded_info, vis_pos=vis_pos_embed, isq=init_spatial_query, itq=init_tempral_query)

        out = {}
        
        # Predict the bounding box
        outputs_coord = outputs_pos.flatten(1, 2) 
        out.update({"pred_boxes": outputs_coord[-1]})
        # Predict the temporal relevance scores
        out.update({"logits_f_m": logits_f_m})
        out.update({"logits_f_a": logits_f_a})
        # Predict the attribution classification scores
        out.update({"logits_r_a": logits_r_a})
        out.update({"logits_r_m": logits_r_m})
        # Predict the start and end probability 
        time_hiden_state = outputs_time  
        outputs_time = self.temp_embed(time_hiden_state)  
        outputs_actioness = self.action_embed(time_hiden_state) 
        out.update({"pred_sted": outputs_time[-1]})
        out.update({"pred_actioness": outputs_actioness[-1]})
        
        if self.use_aux_loss:
            out["aux_outputs"] = [
                {
                    "pred_sted": a,
                    "pred_boxes": b,
                    "pred_actioness": c
                }
                for a, b, c in zip(outputs_time[:-1], outputs_coord[:-1], outputs_actioness[:-1])
            ]

        out['verb_labels'] = self.verb_label[info_key]['verb_index_list'] if self.training else self.verb_label2[info_key]['verb_index_list']
        out['attr_labels'] = self.verb_label[info_key]['adj_index_list'] if self.training else self.verb_label2[info_key]['adj_index_list']
        out['att_sequences'] = att_sequences.unsqueeze(0)
        gt_index = torch.nonzero(targets[0]['actioness']).squeeze().tolist()
        out["pr"] = precision_recall(choose_index, gt_index)
        
        return out