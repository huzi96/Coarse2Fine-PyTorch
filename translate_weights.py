# This file is being made available under the BSD License.  
# Copyright (c) 2021 Yueyu Hu
import torch
import numpy as np
import pickle

name_dict = {
  'a_model.transform.1.weight':'RE6_GPU0/analysis_transform_model/layre_1_mainA/kernel', 
  'a_model.transform.1.bias':'RE6_GPU0/analysis_transform_model/layre_1_mainA/bias',
  'a_model.transform.2.beta':'RE6_GPU0/analysis_transform_model/layre_1_mainA/gdn/reparam_beta',
  'a_model.transform.2.gamma':'RE6_GPU0/analysis_transform_model/layre_1_mainA/gdn/reparam_gamma',
  'a_model.transform.4.weight':'RE6_GPU0/analysis_transform_model/layre_2_mainA/kernel',
  'a_model.transform.4.bias':'RE6_GPU0/analysis_transform_model/layre_2_mainA/bias',
  'a_model.transform.5.beta':'RE6_GPU0/analysis_transform_model/layre_2_mainA/gdn_1/reparam_beta',
  'a_model.transform.5.gamma':'RE6_GPU0/analysis_transform_model/layre_2_mainA/gdn_1/reparam_gamma',
  'a_model.transform.7.weight':'RE6_GPU0/analysis_transform_model/layre_3_mainA/kernel',
  'a_model.transform.7.bias':'RE6_GPU0/analysis_transform_model/layre_3_mainA/bias',
  'a_model.transform.8.beta':'RE6_GPU0/analysis_transform_model/layre_3_mainA/gdn_2/reparam_beta',
  'a_model.transform.8.gamma':'RE6_GPU0/analysis_transform_model/layre_3_mainA/gdn_2/reparam_gamma',
  'a_model.transform.10.weight':'RE6_GPU0/analysis_transform_model/layre_4_mainA/kernel',
  'a_model.transform.10.bias':'RE6_GPU0/analysis_transform_model/layre_4_mainA/bias',

  's_model.transform.1.weight':'RE6_GPU0/synthesis_transform_model/layer_1_mainS/kernel',
  's_model.transform.1.bias':'RE6_GPU0/synthesis_transform_model/layer_1_mainS/bias',
  's_model.transform.2.beta':'RE6_GPU0/synthesis_transform_model/layer_1_mainS/gdn_3/reparam_beta',
  's_model.transform.2.gamma':'RE6_GPU0/synthesis_transform_model/layer_1_mainS/gdn_3/reparam_gamma',
  's_model.transform.4.weight':'RE6_GPU0/synthesis_transform_model/layer_2_mainS/kernel',
  's_model.transform.4.bias':'RE6_GPU0/synthesis_transform_model/layer_2_mainS/bias',
  's_model.transform.5.beta':'RE6_GPU0/synthesis_transform_model/layer_2_mainS/gdn_4/reparam_beta',
  's_model.transform.5.gamma':'RE6_GPU0/synthesis_transform_model/layer_2_mainS/gdn_4/reparam_gamma',
  's_model.transform.7.weight':'RE6_GPU0/synthesis_transform_model/layer_3_mainS/kernel',
  's_model.transform.7.bias':'RE6_GPU0/synthesis_transform_model/layer_3_mainS/bias',
  's_model.transform.8.beta':'RE6_GPU0/synthesis_transform_model/layer_3_mainS/gdn_5/reparam_beta',
  's_model.transform.8.gamma':'RE6_GPU0/synthesis_transform_model/layer_3_mainS/gdn_5/reparam_gamma',
  
  'ha_model_1.transform.0.weight':'RE6_GPU0/h_analysis_transform_model_load/layer_1_h1a/kernel',
  'ha_model_1.transform.0.bias':'RE6_GPU0/h_analysis_transform_model_load/layer_1_h1a/bias',
  'ha_model_1.transform.2.weight':'RE6_GPU0/h_analysis_transform_model_load/layer_2_h1a/kernel',
  'ha_model_1.transform.2.bias':'RE6_GPU0/h_analysis_transform_model_load/layer_2_h1a/bias',
  'ha_model_1.transform.4.weight':'RE6_GPU0/h_analysis_transform_model_load/layer_3_h1a/kernel',
  'ha_model_1.transform.4.bias':'RE6_GPU0/h_analysis_transform_model_load/layer_3_h1a/bias',
  'ha_model_1.transform.6.weight':'RE6_GPU0/h_analysis_transform_model_load/layer_4_h1a/kernel',
  'ha_model_1.transform.6.bias':'RE6_GPU0/h_analysis_transform_model_load/layer_4_h1a/bias',

  'ha_model_2.transform.0.weight':'RE6_GPU0/h_analysis_transform_model_load_1/layer_1_h2a/kernel',
  'ha_model_2.transform.0.bias':'RE6_GPU0/h_analysis_transform_model_load_1/layer_1_h2a/bias',
  'ha_model_2.transform.2.weight':'RE6_GPU0/h_analysis_transform_model_load_1/layer_2_h2a/kernel',
  'ha_model_2.transform.2.bias':'RE6_GPU0/h_analysis_transform_model_load_1/layer_2_h2a/bias',
  'ha_model_2.transform.4.weight':'RE6_GPU0/h_analysis_transform_model_load_1/layer_3_h2a/kernel',
  'ha_model_2.transform.4.bias':'RE6_GPU0/h_analysis_transform_model_load_1/layer_3_h2a/bias',
  'ha_model_2.transform.6.weight':'RE6_GPU0/h_analysis_transform_model_load_1/layer_4_h2a/kernel',
  'ha_model_2.transform.6.bias':'RE6_GPU0/h_analysis_transform_model_load_1/layer_4_h2a/bias',

  'hs_model_1.transform.0.weight':'RE6_GPU0/h_synthesis_transform_model_load/layer_1_h1s/kernel',
  'hs_model_1.transform.0.bias':'RE6_GPU0/h_synthesis_transform_model_load/layer_1_h1s/bias',
  'hs_model_1.transform.1.weight':'RE6_GPU0/h_synthesis_transform_model_load/layer_2_h1s/kernel',
  'hs_model_1.transform.1.bias':'RE6_GPU0/h_synthesis_transform_model_load/layer_2_h1s/bias',
  'hs_model_1.transform.3.weight':'RE6_GPU0/h_synthesis_transform_model_load/layer_3_h1s/kernel',
  'hs_model_1.transform.3.bias':'RE6_GPU0/h_synthesis_transform_model_load/layer_3_h1s/bias',
  'hs_model_1.transform.7.weight':'RE6_GPU0/h_synthesis_transform_model_load/layer_4_h1s/kernel',
  'hs_model_1.transform.7.bias':'RE6_GPU0/h_synthesis_transform_model_load/layer_4_h1s/bias',

  'hs_model_2.transform.0.weight':'RE6_GPU0/h_synthesis_transform_model_load_1/layer_1_h2s/kernel',
  'hs_model_2.transform.0.bias':'RE6_GPU0/h_synthesis_transform_model_load_1/layer_1_h2s/bias',
  'hs_model_2.transform.1.weight':'RE6_GPU0/h_synthesis_transform_model_load_1/layer_2_h2s/kernel',
  'hs_model_2.transform.1.bias':'RE6_GPU0/h_synthesis_transform_model_load_1/layer_2_h2s/bias',
  'hs_model_2.transform.3.weight':'RE6_GPU0/h_synthesis_transform_model_load_1/layer_3_h2s/kernel',
  'hs_model_2.transform.3.bias':'RE6_GPU0/h_synthesis_transform_model_load_1/layer_3_h2s/bias',
  'hs_model_2.transform.7.weight':'RE6_GPU0/h_synthesis_transform_model_load_1/layer_4_h2s/kernel',
  'hs_model_2.transform.7.bias':'RE6_GPU0/h_synthesis_transform_model_load_1/layer_4_h2s/bias',

  'prediction_model_2.transform.1.weight':'RE6_GPU0/prediction_model_load/P_conv1_pred_2/kernel',
  'prediction_model_2.transform.1.bias':'RE6_GPU0/prediction_model_load/P_conv1_pred_2/bias',
  'prediction_model_2.transform.4.weight':'RE6_GPU0/prediction_model_load/P_conv2_pred_2/kernel',
  'prediction_model_2.transform.4.bias':'RE6_GPU0/prediction_model_load/P_conv2_pred_2/bias',
  'prediction_model_2.transform.7.weight':'RE6_GPU0/prediction_model_load/P_conv3_pred_2/kernel',
  'prediction_model_2.transform.7.bias':'RE6_GPU0/prediction_model_load/P_conv3_pred_2/bias',
  'prediction_model_2.fc.weight':'RE6_GPU0/prediction_model_load/P_fc_pred_2/kernel',
  'prediction_model_2.fc.bias':'RE6_GPU0/prediction_model_load/P_fc_pred_2/bias',
  
  'prediction_model_3.transform.1.weight':'RE6_GPU0/prediction_model_load_1/P_conv1_pred_3/kernel',
  'prediction_model_3.transform.1.bias':'RE6_GPU0/prediction_model_load_1/P_conv1_pred_3/bias',
  'prediction_model_3.transform.4.weight':'RE6_GPU0/prediction_model_load_1/P_conv2_pred_3/kernel',
  'prediction_model_3.transform.4.bias':'RE6_GPU0/prediction_model_load_1/P_conv2_pred_3/bias',
  'prediction_model_3.transform.7.weight':'RE6_GPU0/prediction_model_load_1/P_conv3_pred_3/kernel',
  'prediction_model_3.transform.7.bias':'RE6_GPU0/prediction_model_load_1/P_conv3_pred_3/bias',
  'prediction_model_3.fc.weight':'RE6_GPU0/prediction_model_load_1/P_fc_pred_3/kernel',
  'prediction_model_3.fc.bias':'RE6_GPU0/prediction_model_load_1/P_fc_pred_3/bias',

  'side_recon_model.layer_1.1.weight':'RE6_GPU0/side_info_recon_model_load/layer_1_mainS_recon/kernel',
  'side_recon_model.layer_1.1.bias':'RE6_GPU0/side_info_recon_model_load/layer_1_mainS_recon/bias',
  'side_recon_model.layer_1a.1.weight':'RE6_GPU0/side_info_recon_model_load/layer_1a_mainS_recon/kernel',
  'side_recon_model.layer_1a.1.bias':'RE6_GPU0/side_info_recon_model_load/layer_1a_mainS_recon/bias',
  'side_recon_model.layer_1b.1.weight':'RE6_GPU0/side_info_recon_model_load/layer_1b_mainS_recon/kernel',
  'side_recon_model.layer_1b.1.bias':'RE6_GPU0/side_info_recon_model_load/layer_1b_mainS_recon/bias',
  'side_recon_model.layer_3_1.0.weight':'RE6_GPU0/side_info_recon_model_load/layer_3_1_mainS_recon/kernel',
  'side_recon_model.layer_3_1.0.bias':'RE6_GPU0/side_info_recon_model_load/layer_3_1_mainS_recon/bias',
  'side_recon_model.layer_3_2.0.weight':'RE6_GPU0/side_info_recon_model_load/layer_3_2_mainS_recon/kernel',
  'side_recon_model.layer_3_2.0.bias':'RE6_GPU0/side_info_recon_model_load/layer_3_2_mainS_recon/bias',
  'side_recon_model.layer_3_3.0.weight':'RE6_GPU0/side_info_recon_model_load/layer_3_3_mainS_recon/kernel',
  'side_recon_model.layer_3_3.0.bias':'RE6_GPU0/side_info_recon_model_load/layer_3_3_mainS_recon/bias',
  'side_recon_model.layer_4.1.weight':'RE6_GPU0/side_info_recon_model_load/layer_4_mainS_recon/kernel',
  'side_recon_model.layer_4.1.bias':'RE6_GPU0/side_info_recon_model_load/layer_4_mainS_recon/bias',
  'side_recon_model.layer_5.0.weight':'RE6_GPU0/side_info_recon_model_load/layer_5_mainS_recon/kernel',
  'side_recon_model.layer_5.0.bias':'RE6_GPU0/side_info_recon_model_load/layer_5_mainS_recon/bias',
  'side_recon_model.layer_6.weight':'RE6_GPU0/side_info_recon_model_load/layer_6_mainS_recon/kernel',
  'side_recon_model.layer_6.bias':'RE6_GPU0/side_info_recon_model_load/layer_6_mainS_recon/bias'
}

def convert_weight_dict(torch_d, tf_d):
  target_d = {}
  for k in torch_d:
    if not k in name_dict:
      # print('[Warning] %s not in pre-defined name dict' % (k))
      target_d[k] = torch_d[k]
      continue
    tf_k = name_dict[k]
    tf_w = np.array(tf_d[tf_k], dtype=np.float64)
    if 'kernel' in tf_k and ( 'conv' in tf_k or 'mainS_recon' in tf_k or 'transform_model' in tf_k ):
      torch_w = tf_w.transpose((3,2,0,1))
    elif ('dense' in tf_k or 'fc' in tf_k) and 'kernel' in tf_k:
      torch_w = tf_w.transpose((1,0))
    elif 'reparam_gamma' in tf_k:
      torch_w = tf_w.transpose((1,0))
    else:
      torch_w = tf_w
    target_d[k] = torch.tensor(torch_w).float()
  return target_d

def load_tf_weights(torch_d, pkfn='./fin025.pk'):
  with open(pkfn,'rb') as f:
    tf_d = pickle.load(f)
  target_d = convert_weight_dict(torch_d, tf_d)
  target_d['get_h1_sigma'] = torch.tensor(np.array(tf_d['RE6_GPU0/get_h1_sigma/get_h1_sigma/z_sigma']).reshape((1,32*4,1,1)))
  return target_d
