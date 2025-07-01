import torch
import numpy as np
import LoadData
import pandas as pd
from CRN_model import CRNDataset
from CRN_Lightning_model import LitCRN
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# 配置参数 ----------------------------------------------------
ENCODER_CKPT = "./models/encoder_30.ckpt"
DECODER_CKPT = "./models/decoder_30_1_30.ckpt"
DATA_PATH = "./data/sepsis_final_data_withTimes.csv"
MIN_SEQ_LENGTH = 5
OUTPUT_CSV = "./treatment_effects/all_patient_effects.csv"  # 统一输出文件
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    """加载预训练模型"""
    encoder = LitCRN.load_from_checkpoint(ENCODER_CKPT, map_location=DEVICE).crn_model
    decoder = LitCRN.load_from_checkpoint(DECODER_CKPT, map_location=DEVICE).crn_model
    encoder.eval()
    decoder.eval()
    return encoder, decoder

def prepare_encoder_input(patient_data, headers, num_treatments):
    """重构后的encoder输入准备函数"""
    # 获取所有需要的特征索引
    cov_keys = LoadData.getXKeys() + LoadData.getVKeys()
    treat_key = LoadData.getTreatmentKeys()[0]
    
    # 验证特征完整性
    missing = [k for k in cov_keys if k not in headers]
    if missing:
        raise ValueError(f"Missing covariates in headers: {missing}")
        
    cov_indices = [headers.index(k) for k in cov_keys]
    treat_idx = headers.index(treat_key)
    
    seq_data = []
    prev_treatment = np.zeros(num_treatments, dtype=np.float32)  # 初始历史治疗
    
    for step in range(5):
        if step >= len(patient_data):
            break
            
        # 当前协变量特征
        current_cov = patient_data[step][cov_indices]
        
        # 当前治疗（one-hot编码）
        current_treat = np.zeros(num_treatments, dtype=np.float32)
        treatment_id = int(patient_data[step][treat_idx])
        current_treat[treatment_id] = 1.0
        
        # 拼接协变量和前一时刻的治疗
        combined_features = np.concatenate([
            current_cov,       # 协变量特征
            prev_treatment     # 历史治疗（t-1时刻）
        ])
        
        seq_data.append(combined_features)
        prev_treatment = current_treat  # 更新历史治疗
    
    # Padding处理（如果不足5步）
    while len(seq_data) < 5:
        padding = np.zeros(len(cov_indices) + num_treatments, dtype=np.float32)
        seq_data.append(padding)
    
    return torch.tensor([seq_data], dtype=torch.float32).to(DEVICE)


def compute_all_effects(encoder, decoder, eligible_patients, csv_data, headers):
    """计算所有病人的因果效应并保存为CSV"""
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    num_treatments = encoder.num_treatments
    
    # 准备结果DataFrame
    columns = ["patient_id"] + [f"effect_treatment_{i}" for i in range(1, num_treatments)]
    results = []
    
    for pid in tqdm(eligible_patients, desc="Processing patients"):
        try:
            # 生成平衡表示
            encoder_input = prepare_encoder_input(csv_data[pid], headers, num_treatments=encoder.num_treatments)
            with torch.no_grad():
                br, _, _, _ = encoder(encoder_input, 
                                    torch.zeros(1, 5, num_treatments).to(DEVICE), 
                                    None, 0)
            
            # 使用最后一步的br
            last_br = br[:, -1:, :]
            
            # 获取基线预测 (treatment 0)
            baseline_treatment = torch.zeros(1, 1, num_treatments).to(DEVICE)
            baseline_treatment[..., 0] = 1.0
            with torch.no_grad():
                # baseline = decoder(current_covariates=last_br, current_treatments=baseline_treatment, init_states=last_br, alpha=0)[2].cpu().numpy()
                baseline = decoder.outcome_layer(torch.cat((last_br, baseline_treatment), dim=-1))

            # 计算所有treatment的效果
            row = [pid]
            for treat_idx in range(1, num_treatments):
                treatment = torch.zeros(1, 1, num_treatments).to(DEVICE)
                treatment[..., treat_idx] = 1.0
                with torch.no_grad():
                    # pred = decoder(current_covariates=last_br, current_treatments=treatment, init_states=last_br, alpha=0)[2].cpu().numpy()
                    pred = decoder.outcome_layer(torch.cat((last_br, treatment), dim=-1))
                row.append(float(pred - baseline))
            
            results.append(row)
            
        except Exception as e:
            print(f"Error processing patient {pid}: {str(e)}")
            continue
    
    # 转换为DataFrame并保存
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n结果已保存至 {OUTPUT_CSV}")
    print(f"总处理病人数: {len(results)}/{len(eligible_patients)}")
    return df

if __name__ == "__main__":
    # 加载数据
    print("Loading data...")
    csv_data, headers = LoadData.load(DATA_PATH)
    
    # 筛选病人
    eligible_patients = [i for i, traj in enumerate(csv_data) if len(traj) >= MIN_SEQ_LENGTH]
    print(f"找到 {len(eligible_patients)} 个序列长度 >= {MIN_SEQ_LENGTH} 的病人")
    
    # 加载模型
    print("Loading models...")
    encoder, decoder = load_models()

    # 在load_models函数后添加维度检查
    # print(f"Encoder预期输入维度: {encoder.num_covariates}")
    # print(f"实际准备的特征维度: {len(cov_indices) + encoder.num_treatments}")
    
    # 计算并保存结果
    print("Computing treatment effects...")
    _ = compute_all_effects(encoder, decoder, eligible_patients, csv_data, headers)