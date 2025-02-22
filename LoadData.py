import csv
import numpy as np
from typing import List

# given a csv file's path
# return (data, headers)
# where data is a List of List[dict]
# --each element of the is a person's data stored in float
# --which is a List[np.array], first index is step, and second index is "key" which is in headers
# where headers is the titles in this csv
def load(file_path: str) -> tuple[List[List[np.ndarray]], List]:
    data = []
    headers = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        
        # 读取剩余的数据行
        for row in csvreader:
            if len(headers) == 0:
                for k in row:
                    if k not in ["traj", "step"]:
                        headers.append(k)
            # 将每行数据存储到列表中
            tmp =dict(row)
            patient = int(tmp['traj'])
            step = int(tmp['step'])
            
            # ranging [1,???) in the file, make it start from zero
            patient-=1
            assert patient >= 0

            if patient == len(data):
                data.append([])
            assert patient == len(data)-1

            assert step == len(data[patient])

            inner = np.zeros(len(headers), dtype=np.float32)

            for i in range(len(headers)):
                inner[i] = float(tmp[headers[i]])
            
            data[patient].append(inner)
    
    return data, headers

# data: [patient][step][headers]
def getMaxSequenceLength(data) -> int:
    ans = 0
    for patient in data:
        ans = max(ans, len(patient))
    return ans

def getTreatmentId(data:np.ndarray, headers: List[str]) -> int:
    tmp = getDataViaKeys(data, getTreatmentKeys(), headers)
    return int(tmp[0])

def getMaxTreatmentNumber(data, headers: List[str]) -> int:
    ans = 1
    treat_key = getTreatmentKeys()
    assert len(treat_key) == 1

    for patient in data:
        for step in patient:
            ans = max(ans, getTreatmentId(step, headers)+1)
    return ans

# see the paper's picture for details
def getXKeys() -> List[str]:
    return ['o:Weight_kg','o:GCS','o:HR','o:SysBP','o:MeanBP',
            'o:DiaBP','o:RR','o:Temp_C','o:FiO2_1','o:Potassium',
            'o:Sodium','o:Chloride','o:Glucose','o:Magnesium',
            'o:Calcium','o:Hb','o:WBC_count','o:Platelets_count',
            'o:PTT','o:PT','o:Arterial_pH','o:paO2','o:paCO2',
            'o:Arterial_BE','o:HCO3','o:Arterial_lactate']

# see the paper's picture for details
def getVKeys() -> List[str]:
    return ['o:gender', 'o:mechvent', 'o:max_dose_vaso', 'o:re_admission', 'o:age']


# see the paper's picture for details
def getOutputKeys() -> List[str]:
    return ['o:SOFA']


# see the paper's picture for details
def getTreatmentKeys() -> List[str]:
    return ['a:action']

# data one person one step
# return float type np array
def getDataViaKeys(data:np.ndarray, keys: List[str], headers: List[str])->np.ndarray:
    ans = np.zeros(len(keys), dtype=np.float32)
    for i in range(len(keys)):
        key = keys[i]
        idx = headers.index(key)
        ans[i] = data[idx]
    return ans

# data: [patient][step][headers]
# headers: list of keys
def getEncoderData(data, headers, max_sequence_length :int, num_treatments : int) -> dict:
    num_patients:int = len(data)
    cov_key = getVKeys() + getXKeys()
    output_key = getOutputKeys()

    num_cov = len(cov_key)
    num_output = len(output_key)


    covariates = np.zeros((num_patients, max_sequence_length, num_cov), dtype=np.float32)

    inputV = np.zeros((num_patients, max_sequence_length, len(getVKeys())), dtype=np.float32)

    output = np.zeros((num_patients, max_sequence_length, num_output), dtype=np.float32)
    treatments = np.zeros((num_patients, max_sequence_length, num_treatments), dtype=np.float32)
    active_entries = np.zeros((num_patients, max_sequence_length, 1), dtype=np.float32)
    sequence_lengths = np.zeros(num_patients, dtype=np.int32)

    for i in range(len(data)):
        sequence_lengths[i] = len(data[i])

        active_entries[i,:sequence_lengths[i],0] = np.ones_like(active_entries[i,:sequence_lengths[i],0])

        for j in range(len(data[i])):
            
            treatments[i,j,getTreatmentId(data[i][j], headers)] = 1.0

            covariates[i,j,:] = getDataViaKeys(data[i][j], cov_key, headers)
            inputV[i,j,:] = getDataViaKeys(data[i][j], getVKeys(), headers)

            output[i,j,:] = getDataViaKeys(data[i][j], output_key, headers)

    data_dict = dict()

    data_dict['inputV'] = inputV
    previous_treatments = np.concatenate([np.zeros_like(treatments[:,0:1,:]), treatments[:,:-1,:]], axis=1)
    data_dict['current_covariates'] = np.concatenate([covariates, previous_treatments], axis=-1)
    data_dict['current_treatments'] = treatments
    data_dict['outputs'] = output
    data_dict['active_entries'] = active_entries
    data_dict['sequence_lengths'] = sequence_lengths

    return data_dict

# data_map: get from getEncoderData
# states: balancing representation
def getDecoderData(data_map, states, projection_horizon):
    outputs = data_map["outputs"]
    sequence_lengths = data_map["sequence_lengths"]
    active_entries = data_map["active_entries"]
    current_treatments = data_map["current_treatments"]
    current_V = data_map["inputV"]

    num_patients, num_time_steps, num_outputs = outputs.shape

    num_seq2seq_rows = num_patients * num_time_steps

    seq2seq_state_inits = np.zeros((num_seq2seq_rows, states.shape[-1]))


    seq2seq_current_treatments = np.zeros(
        (num_seq2seq_rows, projection_horizon, current_treatments.shape[-1]),
        dtype = np.float32
    )

    seq2seq_previous_treatments = np.zeros(
        (num_seq2seq_rows, projection_horizon, current_treatments.shape[-1]),
        dtype = np.float32
    )

    seq2seq_current_covariates = np.zeros(
        (num_seq2seq_rows, projection_horizon, current_V.shape[-1]+outputs.shape[-1]),
        dtype = np.float32
    )

    seq2seq_outputs = np.zeros(
        (num_seq2seq_rows, projection_horizon, outputs.shape[-1]),
        dtype = np.float32
    )

    seq2seq_active_entries = np.zeros(
        (num_seq2seq_rows, projection_horizon, active_entries.shape[-1]),
        dtype = np.float32
    )

    seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows, dtype=np.int32)

    total_seq2seq_rows = 0  # we use this to shorten any trajectories later

    for i in range(num_patients):

        sequence_length = int(sequence_lengths[i])

        for t in range(1, sequence_length):  # shift outputs back by 1
            seq2seq_state_inits[total_seq2seq_rows, :] = states[
                i, t - 1, :
            ]  # previous state output

            max_projection = min(projection_horizon, sequence_length - t)

            seq2seq_active_entries[total_seq2seq_rows, :max_projection, :] = (
                active_entries[i, t : t + max_projection, :]
            )
            seq2seq_current_treatments[total_seq2seq_rows, :max_projection, :] = (
                current_treatments[i, t : t + max_projection, :]
            )
            seq2seq_previous_treatments[total_seq2seq_rows, :max_projection, :] = (
                current_treatments[i, t-1 : t + max_projection-1, :]
            )
            seq2seq_outputs[total_seq2seq_rows, :max_projection, :] = outputs[
                i, t : t + max_projection, :
            ]
            seq2seq_sequence_lengths[total_seq2seq_rows] = max_projection
            seq2seq_current_covariates[total_seq2seq_rows, :max_projection, :current_V.shape[-1]] = (
                current_V[i, t : t + max_projection, :]
            )
            seq2seq_current_covariates[total_seq2seq_rows, :max_projection, current_V.shape[-1]: ] = (
                outputs[i, t-1 : t + max_projection-1, :]
            )

            total_seq2seq_rows += 1   
    # Filter everything shorter
    seq2seq_state_inits = seq2seq_state_inits[:total_seq2seq_rows, :]
    seq2seq_current_treatments = seq2seq_current_treatments[:total_seq2seq_rows, :, :]
    seq2seq_previous_treatments = seq2seq_previous_treatments[:total_seq2seq_rows,:,:]
    seq2seq_current_covariates = seq2seq_current_covariates[:total_seq2seq_rows, :, :]
    seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
    seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
    seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]

    # Package outputs
    seq2seq_data_map = {
        "init_states": seq2seq_state_inits,
        "current_treatments": seq2seq_current_treatments,
        "current_covariates": np.concatenate([seq2seq_current_covariates,seq2seq_previous_treatments],axis=-1),
        "outputs": seq2seq_outputs,
        "sequence_lengths": seq2seq_sequence_lengths,
        "active_entries": seq2seq_active_entries,
    }

    return seq2seq_data_map
