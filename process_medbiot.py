import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
import hashlib
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.pipeline import Pipeline
import ipaddress
import os

# Khởi tạo scaler
scaler = StandardScaler()
normalizer = Normalizer()

# Cấu hình
chunk_size = 1000000
input_files = ["/mnt/c/Users/hoang/FileCSV_DACN_2025/medbiot.csv","C:\\Users\\hoang\\FileCSV_DACN_2025\\medbiot.csv"]
output_files = ["/mnt/c/Users/hoang/FileCSV_DACN_2025/parquet_medbiot", "C:\\Users\\hoang\\FileCSV_DACN_2025\\parquet_medbiot"]

input_file = ""
output_file =""
############## CHeck os #############
os_type = os.name
print(os_type)
if os_type == "nt":
    input_file = input_files[1]
    output_file = output_files[1]
else:
    input_file = input_files[0]
    output_file = output_files[0]

cols_to_drop = []
# ip_to_float_Cols = ['id.orig_h', 'id.resp_h']
# str_to_float_Cols = ['conn_state', 'history', 'id.orig_p', 'id.resp_p']
numeric_cols = ['MI_dir_5_mean', 'MI_dir_5_std', 'MI_dir_3_weight', 'MI_dir_3_mean', 'MI_dir_3']
# onehot_cols = ['proto', 'service']
label_mapping={}
labels = ['benign', "torii", "mirai_spread", 'bashlite_spread', "mirai_c2", 'bashlite_c2']

for index, label in enumerate(labels):
    label_mapping[label]=index

print(label_mapping)

##### FUNC ############
def ip_to_float(ip):
    try:
        return float(int(ipaddress.IPv4Address(ip)))/1e9
    except:
        return 0.0  # for invalid or empty IPs
    
def sum_of_squares(partition):
    return pd.Series([(partition ** 2).sum()])

def string_to_float(s):
    if pd.notna(s):
        return int(hashlib.sha256(str(s).encode('utf-8')).hexdigest(), 16) % 10**8 / 1e8
    return 0
##### FUNC ##############

proto_categories =set()
service_categories = set()

for index, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, dtype='str')):
    chunk.replace({"(empty)": '0', "-": '0', "":'0', r'[N|n][a|A][N|n]':'0', 'unknown':'0'}, inplace=True)
    chunk.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    chunk.drop_duplicates(inplace=True)
    
    for col in numeric_cols:
        chunk[col]=chunk[col].astype("float32")
        
    scaler.partial_fit(chunk[numeric_cols])  # Fit theo feature
    
    # proto_categories.update(chunk['proto'].unique())
    # service_categories.update(chunk['service'].unique())
    
    print(index)

proto_categories = sorted(proto_categories)
service_categories = sorted(service_categories)

first_chunk = True
for index, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, dtype='str')):
    chunk.replace({"(empty)": '0', "-": '0', "":'0', r'[N|n][a|A][N|n]':'0', 'unknown':'0'}, inplace=True)
    chunk.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    chunk.drop_duplicates(inplace=True)
    
    chunk['label'] = chunk['label'].map(label_mapping).fillna(-1).astype("int32")

    for col in ip_to_float_Cols:
        chunk[col] = chunk[col].apply(ip_to_float).astype("float32")
    for col in str_to_float_Cols:
        chunk[col] = chunk[col].apply(string_to_float).astype("float32")
    for col in numeric_cols:
        chunk[col]=chunk[col].astype("float32")
    
    chunk['proto'] = pd.Categorical(chunk['proto'], categories=proto_categories)
    chunk['service'] = pd.Categorical(chunk['service'], categories=service_categories)
    
    chunk = pd.get_dummies(chunk, columns = onehot_cols, prefix = onehot_cols, dtype='int32')

    chunk.drop(columns=[c for c in chunk.columns if ('service_-' in c or 'service_0' in c or 'proto_-' in c or 'proto_0' in c)], errors='ignore', inplace=True)
    ################ Scaler ###########################
    # print(chunk.head())
    scaled = scaler.transform(chunk[numeric_cols])
    # normalized = normalizer.transform(scaled)
    # del scaled
    chunk[numeric_cols] = pd.DataFrame(scaled, columns=numeric_cols)
    del scaled #del normalized
    
    chunk = chunk.fillna(0)
    # chunk = chunk.fillna(inplace=True)
    # print(chunk['duration'].value_counts())
    
    chunk.to_csv(output_file, mode='w' if first_chunk else 'a', header=first_chunk, index=False)
    first_chunk = False
    print(f"Index: {index}") #Chunktail: {chunk.tail()}