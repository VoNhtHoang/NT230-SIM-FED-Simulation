import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
import hashlib
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.pipeline import Pipeline
import ipaddress
import os
import dask.dataframe as dd
# Khởi tạo scaler
scaler = StandardScaler()
normalizer = Normalizer()

# Cấu hình
chunk_size = 1000000
input_files = ["/mnt/c/Users/hoang/FileCSV_DACN_2025/medbiot.csv","C:\\Users\\hoang\\FileCSV_DACN_2025\\medbiot.csv"]
output_files = ["/mnt/c/Users/hoang/FileCSV_DACN_2025/scaled_medbiot.csv", "C:\\Users\\hoang\\FileCSV_DACN_2025\\scaled_medbiot.csv"]

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

df = dd.read_csv(input_file)
cols = df.columns
cols = cols[:len(cols)-1]
print(cols)
del df


# onehot_cols = ['proto', 'service']
label_mapping={}
labels = ['benign', "torii", "mirai_spread", 'bashlite_spread', "mirai_c2", 'bashlite_c2']

for index, label in enumerate(labels):
    label_mapping[label]=index

print("Label Mapping: ", label_mapping)

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

for index, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
    chunk.replace({"(empty)": '0', "-": '0', "":'0', r'[N|n][a|A][N|n]':'0', 'unknown':'0'}, inplace=True)
    chunk.drop_duplicates(inplace=True)
    
    scaler.partial_fit(chunk[cols])  # Fit theo feature #numeric_cols
    
    print(index)

proto_categories = sorted(proto_categories)
service_categories = sorted(service_categories)

first_chunk = True
for index, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
    chunk.replace({"(empty)": '0', "-": '0', "":'0', r'[N|n][a|A][N|n]':'0', 'unknown':'0'}, inplace=True)
    # chunk.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    chunk.drop_duplicates(inplace=True)
    
    # chunk['label'] = chunk['label'].map(label_mapping).astype("int32")
    
        ################ Scaler ###########################
    scaled = scaler.transform(chunk[cols])
    chunk[cols] = pd.DataFrame(scaled, columns=cols)
    del scaled #del normalized

    chunk = chunk.fillna(0)
    
    chunk.to_csv(output_file, mode='w' if first_chunk else 'a', header=first_chunk, index=False)
    # chunk.to_parquet(output_file, engine='pyarrow', index=False)
    first_chunk = False
    print(f"Index: {index}") #Chunktail: {chunk.tail()}