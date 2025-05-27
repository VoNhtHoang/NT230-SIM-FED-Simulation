import pandas as pd

# Đường dẫn các file
temp_dir = "C:/Users/hoang/FileCSV_DACN_2025/"

files = ["torii_mal_all.csv", "mirai_mal_spread_all.csv", "torii_leg.csv", "bashlite_mal_spread_all.csv", "mirai_mal_CC_all.csv", "mirai_leg.csv", "bashlite_mal_CC_all.csv"]

output = temp_dir + "medbiot.csv"

files = [temp_dir + file for file in files]
labels = ["torii", "mirai_spread", 'benign', 'bashlite_spread', "mirai_c2", 'benign', 'bashlite_c2']
print(files)

chunk_size = 1000
value_threshold = 0.0001  # Ngưỡng loại bỏ

# Hàm lọc dữ liệu xấu
def clean_data(df):
    df = df.dropna()  # Bỏ các dòng có NaN
    # df = df[(df >= value_threshold).all(axis=1)]  # Bỏ dòng có giá trị nhỏ hơn ngưỡng
    return df

# Đọc theo chunk
# iter1 = pd.read_csv(file1, chunksize=chunk_size)
# iter2 = pd.read_csv(file2, chunksize=chunk_size)
iters = [pd.read_csv(file, chunksize=chunk_size) for file in files]
header = True
while True:
    chunks=[]
    for iter in iters:
        try:
            chunk= next(iter)
        except StopIteration:
            chunk= None
        chunks.append(chunk)
    
    if chunks[0] is None and chunks[1] is None and chunks[3] is None and chunks[4] is None and chunks[6] is None:
        break
    for index, chunk in enumerate(chunks):
        if chunk is None:
            continue
        chunk = clean_data(chunk)
        chunk['label'] = labels[index]
        chunk.to_csv(output, mode= 'a', index=False, header=header)
        header=False

# index =0
# for chunk in pd.read_csv(files[3], chunksize=chunk_size):
#     print(index)
#     index +=1