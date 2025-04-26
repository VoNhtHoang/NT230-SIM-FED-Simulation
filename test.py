import csv
import re
import dask.dataframe as dd

df = dd.r
input_file = filesPath[0]
output_file = '/mnt/c/Users/hoang/FileCSV_DACN_2025/opt/log_1.csv'
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    writer = None
    for line in infile:
        line = line.strip()
        if not line or line.startswith('#'):
            # Lấy tiêu đề từ dòng "#fields"
            if line.startswith("#fields"):
                headers = re.split(r'[\t ]+', line)[1:]  # Bỏ '#fields'
                writer = csv.writer(outfile)
                writer.writerow(headers)
            continue
        # Tách dòng dữ liệu theo tab
        fields =re.split(r'[\t ]+', line)
        print(fields)
        # Xử lý giá trị đặc 
        # for f in fields:
        #     if f 
        # fields = ["" if f == "(empty)" else None if f == "-" else f for f in fields]

        # Ghi ra file CSV
        d

print(f'Đã chuyển đổi {input_file} thành {output_file}')

# # 1. Lấy header từ dòng "#fields"
# fields = []
# with open(input_file, "r") as f:
#     for line in f:
#         if line.startswith("#fields"):
#             fields = re.split(r'[\t ]+',line)[1:]  # Bỏ '#fields'
#             break
# # fields[-1] = fields[-1].replace('\n', '')
# print(fields)

# # 2. Đọc file bỏ qua các dòng "#"
# df = dd.read_csv(
#     input_file,
#     sep=r'[\t ]+',
#     comment='#',
#     names=fields,
#     engine='python',
#     blocksize=None
# )
# print(df)

# df.to_csv(output_file, mode='a', index=False,single_file=True)


# headers = []
# data = []
# with open(input_file, 'r')  as infile:
#     writer = None
#     for line in infile:
#         line = line.strip()
#         if not line or line.startswith('#'):
#             # Lấy tiêu đề từ dòng "#fields"
#             if line.startswith("#fields"):
#                 headers = re.split(r'[\t ]+', line)[1:]  # Bỏ '#fields'
#                 # writer = csv.writer(outfile)
#                 # writer.writerow(headers)
#             continue

#         fields =re.split(r'[\t ]+', line)
#         data.append(fields)
#         # print(fields)
        
# print(headers)
# df = pd.DataFrame(data, columns=headers)
# print(df)

# df.to_csv(output_file, index=False)

# 1. Lấy header từ dòng "#fields"