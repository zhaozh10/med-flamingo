import torch

# # 创建原始张量
# input_tensor = torch.tensor(
#     [
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
#         [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4, 0]
#     ]
# )
input_tensor = torch.tensor(
    [
        [0, 0, 2, 0, 0, 0, 4, 0, 0, 6, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4, 0]
    ]
)


def chunk2mask(input_tensor:torch.Tensor):
    INT_MAX=100
    # 创建输出张量并进行元素替换
    output_tensor = input_tensor.clone()
    

    for row in range(output_tensor.size(0)):
        nonzero_indices = torch.nonzero(input_tensor[row], as_tuple=False).flatten()
        if len(nonzero_indices) == 0:
            continue
        nonzero_values = input_tensor[row, nonzero_indices]
        for i in range(1, len(nonzero_indices)):
            output_tensor[row, nonzero_indices[i-1]+1:nonzero_indices[i]+1] = nonzero_values[i]

    # 对每一行处理最后一组零值
    for row in range(output_tensor.size(0)):
        nonzero_indices = torch.nonzero(output_tensor[row], as_tuple=False).flatten()
        if len(nonzero_indices) > 0:
            last_nonzero_index = nonzero_indices[-1]
            
            output_tensor[row, last_nonzero_index+1:] = INT_MAX
            # output_tensor[row, last_nonzero_index+1:] = output_tensor[row, last_nonzero_index] + 1

    # 遍历每一行
    for i in range(output_tensor.shape[0]):
        row = output_tensor[i]
        nonzero_indices = torch.nonzero(row)
        if len(nonzero_indices) > 0:
            first_nonzero_idx = nonzero_indices[0]
            first_nonzero_val = row[first_nonzero_idx]
            # 设置连续的零值为第一个非零值
            for j in range(first_nonzero_idx, 0,-1):
                if row[j] == 0:
                    row[j] = first_nonzero_val

    reverse_tensor=output_tensor.clone()
    for i in range(reverse_tensor.shape[0]):
        row=reverse_tensor[i]
        # 使用unique函数获取不同的值
        unique_values = torch.unique(row)
        # row[torch.where(row==0)]=INT_MAX
        for val in unique_values:
            if val==0:
                continue
                # row[torch.where(row==val)]=INT_MAX
            else:
                row[torch.where(row==val)]=unique_values[unique_values < val].max()
    reverse_tensor[:,0]=INT_MAX
        # for j  in range(1, output_tensor.shape[1]):
        #     cur_val=output_tensor[i][j]
        #     unique_values[unique_values < cur_val]

    
    return output_tensor, reverse_tensor
    



def replace_continuous_zeros(input_tensor):
    # 创建一个与输入张量相同形状的新张量
    output_tensor = torch.zeros_like(input_tensor)

    # 遍历每一行
    for i in range(input_tensor.shape[0]):
        row = input_tensor[i]
        nonzero_indices = torch.nonzero(row)
        if len(nonzero_indices) > 0:
            first_nonzero_idx = nonzero_indices[0]
            first_nonzero_val = row[first_nonzero_idx]
            
            # 设置连续的零值为第一个非零值
            for j in range(first_nonzero_idx, 0,-1):
                if row[j] == 0:
                    row[j] = first_nonzero_val
        
        # 将结果存入输出张量
        output_tensor[i] = row
    return output_tensor


# input_tensor = torch.tensor(
#     [
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 3, 4],
#         [0, 0, 0, 0, 0, 0, 2, 3, 3, 3, 3, 3, 3, 4],
#         [0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4, 4, 5]
#     ]
# )
print(f"input_tensor\n{input_tensor}")
# output_tensor = replace_continuous_zeros(input_tensor)
output_tensor,reverse_tensor = chunk2mask(input_tensor)

print(f"output tensor\n{output_tensor}")
print(f"reverse tensor\n{reverse_tensor}")



# output_tensor=chunk2mask(input_tensor=input_tensor)
