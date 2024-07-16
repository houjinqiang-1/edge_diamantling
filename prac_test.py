def fractional_knapsack(weights, values, capacity):
    # 计算单位价值
    item_values = [(values[i] / weights[i], weights[i], values[i]) for i in range(len(values))]
    # 按单位价值排序
    item_values.sort(reverse=True, key=lambda x: x[0])

    total_value = 0.0  # 背包中物品的总价值
    for value_per_weight, weight, value in item_values:
        if capacity - weight >= 0:
            capacity -= weight
            total_value += value
        else:
            total_value += value_per_weight * capacity
            break

    return total_value


# 示例数据
weights = [10, 20, 30]
values = [60, 100, 120]
capacity = 50

# 调用函数
max_value = fractional_knapsack(weights, values, capacity)
print(f"背包中可以装入的最大价值为: {max_value}")
